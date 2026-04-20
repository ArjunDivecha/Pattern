"""
=============================================================================
SCRIPT NAME: cli.py
=============================================================================
DESCRIPTION:
Command-line entry point: python -m pattern train --config configs/debug.yaml

Phase 1 pipeline:
  1. Load OHLCV data
  2. Compute labels
  3. Apply splits
  4. Balance train/val labels
  5. Build image cache (or load if exists)
  6. Compute pixel normalisation stats
  7. Train a single model (Phase 1) or ensemble (Phase 2+)
  8. Dump predictions parquet to run directory
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────────────────────
from pattern.config import Config
from pattern.data.loader import compute_labels, load_data
from pattern.data.splits import balance_labels, get_splits
from pattern.imaging.cache import build_cache, compute_pixel_stats, load_cache
from pattern.models.cnn import ChartCNN
from pattern.train.dataset import CachedDataset
from pattern.train.loop import predict, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_run_dir(output_dir: Path, cfg: Config) -> Path:
    cfg_str  = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, default=str)
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = output_dir / f"{stamp}_{cfg_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def cmd_train(
    cfg: Config,
    window_indices: list[int] | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Train one pathway. Supports sharded multi-GPU runs:

    * `window_indices`  — subset of window IDs this process should handle;
                          None means all windows.
    * `run_dir`         — pre-created shared run dir (for multi-GPU shards).
                          None means create a fresh timestamped dir.

    GPU selection is done by the caller (`CUDA_VISIBLE_DEVICES=N python -m ...`)
    so each shard sees exactly one device as `cuda:0`.
    """
    if run_dir is None:
        run_dir = _make_run_dir(Path(cfg.output_dir), cfg)
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Run directory: {run_dir}")

    # Save resolved config + reproducibility artifacts (PRD §7).
    # Only the first shard writes these — subsequent shards see the files and skip.
    import subprocess, yaml
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        with open(cfg_path, "w") as f:
            yaml.dump(cfg.model_dump(mode="json"), f, default_flow_style=False)
    sha_path = run_dir / "git_sha.txt"
    if not sha_path.exists():
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent,
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_sha = "unavailable"
        with open(sha_path, "w") as f:
            f.write(git_sha + "\n")
    pip_path = run_dir / "pip_freeze.txt"
    if not pip_path.exists():
        try:
            pip_freeze = subprocess.check_output(
                ["pip", "freeze"], stderr=subprocess.DEVNULL).decode()
            with open(pip_path, "w") as f:
                f.write(pip_freeze)
        except Exception:
            pass

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_data(cfg.data.csv_path, cfg.data.date_format, cfg.data.min_history_days)

    # ── 2. Labels ─────────────────────────────────────────────────────────────
    labelled = compute_labels(df, cfg.label.horizon)

    # ── 3. Splits ─────────────────────────────────────────────────────────────
    # Purge same-ticker overlap: a train row on date T produces a window covering
    # [T - window + 1 … T] with label from T+horizon. A val row whose date is within
    # (window + horizon - 1) days of T has an overlapping image or horizon.
    purge_days = cfg.image.window + cfg.label.horizon - 1
    splits = get_splits(labelled, cfg.split, purge_days=purge_days)
    log.info(f"Total retraining windows: {len(splits)}  purge_days={purge_days}")

    # ── 4. Image cache (built once for all splits) ────────────────────────────
    cache_dir = Path(cfg.image.cache_dir)
    build_cache(labelled, cfg.image, cfg.label, cache_dir)
    images, index_df = load_cache(cache_dir)

    # Vectorised key index for fast split → cache mapping
    index_df["_key"] = index_df["ticker"] + "|" + index_df["end_date"].astype(str)
    index_df["_pos"] = np.arange(len(index_df), dtype=np.int64)

    def _to_cache_idx(split_df: pd.DataFrame) -> np.ndarray:
        keys    = (split_df["Ticker"] + "|" + split_df["Date"].astype(str)).values
        matched = index_df[index_df["_key"].isin(pd.Index(keys))]
        return matched["_pos"].to_numpy(dtype=np.int64)

    # ── 5–8. Loop over all retraining windows ────────────────────────────────
    all_test_predictions = []

    # window_stats.csv is an append-only log; multiple shards write into it concurrently,
    # so only create the header if absent.
    stats_path = run_dir / "window_stats.csv"
    if not stats_path.exists():
        with open(stats_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "window_idx", "train_years", "val_years", "test_years",
                "n_train", "n_val", "n_test", "ensemble_size",
                "wall_seconds", "gpu_peak_mem_gb",
            ])

    subset = set(window_indices) if window_indices is not None else None
    for window_idx, split in enumerate(splits):
        if subset is not None and window_idx not in subset:
            continue
        log.info(f"\n{'='*60}\nRetraining window {window_idx+1}/{len(splits)}\n{'='*60}")
        window_t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_df = split["train"]
        val_df   = split["val"]
        test_df  = split["test"]

        # Balance train only (val + test left at their natural distribution
        # so val metrics reflect real-world class imbalance — PRD §5).
        if cfg.label.balance_train:
            train_df = balance_labels(train_df, seed=window_idx)

        train_idx = _to_cache_idx(train_df)
        val_idx   = _to_cache_idx(val_df)
        test_idx  = _to_cache_idx(test_df)
        log.info(f"Cache indices — train:{len(train_idx):,}  val:{len(val_idx):,}  test:{len(test_idx):,}")

        # Pixel stats keyed by train split (per PRD §5 + Codex review)
        pix_mean, pix_std = compute_pixel_stats(cache_dir, train_idx)

        window_probs, window_logits, window_embs = [], [], []
        for k in range(cfg.train.ensemble_size):
            seed = cfg.train.seeds[k] if k < len(cfg.train.seeds) else k
            _seed_everything(seed)

            train_ds = CachedDataset(images, index_df, train_idx, pix_mean, pix_std)
            val_ds   = CachedDataset(images, index_df, val_idx,   pix_mean, pix_std)
            test_ds  = CachedDataset(images, index_df, test_idx,  pix_mean, pix_std)

            # pin_memory only helps CUDA H2D transfers; on MPS/CPU it is a
            # no-op at best and a source of sporadic non-finite batches at worst.
            pin = cfg.train.device == "cuda" or (
                cfg.train.device == "auto" and torch.cuda.is_available()
            )
            train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                                      shuffle=True,  num_workers=cfg.train.num_workers, pin_memory=pin)
            val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size,
                                      shuffle=False, num_workers=cfg.train.num_workers, pin_memory=pin)
            test_loader  = DataLoader(test_ds,  batch_size=cfg.train.batch_size,
                                      shuffle=False, num_workers=cfg.train.num_workers, pin_memory=pin)

            model_name = f"w{window_idx:02d}_ensemble_{k}"
            model = ChartCNN(cfg.model, cfg.image)
            model, _ = train_model(model, train_loader, val_loader,
                                   cfg.train, run_dir, model_name, seed)

            probs, _, logits, embs = predict(
                model, test_loader, cfg.train.device, return_features=True,
            )
            window_probs.append(probs[:, 1].numpy().astype(np.float32))
            window_logits.append(logits.numpy().astype(np.float32))
            window_embs.append(embs.numpy().astype(np.float32))

        # Assemble predictions for this window's test set
        test_meta = index_df.iloc[test_idx][
            ["ticker", "end_date", "forward_return", "label"]
        ].reset_index(drop=True)
        for k, p in enumerate(window_probs):
            test_meta[f"p_up_{k}"] = p
        p_up_cols = [f"p_up_{k}" for k in range(len(window_probs))]
        test_meta["p_up_mean"] = test_meta[p_up_cols].mean(axis=1)
        test_meta["p_up_std"]  = test_meta[p_up_cols].std(axis=1)
        # Ensemble-mean logits (useful for calibration / analysis)
        logit_stack = np.stack(window_logits, axis=0)      # (K, N, 2)
        test_meta["logit_down_mean"] = logit_stack[:, :, 0].mean(axis=0)
        test_meta["logit_up_mean"]   = logit_stack[:, :, 1].mean(axis=0)

        # Cross-sectional rank/decile per formation date (ticker-level analysis)
        test_meta["rank_pct"] = (
            test_meta.groupby("end_date")["p_up_mean"]
                     .rank(method="average", pct=True)
                     .astype(np.float32)
        )
        test_meta["decile"] = (
            test_meta.groupby("end_date")["p_up_mean"]
                     .transform(lambda s: pd.qcut(
                         s.rank(method="first"),
                         q=cfg.backtest.n_deciles,
                         labels=False,
                         duplicates="drop",
                     ))
                     .astype("Int8")
        )
        test_meta["window"] = np.int16(window_idx)
        all_test_predictions.append(test_meta)
        test_meta.to_parquet(run_dir / f"window_{window_idx:02d}_predictions.parquet",
                             index=False)

        # Per-window embeddings + raw logits — the heavy artifacts go in their
        # own file per window so the main predictions parquet stays compact.
        emb_stack   = np.stack(window_embs, axis=0)        # (K, N, C)
        np.savez_compressed(
            run_dir / f"window_{window_idx:02d}_features.npz",
            ticker         = test_meta["ticker"].to_numpy(),
            end_date       = test_meta["end_date"].astype("datetime64[ns]").to_numpy(),
            logits         = logit_stack,                  # (K, N, 2) fp32
            embeddings     = emb_stack,                    # (K, N, C) fp32
            embedding_mean = emb_stack.mean(axis=0),       # (N, C)    fp32
            window_idx     = np.int32(window_idx),
        )

        wall = time.time() - window_t0
        peak_gb = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else 0.0
        )
        log.info(
            f"Window {window_idx} done  wall={wall:.1f}s  peak_gpu={peak_gb:.2f}GB  "
            f"test_n={len(test_meta):,}"
        )
        with open(stats_path, "a", newline="") as f:
            csv.writer(f).writerow([
                window_idx,
                int(pd.to_datetime(train_df["Date"]).dt.year.nunique()),
                int(pd.to_datetime(val_df["Date"]).dt.year.nunique()) if len(val_df) else 0,
                int(pd.to_datetime(test_df["Date"]).dt.year.nunique()) if len(test_df) else 0,
                len(train_df), len(val_df), len(test_df),
                cfg.train.ensemble_size,
                round(wall, 1), round(peak_gb, 3),
            ])

    # ── Combine all windows and save ─────────────────────────────────────────
    # Shard runs leave the final concat to the driver (which also sees the
    # other shards' per-window parquets). A full-pathway run writes the combined
    # predictions.parquet here.
    if subset is None:
        pred_df  = pd.concat(all_test_predictions, ignore_index=True)
        out_path = run_dir / "predictions.parquet"
        pred_df.to_parquet(out_path, index=False)
        log.info(f"Predictions saved: {out_path}")
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(pred_df["label"], pred_df["p_up_mean"])
            log.info(f"Test AUC: {auc:.4f}")
        except ImportError:
            pass
    else:
        log.info(f"Shard complete — {len(all_test_predictions)} windows; "
                 f"final concat deferred to driver")

    log.info(f"Run complete: {run_dir}")
    return run_dir


def cmd_backtest(run_dir: Path, cfg: Config) -> None:
    """Run the decile/long-short backtest against an existing run's predictions."""
    from pattern.backtest.report import run_backtest
    holding = cfg.backtest.holding_period_days or cfg.label.horizon
    run_backtest(run_dir, cfg.backtest, holding_period_days=holding)


def _parse_window_indices(spec: str) -> list[int]:
    """Parse "0,3,5-9" into [0,3,5,6,7,8,9]."""
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pattern CNN pipeline")
    sub    = parser.add_subparsers(dest="cmd")

    train_p = sub.add_parser("train")
    train_p.add_argument("--config", default="configs/debug.yaml")
    train_p.add_argument("--window-indices", default=None,
                         help="Comma/range spec of windows to process (e.g. '0,3,5-9'). "
                              "Omit to train all windows.")
    train_p.add_argument("--run-dir", default=None,
                         help="Reuse an existing run directory (used by multi-GPU driver "
                              "so shards share a single output dir).")

    bt_p = sub.add_parser("backtest")
    bt_p.add_argument("--config", default="configs/debug.yaml")
    bt_p.add_argument("--run-dir", required=True,
                      help="Run directory containing predictions parquet")

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = Config.from_yaml(args.config)
        win_idx = _parse_window_indices(args.window_indices) if args.window_indices else None
        run_dir = Path(args.run_dir) if args.run_dir else None
        cmd_train(cfg, window_indices=win_idx, run_dir=run_dir)
    elif args.cmd == "backtest":
        cfg = Config.from_yaml(args.config)
        cmd_backtest(Path(args.run_dir), cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
