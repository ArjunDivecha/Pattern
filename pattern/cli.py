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
import hashlib
import json
import logging
import os
import random
import sys
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


def cmd_train(cfg: Config) -> None:
    run_dir = _make_run_dir(Path(cfg.output_dir), cfg)
    log.info(f"Run directory: {run_dir}")

    # Save resolved config + reproducibility artifacts (PRD §7)
    import subprocess, yaml
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg.model_dump(mode="json"), f, default_flow_style=False)
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_sha = "unavailable"
    with open(run_dir / "git_sha.txt", "w") as f:
        f.write(git_sha + "\n")
    try:
        pip_freeze = subprocess.check_output(
            ["pip", "freeze"], stderr=subprocess.DEVNULL).decode()
        with open(run_dir / "pip_freeze.txt", "w") as f:
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

    for window_idx, split in enumerate(splits):
        log.info(f"\n{'='*60}\nRetraining window {window_idx+1}/{len(splits)}\n{'='*60}")

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

        window_preds = []
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

            probs, _ = predict(model, test_loader, cfg.train.device)
            window_preds.append(probs[:, 1].numpy())

        # Assemble predictions for this window's test set
        test_meta = index_df.iloc[test_idx][["ticker", "end_date", "forward_return", "label"]].reset_index(drop=True)
        for k, p in enumerate(window_preds):
            test_meta[f"p_up_{k}"] = p
        p_up_cols = [c for c in test_meta.columns if c.startswith("p_up_")]
        test_meta["p_up_mean"] = test_meta[p_up_cols].mean(axis=1)
        test_meta["p_up_std"]  = test_meta[p_up_cols].std(axis=1)
        test_meta["window"]    = window_idx
        all_test_predictions.append(test_meta)

    # ── Combine all windows and save ─────────────────────────────────────────
    pred_df  = pd.concat(all_test_predictions, ignore_index=True)
    out_path = run_dir / "predictions.parquet"
    pred_df.to_parquet(out_path, index=False)
    log.info(f"Predictions saved: {out_path}")

    # Quick AUC check
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(pred_df["label"], pred_df["p_up_mean"])
        log.info(f"Test AUC: {auc:.4f}")
    except ImportError:
        pass

    log.info(f"Run complete: {run_dir}")


def cmd_backtest(run_dir: Path, cfg: Config) -> None:
    """Run the decile/long-short backtest against an existing run's predictions."""
    from pattern.backtest.report import run_backtest
    holding = cfg.backtest.holding_period_days or cfg.label.horizon
    run_backtest(run_dir, cfg.backtest, holding_period_days=holding)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pattern CNN pipeline")
    sub    = parser.add_subparsers(dest="cmd")

    train_p = sub.add_parser("train")
    train_p.add_argument("--config", default="configs/debug.yaml")

    bt_p = sub.add_parser("backtest")
    bt_p.add_argument("--config", default="configs/debug.yaml")
    bt_p.add_argument("--run-dir", required=True,
                      help="Run directory containing predictions parquet")

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = Config.from_yaml(args.config)
        cmd_train(cfg)
    elif args.cmd == "backtest":
        cfg = Config.from_yaml(args.config)
        cmd_backtest(Path(args.run_dir), cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
