"""
=============================================================================
SCRIPT NAME: extract_embeddings.py
=============================================================================

INPUT FILES:
- cache/r1000_I20_monthly/images.npy + index.parquet
- runs/expanding/<run>/w{WW}_ensemble_{0..4}.pt + config.yaml
- runs/expanding/<run>/predictions_monthly.parquet  (universe + window ids)

OUTPUT FILES:
- {out_dir}/embeddings.npz
    ticker, end_date, window, forward_return, label, emb (N, 256) float32 L2-normed
- {out_dir}/embeddings_meta.json

DESCRIPTION:
Walk-forward penultimate embeddings (ChartCNN GAP, 256-d) for month-end
formation dates used in Pattern predictions_monthly. Each OOS year is
embedded only by that expanding window's 5-seed ensemble. Pixel mean/std
are computed on cache rows from training years (no look-ahead), matching
scripts/score_with_expanding.py.

PRD: PRD_Visual_Analog_KNN.md §5.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pattern.config import Config
from pattern.models.cnn import ChartCNN

log = logging.getLogger("analog_knn.extract")

START_YEAR = 1996
TRAIN_YEARS = 3
FIRST_TEST_YEAR = START_YEAR + TRAIN_YEARS  # 1999
DEFAULT_EXPANDING = ROOT / "runs" / "expanding" / "20260419_174908_cdef6809"
DEFAULT_CACHE = ROOT / "cache" / "r1000_I20_monthly"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def pixel_stats_on_indices(
    images: np.ndarray, indices: np.ndarray, chunk: int = 5000
) -> tuple[float, float]:
    if len(indices) == 0:
        return 0.0, 1.0
    sum_ = 0.0
    sum_sq = 0.0
    n_px = 0
    for start in range(0, len(indices), chunk):
        idx = indices[start : start + chunk]
        batch = images[idx].astype(np.float64)
        sum_ += batch.sum()
        sum_sq += (batch ** 2).sum()
        n_px += batch.size
    mean = sum_ / n_px
    std = float(np.sqrt(max(sum_sq / n_px - mean ** 2, 0.0)))
    return float(mean), float(std if std >= 1e-6 else 1.0)


def load_window_models(
    cfg: Config, src_run: Path, window_idx: int, device: torch.device
) -> list[ChartCNN]:
    models: list[ChartCNN] = []
    for k in range(5):
        ckpt = src_run / f"w{window_idx:02d}_ensemble_{k}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m = ChartCNN(cfg.model, cfg.image)
        m.load_state_dict(state)
        m.to(device).eval()
        models.append(m)
    return models


def embed_window(
    models: list[ChartCNN],
    images: np.ndarray,
    row_idx: np.ndarray,
    pix_mean: float,
    pix_std: float,
    device: torch.device,
    chunk: int,
) -> np.ndarray:
    """Return (N, 256) ensemble-mean of L2-normalized seed embeddings."""
    n = len(row_idx)
    acc = np.zeros((n, 256), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            idx = row_idx[start:end]
            batch = images[idx].astype(np.float32)
            batch = (batch - pix_mean) / max(pix_std, 1e-6)
            x = torch.from_numpy(batch).to(device)
            seed_sum = None
            for m in models:
                _logits, emb = m.forward_with_features(x)
                emb = emb.float()
                emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
                seed_sum = emb if seed_sum is None else seed_sum + emb
            mean_emb = seed_sum / len(models)
            mean_emb = mean_emb / mean_emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
            acc[start:end] = mean_emb.cpu().numpy()
    # re-normalize in float32 for storage
    out = acc.astype(np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-8)
    return out / norms


def build_target_panel(
    cache_index: pd.DataFrame,
    predictions_monthly: Path | None,
) -> pd.DataFrame:
    """
    Rows to embed: inner join of monthly predictions with cache when available,
    else month-end rows of the cache (canonical max date per calendar month).
    """
    idx = cache_index.copy()
    idx["end_date"] = pd.to_datetime(idx["end_date"])
    idx["cache_idx"] = np.arange(len(idx), dtype=np.int64)

    if predictions_monthly is not None and predictions_monthly.exists():
        pred = pd.read_parquet(predictions_monthly)
        pred["end_date"] = pd.to_datetime(pred["end_date"])
        keep = ["ticker", "end_date", "forward_return", "label", "window"]
        pred = pred[[c for c in keep if c in pred.columns]].copy()
        merged = pred.merge(
            idx[["ticker", "end_date", "cache_idx"]],
            on=["ticker", "end_date"],
            how="inner",
        )
        log.info(
            "Joined predictions_monthly ∩ cache: %s / %s pred rows",
            f"{len(merged):,}",
            f"{len(pred):,}",
        )
        if len(merged) == 0:
            raise RuntimeError("No overlap between predictions_monthly and image cache")
        return merged.reset_index(drop=True)

    # Fallback: last trading day of each calendar month present in cache
    dates = idx["end_date"]
    ym = dates.dt.to_period("M")
    canonical = dates.groupby(ym).transform("max")
    mask = dates.values == canonical.values
    out = idx.loc[mask].copy()
    out["window"] = (out["end_date"].dt.year - FIRST_TEST_YEAR).astype(np.int16)
    log.info("No predictions_monthly; using cache month-ends: %s rows", f"{len(out):,}")
    return out.reset_index(drop=True)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="Extract Pattern CNN embeddings for analog kNN")
    ap.add_argument("--expanding-dir", type=Path, default=DEFAULT_EXPANDING)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument(
        "--predictions-monthly",
        type=Path,
        default=DEFAULT_EXPANDING / "predictions_monthly.parquet",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Debug: only first N expanding windows",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
    )
    device = resolve_device(args.device)
    log.info("Device: %s", device)

    cfg = Config.from_yaml(args.expanding_dir / "config.yaml")
    images_path = args.cache_dir / "images.npy"
    index_path = args.cache_dir / "index.parquet"
    if not images_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"Cache incomplete under {args.cache_dir}")

    images = np.load(images_path, mmap_mode="r")
    cache_index = pd.read_parquet(index_path)
    log.info("Cache images %s  index %s", images.shape, f"{len(cache_index):,}")

    panel = build_target_panel(cache_index, args.predictions_monthly)
    panel["end_date"] = pd.to_datetime(panel["end_date"])
    if "window" not in panel.columns:
        panel["window"] = (panel["end_date"].dt.year - FIRST_TEST_YEAR).astype(np.int16)
    panel["window"] = panel["window"].astype(np.int16)

    years_cache = pd.to_datetime(cache_index["end_date"]).dt.year.values

    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows = sorted(int(w) for w in panel["window"].unique())
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
        panel = panel[panel["window"].isin(windows)].copy()

    emb_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []

    for w in windows:
        sub = panel[panel["window"] == w]
        if len(sub) == 0:
            continue
        test_year = FIRST_TEST_YEAR + w
        train_year_hi = FIRST_TEST_YEAR - 1 + w
        train_mask = (years_cache >= START_YEAR) & (years_cache <= train_year_hi)
        train_indices = np.flatnonzero(train_mask).astype(np.int64)
        if len(train_indices) < 1000:
            train_indices = np.flatnonzero(years_cache < test_year).astype(np.int64)
        if len(train_indices) < 1000:
            train_indices = np.arange(len(cache_index), dtype=np.int64)
        pix_mean, pix_std = pixel_stats_on_indices(images, train_indices)

        ckpt0 = args.expanding_dir / f"w{w:02d}_ensemble_0.pt"
        if not ckpt0.exists():
            log.warning("Missing checkpoints for window %02d — skip", w)
            continue

        models = load_window_models(cfg, args.expanding_dir, w, device)
        row_idx = sub["cache_idx"].to_numpy(dtype=np.int64)
        emb = embed_window(
            models, images, row_idx, pix_mean, pix_std, device, args.chunk
        )
        emb_parts.append(emb)
        meta_parts.append(sub.reset_index(drop=True))
        log.info(
            "window %02d  test_year=%d  n=%s  pix_mean=%.2f pix_std=%.2f  "
            "‖e‖ mean=%.4f",
            w,
            test_year,
            f"{len(sub):,}",
            pix_mean,
            pix_std,
            float(np.linalg.norm(emb, axis=1).mean()),
        )
        del models
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    if not emb_parts:
        raise RuntimeError("No embeddings extracted")

    emb_all = np.vstack(emb_parts).astype(np.float32)
    meta = pd.concat(meta_parts, ignore_index=True)
    assert len(meta) == len(emb_all)

    out_npz = args.out_dir / "embeddings.npz"
    np.savez_compressed(
        out_npz,
        ticker=meta["ticker"].to_numpy(),
        end_date=meta["end_date"].astype("datetime64[ns]").to_numpy(),
        window=meta["window"].to_numpy(dtype=np.int16),
        forward_return=meta["forward_return"].to_numpy(dtype=np.float64),
        label=meta["label"].to_numpy(dtype=np.int8),
        emb=emb_all,
    )
    meta_path = args.out_dir / "embeddings_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "n_rows": int(len(meta)),
                "dim": 256,
                "ensemble": "mean_of_l2_normalized_seeds",
                "expanding_dir": str(args.expanding_dir),
                "cache_dir": str(args.cache_dir),
                "windows": windows,
                "date_min": str(meta["end_date"].min().date()),
                "date_max": str(meta["end_date"].max().date()),
                "device": str(device),
            },
            indent=2,
        )
    )
    log.info("Wrote %s  (%s rows)", out_npz, f"{len(meta):,}")
    return out_npz


if __name__ == "__main__":
    main()
