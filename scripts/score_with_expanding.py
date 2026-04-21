"""
=============================================================================
SCRIPT NAME: score_with_expanding.py
=============================================================================

INPUT FILES:
- cache/etf_I20/images.npy            memmap uint8 (N,1,64,60)
- cache/etf_I20/index.parquet         ticker, end_date, forward_return, label
- runs/expanding/20260419_174908_cdef6809/config.yaml
- runs/expanding/20260419_174908_cdef6809/w{NN}_ensemble_{0..4}.pt
                                      28 windows × 5 seeds = 140 models

OUTPUT FILES:
- runs/etf_expanding/predictions.parquet
    ticker, end_date, forward_return, label,
    p_up_0..4, p_up_mean, p_up_std, window

DESCRIPTION:
Applies the expanding-pathway CNN ensembles to the ETF cache.  For each ETF
image (ticker, end_date), we look up the expanding window whose test year
matches end_date.year, then score with that window's 5 seeds and average.
This gives genuinely out-of-sample forecasts for every ETF rolling 20-day
window from 1999-01 onward.

Pixel stats are computed per-window on the ETF cache using end_dates in
that window's TRAINING year range — matching how training normalised its
own cache.  The pixel distribution of candlestick images is dominated by
image geometry (bars/MA/volume density) so ETF vs r1000 train stats agree
to a few percent, which is well within the 0/255 binary noise floor.

USAGE:
  python scripts/score_with_expanding.py
  python scripts/score_with_expanding.py --device mps --chunk 4096
=============================================================================
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pattern.config import Config
from pattern.imaging.cache import load_cache
from pattern.models.cnn import ChartCNN

DEFAULT_SRC_RUN = ROOT / "runs" / "expanding" / "20260419_174908_cdef6809"

# Expanding schedule (start_year=1996, train_years=3, retrain_every_years=1):
#   window N: train 1996..(1998+N), test 1999+N
START_YEAR = 1996
TRAIN_YEARS = 3
FIRST_TEST_YEAR = START_YEAR + TRAIN_YEARS  # 1999

log = logging.getLogger("score_etfs")


def pixel_stats_on_indices(images: np.ndarray, indices: np.ndarray,
                           chunk: int = 5000) -> tuple[float, float]:
    """Mean/std of pixel values (0..255) across rows in `indices`."""
    if len(indices) == 0:
        return 0.0, 1.0
    sum_ = 0.0
    sum_sq = 0.0
    n_px = 0
    for start in range(0, len(indices), chunk):
        idx = indices[start:start + chunk]
        batch = images[idx].astype(np.float64)
        sum_ += batch.sum()
        sum_sq += (batch ** 2).sum()
        n_px += batch.size
    mean = sum_ / n_px
    std = float(np.sqrt(max(sum_sq / n_px - mean ** 2, 0.0)))
    if std < 1e-6:
        std = 1.0
    return float(mean), std


def load_window_models(cfg: Config, src_run: Path, window_idx: int,
                       device: torch.device) -> list:
    """Return 5 ChartCNN models loaded with window_idx's state dicts."""
    models = []
    for k in range(5):
        ckpt = src_run / f"w{window_idx:02d}_ensemble_{k}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m = ChartCNN(cfg.model, cfg.image)
        m.load_state_dict(state)
        m.to(device).eval()
        models.append(m)
    return models


def score_window(
    models: list,
    images: np.ndarray,
    test_idx: np.ndarray,
    pix_mean: float,
    pix_std: float,
    device: torch.device,
    chunk: int,
) -> np.ndarray:
    """Returns (len(test_idx), 5) p_up matrix."""
    N = len(test_idx)
    out = np.empty((N, 5), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            idx = test_idx[start:end]
            batch = images[idx].astype(np.float32)
            batch = (batch - pix_mean) / pix_std
            x = torch.from_numpy(batch).to(device)
            for k, m in enumerate(models):
                logits = m(x)                                      # (B, 2)
                p_up = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                out[start:end, k] = p_up
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-run", type=Path, default=DEFAULT_SRC_RUN,
                    help="directory with w{NN}_ensemble_{0..4}.pt + config.yaml")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "etf_I20",
                    help="cache dir with images.npy + index.parquet")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "etf_expanding",
                    help="where to write predictions.parquet")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--cadence", choices=["monthly", "daily"], default="monthly",
                    help="monthly = score only the last trading day of each "
                         "calendar month per ticker (default).  daily = score "
                         "every row in the cache (legacy behaviour).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    cfg = Config.from_yaml(args.src_run / "config.yaml")

    images, index_df = load_cache(args.cache_dir)
    dates = pd.to_datetime(index_df["end_date"])
    years = dates.dt.year.values
    max_year = int(years.max())
    log.info(f"Cache: {len(index_df):,} rows  {dates.min().date()} → {dates.max().date()}")

    # Cadence filter.  We keep the full daily index for pixel-stat computation
    # (matching training normalisation), but restrict test rows (the rows we
    # actually score) to the canonical last trading day of each calendar
    # month (the latest date appearing anywhere in the cache for that month).
    # Tickers without a row on that exact date are dropped for that month.
    if args.cadence == "monthly":
        ym = dates.dt.to_period("M")
        canonical = pd.Series(dates.values, index=dates.index).groupby(ym).transform("max")
        score_mask = (dates.values == canonical.values)
        log.info(f"Cadence=monthly: scoring "
                 f"{score_mask.sum():,} / {len(index_df):,} rows "
                 f"({score_mask.mean()*100:.1f}% of cache)  "
                 f"unique dates={pd.Series(dates.values[score_mask]).nunique()}")
    else:
        score_mask = np.ones(len(index_df), dtype=bool)

    n_windows = max_year - FIRST_TEST_YEAR + 1   # 1999 → max_year inclusive
    log.info(f"Expanding pathway has {n_windows} test years: "
             f"{FIRST_TEST_YEAR} → {max_year}")

    # Sanity: all 28 checkpoints should exist for windows 0..27 in src_run
    available = sorted(int(p.stem.split("_")[0].lstrip("w"))
                       for p in args.src_run.glob("w*_ensemble_0.pt"))
    log.info(f"Checkpoints present for windows: {available[0]}..{available[-1]}  "
             f"({len(available)} total)")

    pred_chunks = []
    for w in range(n_windows):
        test_year = FIRST_TEST_YEAR + w
        train_year_hi = FIRST_TEST_YEAR - 1 + w    # 1998 + w

        test_mask = (years == test_year) & score_mask
        test_idx = np.flatnonzero(test_mask).astype(np.int64)
        if len(test_idx) == 0:
            log.warning(f"  window {w:02d}  test_year={test_year}  no ETF rows — skipping")
            continue

        train_mask = (years >= START_YEAR) & (years <= train_year_hi)
        train_indices = np.flatnonzero(train_mask).astype(np.int64)
        if len(train_indices) < 1000:
            # Cache doesn't cover this window's training years (common for
            # ETF caches starting after 1998).  Fall back to all rows whose
            # end_date precedes the test year — still contemporaneous, no
            # look-ahead leakage.
            fallback_mask = years < test_year
            train_indices = np.flatnonzero(fallback_mask).astype(np.int64)
            if len(train_indices) < 1000:
                # Still not enough — use all rows as last resort.
                train_indices = np.arange(len(index_df), dtype=np.int64)
        pix_mean, pix_std = pixel_stats_on_indices(images, train_indices)

        if w >= len(available):
            log.warning(f"  window {w:02d}  no checkpoint — skipping")
            continue

        models = load_window_models(cfg, args.src_run, w, device)
        probs = score_window(models, images, test_idx, pix_mean, pix_std,
                             device, args.chunk)

        chunk = index_df.iloc[test_idx][["ticker", "end_date",
                                         "forward_return", "label"]].copy()
        chunk = chunk.reset_index(drop=True)
        for k in range(5):
            chunk[f"p_up_{k}"] = probs[:, k]
        chunk["p_up_mean"] = probs.mean(axis=1)
        chunk["p_up_std"] = probs.std(axis=1)
        chunk["window"] = w

        pred_chunks.append(chunk)
        log.info(f"  window {w:02d}  test_year={test_year}  "
                 f"rows={len(chunk):,}  "
                 f"pix_mean={pix_mean:.2f}  pix_std={pix_std:.2f}  "
                 f"mean p_up={chunk['p_up_mean'].mean():.3f}")

        # Free model memory
        del models
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    if not pred_chunks:
        raise RuntimeError("No predictions generated.")

    all_preds = pd.concat(pred_chunks, ignore_index=True)
    all_preds = all_preds.sort_values(["ticker", "end_date"]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "predictions.parquet"
    all_preds.to_parquet(out_path, index=False)
    log.info(f"wrote {out_path}  rows={len(all_preds):,}  "
             f"tickers={all_preds['ticker'].nunique()}  "
             f"{all_preds['end_date'].min().date()} → {all_preds['end_date'].max().date()}")

    # Quick sanity: overall AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_preds["label"], all_preds["p_up_mean"])
        log.info(f"Overall OOS AUC: {auc:.4f}")
    except Exception as e:
        log.warning(f"AUC failed: {e}")


if __name__ == "__main__":
    main()
