"""
=============================================================================
SCRIPT NAME: cache.py
=============================================================================
INPUT FILES:
- OHLCV DataFrame (from loader.py) with labelled samples
OUTPUT FILES:
- {cache_dir}/images.npy   — memmap uint8 (N, 1, H, W)
- {cache_dir}/index.parquet — (ticker, end_date, forward_return, label)

DESCRIPTION:
Builds and loads the memmap image cache (PRD §4.4).  Building is
parallelised across tickers using all available CPU cores.
=============================================================================
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hashlib

import numpy as np
import pandas as pd

from pattern.config import ImageConfig, LabelConfig
from pattern.data.loader import build_ticker_index, get_window
from pattern.imaging.renderer import render_window

log = logging.getLogger(__name__)

INDEX_FILE  = "index.parquet"
IMAGES_FILE = "images.npy"
STATS_FILE  = "pixel_stats.npz"


# ---------------------------------------------------------------------------
# Worker (top-level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------

def _worker_init(ticker_index_dict, img_cfg_dict, lbl_cfg_dict):
    global _TI, _ICFG, _LCFG
    _TI   = ticker_index_dict
    _ICFG = img_cfg_dict
    _LCFG = lbl_cfg_dict


def _process_ticker(ticker: str) -> List[dict]:
    """Generate all valid (image, label) records for one ticker."""
    tdf  = _TI[ticker]
    rows = []

    window   = _ICFG["window"]
    height   = _ICFG["height"]
    width    = _ICFG["width"]
    ratio    = _ICFG["ohlc_height_ratio"]
    inc_ma   = _ICFG["include_ma"]
    inc_vol  = _ICFG["include_volume"]
    horizon  = _LCFG["horizon"]
    lookback = window - 1   # extra rows for full MA lookback

    for i in range(len(tdf)):
        end_date = tdf.iloc[i]["Date"]

        # Check label is available at this row
        if "label" not in tdf.columns or pd.isna(tdf.iloc[i].get("label")):
            continue
        label        = int(tdf.iloc[i]["label"])
        fwd_return   = float(tdf.iloc[i].get("forward_return", np.nan))

        # Extract extended window
        result = get_window(tdf, end_date, window, lookback)
        if result is None:
            continue
        ohlcv, tri = result

        img = render_window(
            ohlcv, tri, window, height, width, ratio, inc_ma, inc_vol
        )
        if img is None:
            continue

        rows.append({
            "ticker":         ticker,
            "end_date":       end_date,
            "forward_return": fwd_return,
            "label":          label,
            "img":            img,    # (1, H, W) uint8
        })

    return rows


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_cache(
    labelled_df: pd.DataFrame,
    img_cfg: ImageConfig,
    lbl_cfg: LabelConfig,
    cache_dir: Path,
    n_workers: int = 0,
) -> Path:
    """
    Generate all images from labelled_df and write the memmap + parquet index.

    Args:
        labelled_df: DataFrame with label + forward_return columns (output of
                     compute_labels in loader.py).
        img_cfg:     Image configuration.
        lbl_cfg:     Label configuration.
        cache_dir:   Directory to write cache files.
        n_workers:   Parallel workers (0 = use all CPUs).

    Returns:
        Path to the cache directory.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_file   = cache_dir / IMAGES_FILE
    idx_file   = cache_dir / INDEX_FILE
    stats_file = cache_dir / STATS_FILE

    if img_file.exists() and idx_file.exists():
        log.info(f"Cache already exists at {cache_dir} — skipping build.")
        return cache_dir

    ticker_index = build_ticker_index(labelled_df)
    tickers      = list(ticker_index.keys())
    n_workers    = n_workers or max(1, mp.cpu_count() - 1)

    log.info(f"Building image cache for {len(tickers)} tickers "
             f"({img_cfg.height}×{img_cfg.width}, window={img_cfg.window}) "
             f"using {n_workers} workers ...")

    icfg_d = img_cfg.model_dump(exclude={"cache_dir"})
    lcfg_d = lbl_cfg.model_dump()

    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(ticker_index, icfg_d, lcfg_d),
    ) as pool:
        results_nested = pool.map(_process_ticker, tickers)

    # Flatten
    all_records = [rec for sublist in results_nested for rec in sublist]

    if not all_records:
        raise RuntimeError("Cache build produced 0 valid images — check your data.")

    N = len(all_records)
    H, W = img_cfg.height, img_cfg.width
    log.info(f"  {N:,} valid images generated — writing memmap ...")

    # Write memmap
    fp = np.lib.format.open_memmap(
        str(img_file), mode="w+", dtype=np.uint8, shape=(N, 1, H, W)
    )
    for i, rec in enumerate(all_records):
        fp[i] = rec["img"]
    del fp   # flush

    # Write parquet index
    index_df = pd.DataFrame([
        {
            "ticker":         r["ticker"],
            "end_date":       r["end_date"],
            "forward_return": r["forward_return"],
            "label":          r["label"],
        }
        for r in all_records
    ])
    index_df.to_parquet(idx_file, index=False)

    log.info(f"  Cache written: {img_file}  ({N:,} images, "
             f"{img_file.stat().st_size / 1e9:.2f} GB)")

    return cache_dir


def compute_pixel_stats(
    cache_dir: Path,
    train_indices: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute mean and std of pixel values over the training set (PRD §5).
    Saves to pixel_stats.npz in cache_dir for reuse.

    Returns:
        (mean, std)  — scalar floats in [0, 255] range
    """
    # Include a hash of train_indices so different splits get separate stats
    idx_hash   = hashlib.md5(train_indices.tobytes()).hexdigest()[:8]
    stats_file = cache_dir / f"pixel_stats_{idx_hash}.npz"
    if stats_file.exists():
        d = np.load(stats_file)
        return float(d["mean"]), float(d["std"])

    img_file = cache_dir / IMAGES_FILE
    idx_file = cache_dir / INDEX_FILE
    index_df = pd.read_parquet(idx_file)
    N        = len(index_df)
    H        = int(np.load(str(img_file), mmap_mode="r").shape[2])
    W        = int(np.load(str(img_file), mmap_mode="r").shape[3])

    fp = np.load(str(img_file), mmap_mode="r")

    # Vectorised two-pass (chunked) mean/std — fast on numpy float32
    chunk      = 5000
    n_chunks   = (len(train_indices) + chunk - 1) // chunk
    sum_       = np.float64(0.0)
    sum_sq     = np.float64(0.0)
    n_pixels   = np.int64(0)

    for start in range(0, len(train_indices), chunk):
        idx   = train_indices[start : start + chunk]
        batch = fp[idx].astype(np.float32).ravel().astype(np.float64)
        sum_     += batch.sum()
        sum_sq   += (batch ** 2).sum()
        n_pixels += len(batch)

    mean = float(sum_ / n_pixels)
    std  = float(np.sqrt(max(sum_sq / n_pixels - mean ** 2, 0.0)))
    if std < 1e-6:
        std = 1.0
    np.savez(stats_file, mean=mean, std=std)
    log.info(f"Pixel stats: mean={mean:.4f}  std={std:.4f}")
    return mean, std


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_cache(cache_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load existing cache.

    Returns:
        (images_memmap, index_df)
        images_memmap: (N, 1, H, W) uint8 memmap
        index_df:      DataFrame with ticker, end_date, forward_return, label
    """
    cache_dir = Path(cache_dir)
    img_file  = cache_dir / IMAGES_FILE
    idx_file  = cache_dir / INDEX_FILE

    if not img_file.exists() or not idx_file.exists():
        raise FileNotFoundError(f"Cache not found at {cache_dir}. Run build_cache first.")

    images   = np.load(str(img_file), mmap_mode="r")
    index_df = pd.read_parquet(idx_file)
    log.info(f"Loaded cache: {images.shape}  |  {len(index_df):,} samples")
    return images, index_df
