"""
=============================================================================
SCRIPT NAME: dataset.py
=============================================================================
DESCRIPTION:
Two PyTorch Dataset implementations:
  LiveDataset   — generates images on-the-fly from an OHLCV DataFrame.
                  Good for debugging and small splits.
  CachedDataset — reads from the pre-built memmap cache.  Recommended for
                  full training runs.

Both normalise pixel values using the training-set pixel mean/std (PRD §5).
=============================================================================
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pattern.config import ImageConfig
from pattern.data.loader import build_ticker_index, get_window
from pattern.imaging.renderer import render_window


# ---------------------------------------------------------------------------
# Live (on-the-fly) dataset
# ---------------------------------------------------------------------------

class LiveDataset(Dataset):
    """
    Generates images on demand from an OHLCV DataFrame.

    Args:
        labelled_df:  DataFrame with columns including label, forward_return,
                      Ticker, Date, Open/High/Low/Close/Volume, AdjClose.
        img_cfg:      Image configuration.
        pixel_mean:   Training-set pixel mean (for normalisation).
        pixel_std:    Training-set pixel std.
    """

    def __init__(
        self,
        labelled_df: pd.DataFrame,
        img_cfg: ImageConfig,
        pixel_mean: float = 0.0,
        pixel_std:  float = 1.0,
    ):
        self.img_cfg     = img_cfg
        self.pixel_mean  = pixel_mean
        self.pixel_std   = max(pixel_std, 1e-6)
        self.lookback    = img_cfg.window - 1   # extra days for full MA

        self.ticker_index = build_ticker_index(labelled_df)

        # Build a flat list of (ticker, end_date, label) records
        self.records = labelled_df[["Ticker", "Date", "label", "forward_return"]].copy()
        self.records = self.records.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row       = self.records.iloc[idx]
        ticker    = row["Ticker"]
        end_date  = row["Date"]
        label     = int(row["label"])

        cfg    = self.img_cfg
        tdf    = self.ticker_index.get(ticker)
        result = get_window(tdf, end_date, cfg.window, self.lookback) if tdf is not None else None

        if result is not None:
            ohlcv, tri = result
            img = render_window(
                ohlcv, tri,
                cfg.window, cfg.height, cfg.width,
                cfg.ohlc_height_ratio, cfg.include_ma, cfg.include_volume,
            )
        else:
            img = None

        if img is None:
            # Fallback: return black image (the sample will have valid label though)
            img = np.zeros((1, cfg.height, cfg.width), dtype=np.uint8)

        # Normalise pixel values to ~N(0,1)
        x = (img.astype(np.float32) - self.pixel_mean) / self.pixel_std
        return torch.from_numpy(x), label


# ---------------------------------------------------------------------------
# Cached dataset
# ---------------------------------------------------------------------------

class CachedDataset(Dataset):
    """
    Loads images from the pre-built memmap cache (much faster than LiveDataset).

    Args:
        images:      (N, 1, H, W) uint8 memmap array.
        index_df:    Parquet index DataFrame (ticker, end_date, forward_return, label).
        indices:     Subset of row indices to use (e.g. train or test split).
        pixel_mean:  Training-set pixel mean.
        pixel_std:   Training-set pixel std.
    """

    def __init__(
        self,
        images: np.ndarray,
        index_df: pd.DataFrame,
        indices: np.ndarray,
        pixel_mean: float = 0.0,
        pixel_std:  float = 1.0,
        preload: bool = True,
    ):
        self.index_df   = index_df.iloc[indices].reset_index(drop=True)
        self.indices    = indices
        self.pixel_mean = pixel_mean
        self.pixel_std  = max(pixel_std, 1e-6)

        # Copy subset into a contiguous RAM array to eliminate mmap random-access I/O.
        # At 64×60 bytes per image, 1M images ≈ 3.8 GB — fits easily in 128 GB RAM.
        if preload:
            self.images = np.ascontiguousarray(images[indices])
            self._use_preloaded = True
        else:
            self.images = images
            self._use_preloaded = False

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self._use_preloaded:
            img = self.images[idx].astype(np.float32)
        else:
            img = self.images[self.indices[idx]].astype(np.float32)
        label = int(self.index_df.iloc[idx]["label"])
        x = (img - self.pixel_mean) / self.pixel_std
        return torch.from_numpy(x), label
