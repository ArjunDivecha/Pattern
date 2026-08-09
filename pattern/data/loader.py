"""
=============================================================================
SCRIPT NAME: loader.py
=============================================================================
INPUT FILES:
- CSV with columns: Ticker, Date, Open, High, Low, Close, Volume, AdjClose
  (AdjClose = Bloomberg TOT_RETURN_INDEX_GROSS_DVDS — the total return index)

OUTPUT:
- Sorted DataFrame with a computed Return column (daily total return from AdjClose)
- per_ticker dict for fast window lookups

DESCRIPTION:
Loads the OHLCV database and prepares it for image generation and labelling.
Key design: returns the TOTAL RETURN index as-is (not an adjusted price).
The renderer uses it to compute normalized closes; the labeller uses it for
forward return computation.
=============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Column names we expect / emit
REQUIRED_COLS = {"Ticker", "Date", "Open", "High", "Low", "Close", "Volume"}
OHLCV_COLS    = ["Open", "High", "Low", "Close", "Volume"]


def load_data(
    csv_path: str | Path,
    date_format: str = "%Y-%m-%d",
    min_history_days: int = 0,
) -> pd.DataFrame:
    """
    Load and clean the OHLCV + AdjClose CSV.

    Returns a DataFrame sorted by (Ticker, Date) with columns:
        Ticker, Date, Open, High, Low, Close, Volume, AdjClose, Return

    AdjClose is the total-return index (Bloomberg TOT_RETURN_INDEX_GROSS_DVDS).
    Return = AdjClose_t / AdjClose_{t-1} - 1  (daily total return, NaN for first row).
    """
    log.info(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], format="mixed")

    for col in OHLCV_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "AdjClose" in df.columns:
        df["AdjClose"] = pd.to_numeric(df["AdjClose"], errors="coerce")
    else:
        log.warning("AdjClose not found — using Close as proxy (splits/dividends not adjusted)")
        df["AdjClose"] = df["Close"]

    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Daily total return from TRI
    df["Return"] = df.groupby("Ticker", sort=False)["AdjClose"].transform(
        lambda s: s.pct_change(fill_method=None)
    )

    # Drop tickers with fewer than min_history_days valid rows
    if min_history_days > 0:
        counts = df.groupby("Ticker")["Close"].count()
        keep   = counts[counts >= min_history_days].index
        before = df["Ticker"].nunique()
        df     = df[df["Ticker"].isin(keep)].copy()
        log.info(f"  Dropped {before - len(keep)} tickers with < {min_history_days} rows")

    n_tickers = df["Ticker"].nunique()
    log.info(f"  Loaded {len(df):,} rows  |  {n_tickers:,} tickers  |  "
             f"{df['Date'].min().date()} → {df['Date'].max().date()}")

    return df


def build_ticker_index(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build a per-ticker dict of DataFrames for O(1) ticker lookups.
    Each sub-DataFrame is indexed 0..N-1 with columns preserved.
    """
    return {ticker: grp.reset_index(drop=True)
            for ticker, grp in df.groupby("Ticker", sort=False)}


def get_window(
    ticker_df: pd.DataFrame,
    end_date: pd.Timestamp,
    window: int,
    lookback: int = 0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract OHLCV and AdjClose arrays for a window ending at end_date.

    Args:
        ticker_df:  Per-ticker DataFrame (from build_ticker_index).
        end_date:   Last date of the image window (inclusive).
        window:     Number of trading days in the image (e.g. 20).
        lookback:   Extra days BEFORE the window start for MA computation.
                    Total rows returned = window + lookback.

    Returns:
        (ohlcv, tri) where:
            ohlcv: float64 array (window+lookback, 5) — O/H/L/C/V
            tri:   float64 array (window+lookback,)  — AdjClose (TRI)
        or None if not enough data.
    """
    idx = ticker_df["Date"].searchsorted(end_date, side="right") - 1
    if idx < 0 or ticker_df.iloc[idx]["Date"] != end_date:
        return None   # end_date not in this ticker's data

    total = window + lookback
    start_idx = idx - total + 1
    if start_idx < 0:
        return None   # not enough history

    rows  = ticker_df.iloc[start_idx : idx + 1]
    ohlcv = rows[OHLCV_COLS].to_numpy(dtype=np.float64)
    tri   = rows["AdjClose"].to_numpy(dtype=np.float64)

    return ohlcv, tri


def compute_labels(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """
    Add forward_return and label columns to df (returns a copy).

    forward_return_t = log(AdjClose_{t+horizon} / AdjClose_t)
    label_t          = 1 if forward_return_t > 0 else 0

    Rows where the forward return cannot be computed are dropped.
    """
    log.info(f"Computing {horizon}-day forward returns ...")

    df = df.copy()
    df["_adj_fwd"] = df.groupby("Ticker", sort=False)["AdjClose"].shift(-horizon)
    valid_mask = df["AdjClose"].gt(0) & df["_adj_fwd"].gt(0)
    df.loc[valid_mask, "forward_return"] = np.log(
        df.loc[valid_mask, "_adj_fwd"] / df.loc[valid_mask, "AdjClose"]
    )
    df = df[valid_mask].drop(columns=["_adj_fwd"]).copy()
    df["label"] = (df["forward_return"] > 0).astype(np.int8)
    df.reset_index(drop=True, inplace=True)

    pos_rate = df["label"].mean()
    log.info(f"  {len(df):,} labelled rows  |  {pos_rate:.2%} positive")
    return df
