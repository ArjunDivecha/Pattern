"""
=============================================================================
SCRIPT NAME: deciles.py
=============================================================================
INPUT:
- predictions DataFrame with columns:
    ticker, end_date, forward_return, label, p_up_mean

OUTPUT:
- portfolios_long  : DataFrame [end_date, decile, mean_forward_return, n_stocks]
- ls_series        : DataFrame [end_date, long_ret, short_ret, ls_ret, n_long, n_short]

DESCRIPTION:
Cross-sectional decile construction for PRD §8 backtest.

Each formation day t:
  1. Rank stocks by the ensemble score (default: p_up_mean) into n_deciles buckets
  2. Equal-weight within each bucket
  3. Report each bucket's mean forward_return (log, horizon = label horizon)
  4. Long-short  (D10 - D1)  is the headline portfolio

Holdings are overlapping by construction: every formation day produces a new
portfolio held for `holding_period_days` (usually 20). We do NOT spread
returns across daily buckets — each formation day gets a single 20-day
forward observation. Newey-West standard errors (lag ≥ holding_period - 1)
handle the serial correlation induced by the overlap.
=============================================================================
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _assign_deciles(scores: pd.Series, n_deciles: int) -> pd.Series:
    """Assign 1..n_deciles by ascending score. 1 = lowest P(up), n = highest."""
    if len(scores) < n_deciles:
        return pd.Series(np.nan, index=scores.index)
    # rank with method='first' avoids ties breaking the quantile cut
    ranks = scores.rank(method="first", ascending=True)
    # qcut returns 0..n-1; shift to 1..n
    bins = pd.qcut(ranks, n_deciles, labels=False, duplicates="drop")
    return bins.astype("float") + 1.0


def build_portfolios(
    pred_df: pd.DataFrame,
    n_deciles: int = 10,
    score_col: str = "p_up_mean",
    return_col: str = "forward_return",
    min_stocks_per_day: int | None = None,
) -> pd.DataFrame:
    """
    Per formation day: rank by score, cut into `n_deciles` equal-size buckets,
    compute equal-weighted mean of `return_col` within each bucket.

    Returns a long DataFrame: [end_date, decile, mean_return, n_stocks].
    Days with fewer than `min_stocks_per_day` (default = 2 * n_deciles) are skipped.
    """
    if min_stocks_per_day is None:
        min_stocks_per_day = 2 * n_deciles

    needed = {"end_date", score_col, return_col}
    missing = needed - set(pred_df.columns)
    if missing:
        raise ValueError(f"pred_df missing columns: {missing}")

    df = pred_df[["end_date", score_col, return_col]].copy()
    df = df.dropna(subset=[score_col, return_col])

    # Drop sparse days
    day_sizes = df.groupby("end_date").size()
    keep_days = day_sizes[day_sizes >= min_stocks_per_day].index
    dropped = len(day_sizes) - len(keep_days)
    if dropped:
        log.info(f"  Skipped {dropped} days with < {min_stocks_per_day} stocks")
    df = df[df["end_date"].isin(keep_days)].copy()

    df["decile"] = (
        df.groupby("end_date", group_keys=False)[score_col]
          .transform(lambda s: _assign_deciles(s, n_deciles))
    )
    df = df.dropna(subset=["decile"])
    df["decile"] = df["decile"].astype(int)

    port = (
        df.groupby(["end_date", "decile"], observed=True)[return_col]
          .agg(["mean", "size"])
          .reset_index()
          .rename(columns={"mean": "mean_return", "size": "n_stocks"})
          .sort_values(["end_date", "decile"])
    )
    log.info(f"  Built portfolios: {port['end_date'].nunique():,} days × "
             f"{n_deciles} deciles  ({len(port):,} rows)")
    return port


def long_short_series(
    portfolios: pd.DataFrame,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Extract the long-short time series: D_top - D_bottom per formation day.
    Requires `portfolios` from build_portfolios().
    """
    top    = portfolios[portfolios["decile"] == n_deciles][["end_date", "mean_return", "n_stocks"]]
    bottom = portfolios[portfolios["decile"] == 1         ][["end_date", "mean_return", "n_stocks"]]
    top    = top.rename(columns={"mean_return": "long_ret",  "n_stocks": "n_long"})
    bottom = bottom.rename(columns={"mean_return": "short_ret", "n_stocks": "n_short"})

    merged = top.merge(bottom, on="end_date", how="inner").sort_values("end_date")
    merged["ls_ret"] = merged["long_ret"] - merged["short_ret"]
    return merged.reset_index(drop=True)
