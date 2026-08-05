"""
=============================================================================
SCRIPT NAME: metrics.py
=============================================================================

DESCRIPTION:
    Computes return-series statistics for PRD Section 8 backtest analysis.
    This module provides functions to produce scalar portfolio metrics
    (mean annualized return, volatility, Sharpe ratio, Newey-West t-stat,
    maximum drawdown) from a series of overlapping log returns sampled once
    per formation day, and to compute decile-membership turnover rates.
    All data flows through in-memory pandas Series/DataFrame arguments and
    dict/DataFrame return values -- no files are read or written.

    The Newey-West HAC (Bartlett-kernel) variance estimator corrects for
    autocorrelation introduced by overlapping 20-business-day holding
    periods. The non-overlapping max drawdown sub-samples every h-th
    observation to produce a valid wealth path (not inflated by overlap).
    Turnover uses Jegadeesh-Titman-style symmetric-difference fractions
    per formation day, aggregated to per-day and per-month measures.

INPUT FILES:
    (none -- this script operates entirely on in-memory data structures)

OUTPUT FILES:
    (none -- the caller receives return values in memory)

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - numpy
    - pandas
    - pattern.backtest.deciles (for decile assignment in compute_turnover)

USAGE:
    python metrics.py          # (this file is intended for import, not direct execution)

NOTES:
    - Log returns are expected to span a 20-business-day label horizon,
      sampled once per formation day, creating overlapping observations.
    - Annualization applies a factor of 252/h for mean and sqrt(252/h) for
      volatility, where h is the holding period in business days.
    - The default Newey-West lag is max(4, h - 1).
    - Turnover requires a pred_df with columns: end_date, ticker, score_col.
    - Days with fewer than 2 * n_deciles stocks are dropped from turnover.
============================================================================="""
from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BUSINESS_DAYS_PER_YEAR = 252


def newey_west_variance(x: np.ndarray, lags: int) -> float:
    """HAC (Bartlett-kernel) variance estimator for the mean of x.

    Returns an estimate of Var(mean) so that SE = sqrt(NW-var).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    mu  = x.mean()
    dev = x - mu

    # Autocovariances γ_0 … γ_lags
    gammas = np.empty(lags + 1)
    gammas[0] = (dev * dev).sum() / n
    for k in range(1, lags + 1):
        gammas[k] = (dev[k:] * dev[:-k]).sum() / n

    # Bartlett kernel  w_k = 1 - k/(lags+1)
    weights = 1.0 - np.arange(lags + 1) / (lags + 1)
    # Long-run variance: γ_0 + 2 Σ w_k γ_k
    long_run_var = gammas[0] + 2.0 * (weights[1:] * gammas[1:]).sum()
    # Var(mean) = long_run_var / n
    return float(long_run_var / n)


def _non_overlapping_max_drawdown(x: np.ndarray, h: int) -> float:
    """Max drawdown on the non-overlapping sub-sample (every h-th observation).

    The overlapping cumulative log-return path (cumsum of h-day returns
    sampled daily) is NOT a valid strategy wealth path — its cumulative
    level is inflated h-fold relative to a real ladder strategy. Drawdown
    on that inflated series is not a drawdown anyone can experience.

    The honest alternative is to sub-sample every h-th formation day,
    yielding a non-overlapping series of h-day log returns whose cumsum
    IS a valid wealth path (for someone who holds one 20-day portfolio
    at a time). We report DD on that series.
    """
    if len(x) < 2:
        return 0.0
    sub = x[::h]
    if len(sub) < 2:
        return 0.0
    cum = np.cumsum(sub)
    run_max = np.maximum.accumulate(cum)
    return float((cum - run_max).min())


def summarize_series(
    returns: pd.Series,
    holding_period_days: int,
    nw_lags: int | None = None,
) -> Dict[str, float]:
    """Compute headline stats for a return series.

    Inputs are log returns over `holding_period_days` sampled per formation day.
    """
    x = pd.Series(returns).dropna().astype(float).values
    n = len(x)
    if n == 0:
        return {k: float("nan") for k in [
            "n_obs", "mean", "std", "mean_ann", "vol_ann", "sharpe_ann",
            "nw_se", "nw_tstat", "cum_log_return", "max_drawdown",
        ]}

    if nw_lags is None:
        nw_lags = max(4, holding_period_days - 1)

    mu    = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))

    # Annualization — log returns sampled per formation day, each spanning h days.
    ann_factor_mean = BUSINESS_DAYS_PER_YEAR / holding_period_days
    mean_ann        = mu    * ann_factor_mean
    vol_ann         = sigma * np.sqrt(ann_factor_mean)
    sharpe_ann      = mean_ann / vol_ann if vol_ann > 0 else float("nan")

    nw_var = newey_west_variance(x, nw_lags)
    nw_se  = float(np.sqrt(nw_var)) if nw_var == nw_var else float("nan")
    nw_t   = mu / nw_se if nw_se and nw_se > 0 else float("nan")

    # Reported DD uses the non-overlapping sub-sample (valid wealth path).
    max_dd = _non_overlapping_max_drawdown(x, holding_period_days)
    cum_log_return = float(np.cumsum(x[::holding_period_days]).sum())

    return {
        "n_obs":           int(n),
        "mean":            mu,
        "std":             sigma,
        "mean_ann":        mean_ann,
        "vol_ann":         vol_ann,
        "sharpe_ann":      sharpe_ann,
        "nw_se":           nw_se,
        "nw_tstat":        nw_t,
        "cum_log_return":  cum_log_return,
        "max_drawdown":    max_dd,
        "nw_lags":         int(nw_lags),
    }


def compute_turnover(
    pred_df: pd.DataFrame,
    score_col: str = "p_up_mean",
    n_deciles: int = 10,
    target_decile: int | None = None,
) -> Dict[str, float]:
    """Decile-membership turnover between consecutive formation days.

    Turnover_t = |S_t Δ S_{t-1}| / (|S_t| + |S_{t-1}|)
    When decile sizes are equal this equals the classic fraction-replaced
    metric (Jegadeesh-Titman): 0 = identical book, 1 = fully disjoint.

    Returns per-day and per-month (×21 trading days) turnover for the long
    decile (default: `n_deciles`), the short decile (1), and the LS book
    (mean of the two).
    """
    from pattern.backtest.deciles import _assign_deciles

    needed = {"end_date", "ticker", score_col}
    missing = needed - set(pred_df.columns)
    if missing:
        raise ValueError(f"pred_df missing columns for turnover: {missing}")

    df = pred_df[["end_date", "ticker", score_col]].dropna(subset=[score_col]).copy()
    # Drop sparse days (same rule as build_portfolios)
    day_sizes = df.groupby("end_date").size()
    keep_days = day_sizes[day_sizes >= 2 * n_deciles].index
    df = df[df["end_date"].isin(keep_days)]
    df["decile"] = (
        df.groupby("end_date", group_keys=False)[score_col]
          .transform(lambda s: _assign_deciles(s, n_deciles))
    )
    df = df.dropna(subset=["decile"])
    df["decile"] = df["decile"].astype(int)

    top_decile = n_deciles if target_decile is None else target_decile
    bot_decile = 1

    def _per_decile_turnover(dec: int) -> float:
        by_day = {d: set(grp["ticker"])
                  for d, grp in df[df["decile"] == dec].groupby("end_date")}
        days = sorted(by_day.keys())
        if len(days) < 2:
            return float("nan")
        vals = []
        for a, b in zip(days[:-1], days[1:]):
            sa, sb = by_day[a], by_day[b]
            denom = len(sa) + len(sb)
            if denom == 0:
                continue
            vals.append(len(sa.symmetric_difference(sb)) / denom)
        return float(np.mean(vals)) if vals else float("nan")

    long_to  = _per_decile_turnover(top_decile)
    short_to = _per_decile_turnover(bot_decile)

    # LS turnover averages the two books (equal capital)
    ls_to = np.nanmean([long_to, short_to])

    tdays_per_month = BUSINESS_DAYS_PER_YEAR / 12
    return {
        "long_turnover_per_day":    long_to,
        "short_turnover_per_day":   short_to,
        "ls_turnover_per_day":      float(ls_to),
        "long_turnover_per_month":  long_to  * tdays_per_month if long_to  == long_to  else float("nan"),
        "short_turnover_per_month": short_to * tdays_per_month if short_to == short_to else float("nan"),
        "ls_turnover_per_month":    ls_to    * tdays_per_month if ls_to    == ls_to    else float("nan"),
    }


def per_decile_stats(
    portfolios: pd.DataFrame,
    holding_period_days: int,
    n_deciles: int,
    nw_lags: int | None = None,
) -> pd.DataFrame:
    """Statistics for each decile's formation-day return series.

    Returns a DataFrame indexed by decile 1..n_deciles with columns
    matching summarize_series (plus `hit_rate`).
    """
    rows = []
    for d in range(1, n_deciles + 1):
        ser = (portfolios[portfolios["decile"] == d]
                 .sort_values("end_date")["mean_return"])
        stats = summarize_series(ser, holding_period_days, nw_lags)
        stats["decile"] = d
        stats["hit_rate"] = float((ser > 0).mean()) if len(ser) else float("nan")
        rows.append(stats)

    df = pd.DataFrame(rows).set_index("decile")
    return df
