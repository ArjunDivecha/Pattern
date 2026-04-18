"""
=============================================================================
SCRIPT NAME: metrics.py
=============================================================================
INPUT:
- pd.Series of per-formation-day log returns (long-short or single decile)

OUTPUT:
- Dict of scalar stats  +  DataFrame of per-decile stats

DESCRIPTION:
Return-series statistics for PRD §8 backtest.

Inputs are assumed to be log returns over the label horizon (20 business days),
sampled ONCE PER FORMATION DAY (overlapping observations). That means:

- N observations ≈ business days in test set
- Each observation is a 20-day log return → annualization factor = 252/h
- Overlap → Newey-West with lag ≥ h - 1 to get an honest t-statistic

For presentation we report:
  - mean_ann       : 252/h × mean of log returns  (continuously compounded)
  - vol_ann        : sqrt(252/h) × std of log returns
  - sharpe_ann     : mean_ann / vol_ann  (assumes rf ≈ 0, same as paper)
  - nw_tstat       : mean / NW-SE, where NW-SE uses HAC Bartlett kernel
  - cum_log_return : sum of daily-formation log returns  (cumulative P&L if
                     you held ONE 20-day portfolio at a time — not the same
                     as the overlapping strategy's wealth path; used only
                     for plotting shape)
=============================================================================
"""
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

    # Cumulative log-return path & max drawdown
    cum = np.cumsum(x)
    run_max = np.maximum.accumulate(cum)
    drawdown = cum - run_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    return {
        "n_obs":           int(n),
        "mean":            mu,
        "std":             sigma,
        "mean_ann":        mean_ann,
        "vol_ann":         vol_ann,
        "sharpe_ann":      sharpe_ann,
        "nw_se":           nw_se,
        "nw_tstat":        nw_t,
        "cum_log_return":  float(cum[-1]),
        "max_drawdown":    max_dd,
        "nw_lags":         int(nw_lags),
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
