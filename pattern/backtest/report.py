"""
=============================================================================
SCRIPT NAME: report.py
=============================================================================

DESCRIPTION:
    Runs a backtest from saved ensemble predictions and writes summary
    artifacts into the specified run directory.  Predictions are loaded
    from either predictions_ensemble.parquet or predictions.parquet (the
    latter is a fallback).  Each formation day's stocks are sorted by
    predicted probability of upward return (p_up_mean) and split into
    equal-weighted deciles.  A long-short portfolio goes long the
    highest-decile stocks and short the lowest-decile stocks, held for a
    configurable overlapping holding period (default 20 business days).
    For each portfolio, the script computes:
      - Log returns per formation day
      - Annualized mean, volatility, Sharpe ratio
      - Newey-West HAC t-statistic (with lags >= h-1)
      - Maximum drawdown (on the non-overlapping sub-sample)
      - Decile-membership turnover
    Outputs include Parquet files for machine consumption, an Excel
    spreadsheet with per-decile and summary statistics, two PDF plots,
    and a human-readable Markdown report.

INPUT FILES:
    <run_dir>/predictions_ensemble.parquet
        Ensemble-averaged predictions with columns: ticker, end_date,
        p_up_mean, forward_return, and optionally MarketCap.  Generated
        by a prior ensemble/aggregation step.  This is the primary input;
        if absent the script falls back to predictions.parquet.
    <run_dir>/predictions.parquet
        Raw (single-model) predictions, same schema.  Used only when the
        ensemble variant is missing.

OUTPUT FILES:
    <run_dir>/backtest_portfolios.parquet
        Per-day decile log returns with columns: end_date, decile (int),
        ret (mean log return across stocks in the decile).
    <run_dir>/backtest_ls.parquet
        Long-short time series with columns: end_date, ls_ret, long_ret,
        short_ret.  One row per formation day.
    <run_dir>/backtest_decile_stats.xlsx
        Excel spreadsheet with two sheets:
          - "per_decile":  annualized mean, vol, Sharpe, NW t-stat,
             hit rate, and observation count per decile.
          - "summary":     aggregate metrics for the long portfolio,
             short portfolio, and long-short spread (same statistics
             plus turnover).
    <run_dir>/backtest_cum_return.pdf
        Line plot of cumulative log returns for the long, short, and
        long-short portfolios over the test window.
    <run_dir>/backtest_decile_bar.pdf
        Bar chart of annualized mean log return per decile (darker bars
        for D1 and D{n_deciles}).
    <run_dir>/report.md
        Human-readable Markdown report with methodology notes, summary
        statistics table, per-decile table, and plot references.

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - Python >= 3.10
    - pandas
    - numpy
    - matplotlib
    - pattern (local library: pattern.config, pattern.backtest.deciles,
      pattern.backtest.metrics)

USAGE:
    This module is not meant to be run as a standalone script.  Import
    it and call `run_backtest(run_dir, cfg)` where `run_dir` is a Path to
    the directory containing predictions and `cfg` is a BacktestConfig.

NOTES:
    - All file paths are relative to the caller-provided <run_dir> Path;
      no hardcoded absolute paths are used in this module.
    - Returns are log returns, not simple returns.
    - Maximum drawdown is computed on a non-overlapping sub-sample (every
      h-th formation day) to produce a valid strategy wealth path.
    - Value-weighted backtesting requires a MarketCap column in the
      predictions.  Equal-weight is always available.
    - Newey-West lags default to max(cfg.newey_west_lags, h-1).
    - Plots use matplotlib with the "Agg" backend (no display).
=============================================================================
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pattern.config import BacktestConfig
from pattern.backtest.deciles import build_portfolios, long_short_series
from pattern.backtest.metrics import (
    summarize_series,
    per_decile_stats,
    compute_turnover,
    BUSINESS_DAYS_PER_YEAR,
)

log = logging.getLogger(__name__)


def _pick_predictions(run_dir: Path) -> Path:
    candidates = [
        run_dir / "predictions_ensemble.parquet",
        run_dir / "predictions.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No predictions parquet in {run_dir} "
        f"(looked for {[c.name for c in candidates]})"
    )


def run_backtest(
    run_dir: Path,
    cfg: BacktestConfig,
    score_col: str = "p_up_mean",
    return_col: str = "forward_return",
    holding_period_days: int = 20,
) -> dict:
    """Main entry point. Writes artifacts into `run_dir` and returns a dict summary."""
    pred_path = _pick_predictions(run_dir)
    log.info(f"Loading predictions from {pred_path.name}")
    pred = pd.read_parquet(pred_path)
    log.info(f"  {len(pred):,} rows | unique days: {pred['end_date'].nunique():,} | "
             f"unique tickers: {pred['ticker'].nunique():,}")

    # --- Weighting fallback warning ---------------------------------------
    # The config advertises ["equal", "value"]; if MarketCap is absent from the
    # predictions DataFrame we can only do equal-weighting. Fail loud here so
    # the user knows the value-weighted results are NOT being produced.
    if "value" in [w.lower() for w in (cfg.weighting or ["equal"])] \
       and "MarketCap" not in pred.columns and "mktcap" not in pred.columns:
        log.warning(
            "Value-weighted backtest requested (cfg.weighting includes 'value') "
            "but no MarketCap/mktcap column in predictions — falling back to "
            "equal-weight only. Add MarketCap to the source CSV + carry it "
            "through predictions to enable VW."
        )

    # --- Build portfolios --------------------------------------------------
    portfolios = build_portfolios(
        pred,
        n_deciles=cfg.n_deciles,
        score_col=score_col,
        return_col=return_col,
    )
    portfolios.to_parquet(run_dir / "backtest_portfolios.parquet", index=False)

    ls = long_short_series(portfolios, n_deciles=cfg.n_deciles)
    ls.to_parquet(run_dir / "backtest_ls.parquet", index=False)

    # --- Stats -------------------------------------------------------------
    nw_lags = max(cfg.newey_west_lags, holding_period_days - 1)
    ls_stats = summarize_series(ls["ls_ret"], holding_period_days, nw_lags)
    long_stats  = summarize_series(ls["long_ret"],  holding_period_days, nw_lags)
    short_stats = summarize_series(ls["short_ret"], holding_period_days, nw_lags)

    per_dec = per_decile_stats(portfolios, holding_period_days, cfg.n_deciles, nw_lags)

    # Turnover (decile-membership churn between consecutive formation days)
    turnover = compute_turnover(pred, score_col=score_col, n_deciles=cfg.n_deciles)
    ls_stats   = {**ls_stats,
                  "turnover_per_day":   turnover["ls_turnover_per_day"],
                  "turnover_per_month": turnover["ls_turnover_per_month"]}
    long_stats = {**long_stats,
                  "turnover_per_day":   turnover["long_turnover_per_day"],
                  "turnover_per_month": turnover["long_turnover_per_month"]}
    short_stats = {**short_stats,
                   "turnover_per_day":   turnover["short_turnover_per_day"],
                   "turnover_per_month": turnover["short_turnover_per_month"]}

    # Write decile stats to xlsx (project convention)
    with pd.ExcelWriter(run_dir / "backtest_decile_stats.xlsx") as xw:
        per_dec.to_excel(xw, sheet_name="per_decile")
        pd.DataFrame([
            {"portfolio": f"D{cfg.n_deciles} (long)",    **long_stats},
            {"portfolio": f"D1 (short)",                 **short_stats},
            {"portfolio": f"D{cfg.n_deciles}-D1 (LS)",   **ls_stats},
        ]).to_excel(xw, sheet_name="summary", index=False)

    # --- Plots (PDF, matplotlib only per project rule) ---------------------
    _plot_cum_return(ls, run_dir / "backtest_cum_return.pdf", cfg.n_deciles)
    _plot_decile_bar(per_dec, run_dir / "backtest_decile_bar.pdf",
                     holding_period_days, cfg.n_deciles)

    # --- report.md ---------------------------------------------------------
    _write_report_md(run_dir, cfg, ls_stats, long_stats, short_stats,
                     per_dec, holding_period_days, nw_lags, pred)

    log.info(
        f"Backtest complete   LS Sharpe={ls_stats['sharpe_ann']:.2f}   "
        f"mean_ann={ls_stats['mean_ann']*100:.2f}%   "
        f"t(NW)={ls_stats['nw_tstat']:.2f}   "
        f"DD(non-overlap)={ls_stats['max_drawdown']*100:.1f}%   "
        f"turnover/mo={ls_stats['turnover_per_month']*100:.1f}%"
    )
    return {
        "ls": ls_stats,
        "long": long_stats,
        "short": short_stats,
        "per_decile": per_dec,
    }


def _plot_cum_return(ls: pd.DataFrame, out: Path, n_deciles: int) -> None:
    ls = ls.sort_values("end_date").copy()
    ls["cum_ls"]    = ls["ls_ret"].cumsum()
    ls["cum_long"]  = ls["long_ret"].cumsum()
    ls["cum_short"] = ls["short_ret"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ls["end_date"], ls["cum_long"],  label=f"D{n_deciles} (long)",  color="#1f77b4")
    ax.plot(ls["end_date"], ls["cum_short"], label="D1 (short)",           color="#d62728")
    ax.plot(ls["end_date"], ls["cum_ls"],    label=f"D{n_deciles} - D1",   color="black", linewidth=1.8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Formation date")
    ax.set_ylabel("Cumulative h-day log return (summed across formation days — illustrative only)")
    ax.set_title("Decile & long-short cumulative log returns (overlapping cohorts)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    log.info(f"  Wrote {out.name}")


def _plot_decile_bar(
    per_dec: pd.DataFrame,
    out: Path,
    holding_period_days: int,
    n_deciles: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = per_dec.index.values
    y = per_dec["mean_ann"].values * 100.0
    colors = ["#d62728" if i == 1 else ("#1f77b4" if i == n_deciles else "#888888")
              for i in x]
    ax.bar(x, y, color=colors)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Decile (1 = lowest P(up),  {n} = highest P(up))".format(n=n_deciles))
    ax.set_ylabel(f"Annualized mean {holding_period_days}-day log return (%)")
    ax.set_title("Per-decile annualized mean return")
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    log.info(f"  Wrote {out.name}")


def _write_report_md(
    run_dir: Path,
    cfg: BacktestConfig,
    ls_stats: dict,
    long_stats: dict,
    short_stats: dict,
    per_dec: pd.DataFrame,
    holding_period_days: int,
    nw_lags: int,
    pred: pd.DataFrame,
) -> None:
    n_d = cfg.n_deciles

    def _fmt_stats(label: str, s: dict) -> list[str]:
        to_pm = s.get("turnover_per_month", float("nan"))
        to_str = f"{to_pm*100:.1f}%" if to_pm == to_pm else "n/a"
        return [
            f"| {label} | "
            f"{s['n_obs']:,} | "
            f"{s['mean']*100:.4f}% | "
            f"{s['std']*100:.4f}% | "
            f"{s['mean_ann']*100:.2f}% | "
            f"{s['vol_ann']*100:.2f}% | "
            f"{s['sharpe_ann']:.2f} | "
            f"{s['nw_tstat']:.2f} | "
            f"{s['max_drawdown']*100:.2f}% | "
            f"{to_str} |"
        ]

    head = (
        f"# Backtest report — {run_dir.name}\n\n"
        f"- Predictions: `{(_pick_predictions(run_dir)).name}`\n"
        f"- Test window: {pred['end_date'].min()} → {pred['end_date'].max()}\n"
        f"- Tickers: {pred['ticker'].nunique():,}   "
        f"Formation days: {pred['end_date'].nunique():,}   "
        f"Stock-days: {len(pred):,}\n"
        f"- Deciles: {n_d}   Holding: {holding_period_days}d overlapping   "
        f"Weighting: equal   Newey-West lags: {nw_lags}\n"
        f"\n"
        "Returns are **log** returns over the label horizon, sampled once per formation day.\n"
        "The long-short series below has 20-day overlapping observations; the Newey-West\n"
        "lag is set to max(config, h-1) to produce an honest t-statistic.\n\n"
    )

    summary_tbl = [
        "## Long-short summary\n",
        "| Portfolio | N | Per-period mean | Per-period std | Ann. mean | Ann. vol | Sharpe | t(NW) | Max DD (non-overlap) | Turnover/mo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary_tbl += _fmt_stats(f"D{n_d} (long)",  long_stats)
    summary_tbl += _fmt_stats("D1 (short)",      short_stats)
    summary_tbl += _fmt_stats(f"D{n_d} − D1 (LS)", ls_stats)
    summary_tbl.append("")

    dec_tbl = [
        "## Per-decile statistics\n",
        "| Decile | N | Ann. mean | Ann. vol | Sharpe | t(NW) | Hit rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in range(1, n_d + 1):
        row = per_dec.loc[d]
        dec_tbl.append(
            f"| {d} | {int(row['n_obs']):,} | "
            f"{row['mean_ann']*100:.2f}% | "
            f"{row['vol_ann']*100:.2f}% | "
            f"{row['sharpe_ann']:.2f} | "
            f"{row['nw_tstat']:.2f} | "
            f"{row['hit_rate']*100:.1f}% |"
        )
    dec_tbl.append("")

    plots = (
        "## Plots\n\n"
        "- `backtest_cum_return.pdf` — cumulative log returns of long, short, and LS\n"
        "- `backtest_decile_bar.pdf` — annualized mean return by decile\n"
        "- `backtest_decile_stats.xlsx` — spreadsheet with per-decile + summary stats\n\n"
    )

    notes = (
        "## Notes on methodology\n\n"
        f"- **Overlap handling**: each formation day produces a new portfolio held for "
        f"{holding_period_days} business days. Successive observations overlap by "
        f"{holding_period_days - 1} days. We report per-formation-day log returns and "
        f"apply a Newey-West HAC correction with lag = {nw_lags} to the t-statistic.\n"
        "- **Annualization**: mean × (252 / h), vol × sqrt(252 / h). No risk-free rate "
        "is subtracted — Sharpe matches paper's convention.\n"
        f"- **Max DD is computed on the non-overlapping sub-sample** (every {holding_period_days}th "
        "formation day), because the daily cumulative sum of overlapping h-day log returns "
        "is not a valid strategy wealth path — it over-counts returns by a factor of ≈h. "
        "Sub-sampling gives the wealth path of a hold-one-portfolio-at-a-time strategy.\n"
        "- **Turnover** is reported per month (≈21 trading days) as the "
        "fraction-replaced metric `|S_t Δ S_{t-1}| / (|S_t| + |S_{t-1}|)` summed "
        "across consecutive formation days. Full replacement between two "
        "formation days = 100%; a naive upper bound for a daily-rebalanced book "
        "is `21 × per-day turnover`.\n"
        "- **No transaction costs**. Sharpe above is gross. A daily-rebalanced overlapping "
        "strategy rotates 1/h of capital each day; typical round-trip cost ≈ turnover × spread.\n"
        "- **Equal-weight only**. Value-weighted variants require a `MarketCap` column "
        "in the source CSV. A warning is logged at backtest time if `cfg.weighting` "
        "requests `value` but the column is absent.\n"
    )

    (run_dir / "report.md").write_text(
        head + "\n".join(summary_tbl) + "\n" + "\n".join(dec_tbl) + "\n" + plots + notes
    )
    log.info(f"  Wrote report.md")
