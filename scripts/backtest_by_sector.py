"""
=============================================================================
SCRIPT NAME: backtest_by_sector.py
=============================================================================

INPUT FILES:
- --predictions PATH        predictions.parquet (ticker, end_date,
                            forward_return, label, p_up_mean)
- --classifications PATH    data/r1000_classifications.parquet
- --sector-col COL          column in classifications to group by
                            (default: bics_level_1)
- --top-pct / --bot-pct     within-sector percentile thresholds
                            (default 10/10 → D10-D1 within each sector)
- --hold-days               forward horizon for non-overlap sampling (20)

OUTPUT FILES:
- {out_dir}/by_sector_summary.xlsx     per-sector stats table
- {out_dir}/by_sector_portfolios.parquet  per-sector daily returns
- {out_dir}/by_sector_cumulative.pdf   grid of cumulative LS curves

DESCRIPTION:
For each sector, rank tickers cross-sectionally by p_up_mean WITHIN that
sector at each end_date, form equal-weight top X% and bottom X% portfolios,
and compute annualised stats + Newey-West t.  This isolates whether the
CNN adds value beyond sector rotation — a real within-sector edge means
the model sorts homogeneous stocks, not just "buy tech / sell energy".

USAGE:
  python scripts/backtest_by_sector.py \
      --predictions runs/expanding/20260419_174908_cdef6809/predictions.parquet \
      --classifications data/r1000_classifications.parquet \
      --out-dir runs/expanding/20260419_174908_cdef6809/backtest_by_sector \
      --sector-col bics_level_1 --top-pct 10 --bot-pct 10
=============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_portfolios_by_sector(
    preds: pd.DataFrame,
    sector_col: str,
    top_pct: float,
    bot_pct: float,
) -> pd.DataFrame:
    """Return (sector, end_date) indexed DataFrame with TOP, BOT, LS, EW."""
    rows = []
    for (sector, date), g in preds.groupby([sector_col, "end_date"], sort=True):
        if len(g) < 2:
            continue
        r = g["forward_return"].to_numpy()
        p = g["p_up_mean"].to_numpy()
        n = len(g)
        rank = pd.Series(p).rank(method="first") / n
        top_mask = (rank >  1 - top_pct / 100).to_numpy()
        bot_mask = (rank <=     bot_pct / 100).to_numpy()
        top_ret = r[top_mask].mean() if top_mask.any() else np.nan
        bot_ret = r[bot_mask].mean() if bot_mask.any() else np.nan
        rows.append({
            "sector":   sector,
            "end_date": date,
            "n":        n,
            "n_top":    int(top_mask.sum()),
            "n_bot":    int(bot_mask.sum()),
            "TOP":      top_ret,
            "BOT":      bot_ret,
            "LS":       top_ret - bot_ret,
            "EW":       r.mean(),
        })
    return pd.DataFrame(rows)


def ann_stats(r: pd.Series, hold_days: int = 20) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return dict(n=0, cum_x=np.nan, ann_comp=np.nan,
                    ann_vol=np.nan, sharpe=np.nan)
    m = r.mean()
    s = r.std()
    cum = float(np.exp(r.sum()))
    ann_comp = float(np.exp(m * 252 / hold_days) - 1)
    ann_vol = s * np.sqrt(252 / hold_days)
    sharpe = (m * 252 / hold_days) / ann_vol if ann_vol > 0 else np.nan
    return dict(n=int(len(r)), cum_x=cum, ann_comp=ann_comp * 100,
                ann_vol=ann_vol * 100, sharpe=sharpe)


def newey_west_t(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan
    mu = x.mean()
    xc = x - mu
    s = (xc ** 2).mean()
    for k in range(1, lag + 1):
        if k >= n:
            break
        gk = (xc[k:] * xc[:-k]).mean()
        w = 1 - k / (lag + 1)
        s += 2 * w * gk
    se = np.sqrt(max(s, 0) / n)
    return float(mu / se) if se > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--classifications", type=Path,
                    default=Path("data/r1000_classifications.parquet"))
    ap.add_argument("--sector-col", default="bics_level_1")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top-pct", type=float, default=10.0)
    ap.add_argument("--bot-pct", type=float, default=10.0)
    ap.add_argument("--hold-days", type=int, default=20)
    ap.add_argument("--monthly", action="store_true",
                    help="input already monthly non-overlap — no sub-sampling, plain t.")
    ap.add_argument("--min-sector-size", type=int, default=20,
                    help="skip sectors with mean daily universe < this")
    args = ap.parse_args()

    preds = pd.read_parquet(args.predictions)
    preds["end_date"] = pd.to_datetime(preds["end_date"])

    cls = pd.read_parquet(args.classifications)
    if args.sector_col not in cls.columns:
        raise KeyError(f"{args.sector_col} not in classifications "
                       f"(have: {cls.columns.tolist()})")

    preds = preds.merge(cls[["ticker", args.sector_col]], on="ticker", how="left")
    unclassified = preds[args.sector_col].isna().sum()
    preds = preds.dropna(subset=[args.sector_col])
    sector_label = args.sector_col

    print(f"Predictions: {len(preds):,} rows  |  unclassified dropped: {unclassified:,}")
    print(f"Sector column: {sector_label}  "
          f"({preds[sector_label].nunique()} sectors)")

    # Build per-sector portfolios
    port = build_portfolios_by_sector(preds, sector_label,
                                       args.top_pct, args.bot_pct)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    port.to_parquet(args.out_dir / "by_sector_portfolios.parquet", index=False)

    # ── per-sector summary ────────────────────────────────────────────────
    summary_rows = []
    for sector, g in port.groupby("sector"):
        g = g.sort_values("end_date").set_index("end_date")
        if args.monthly:
            monthly = g                         # already non-overlap
            nw_lag = 0
        else:
            monthly = g.iloc[::args.hold_days]
            nw_lag = args.hold_days - 1

        ls = ann_stats(monthly["LS"],  args.hold_days)
        tp = ann_stats(monthly["TOP"], args.hold_days)
        bt = ann_stats(monthly["BOT"], args.hold_days)
        ew = ann_stats(monthly["EW"],  args.hold_days)
        nw_ls  = newey_west_t(monthly["LS"].to_numpy(),  nw_lag)
        nw_top = newey_west_t(monthly["TOP"].to_numpy(), nw_lag)
        nw_bot = newey_west_t(monthly["BOT"].to_numpy(), nw_lag)

        summary_rows.append({
            "sector":        sector,
            "mean_universe": g["n"].mean(),
            "mean_n_top":    g["n_top"].mean(),
            "mean_n_bot":    g["n_bot"].mean(),
            "months":        ls["n"],
            "top_cagr":      tp["ann_comp"],
            "bot_cagr":      bt["ann_comp"],
            "ls_cagr":       ls["ann_comp"],
            "ls_vol":        ls["ann_vol"],
            "ls_sharpe":     ls["sharpe"],
            "ls_cum_x":      ls["cum_x"],
            "ew_cagr":       ew["ann_comp"],
            "nw_t_ls":       nw_ls,
            "nw_t_top":      nw_top,
            "nw_t_bot":      nw_bot,
        })
    summary = pd.DataFrame(summary_rows).sort_values("ls_cagr", ascending=False)

    # Filter by min universe size
    big_enough = summary["mean_universe"] >= args.min_sector_size
    print()
    print(f"Skipping sectors with mean universe < {args.min_sector_size}: "
          f"{(~big_enough).sum()}")
    print(summary[~big_enough][["sector", "mean_universe"]].to_string(index=False))
    print()

    print(f"Within-sector Top {args.top_pct:.0f}% / Bot {args.bot_pct:.0f}%  "
          f"(hold={args.hold_days}d)")
    cols = ["sector", "mean_universe", "mean_n_top", "months",
            "top_cagr", "bot_cagr", "ls_cagr", "ls_sharpe",
            "nw_t_ls", "ew_cagr"]
    print(summary[big_enough][cols].round(2).to_string(index=False))
    print()

    # Universe-wide (for comparison)
    total_rows = []
    for date, g in preds.groupby("end_date"):
        r = g["forward_return"].to_numpy()
        p = g["p_up_mean"].to_numpy()
        n = len(g)
        rank = pd.Series(p).rank(method="first") / n
        top_mask = (rank >  1 - args.top_pct / 100).to_numpy()
        bot_mask = (rank <=     args.bot_pct / 100).to_numpy()
        total_rows.append({
            "end_date": date,
            "TOP": r[top_mask].mean() if top_mask.any() else np.nan,
            "BOT": r[bot_mask].mean() if bot_mask.any() else np.nan,
            "EW":  r.mean(),
        })
    tot = pd.DataFrame(total_rows).set_index("end_date").sort_index()
    tot["LS"] = tot["TOP"] - tot["BOT"]
    if args.monthly:
        totm = tot
        tot_nw_lag = 0
    else:
        totm = tot.iloc[::args.hold_days]
        tot_nw_lag = args.hold_days - 1
    tot_ls = ann_stats(totm["LS"], args.hold_days)
    tot_ew = ann_stats(totm["EW"], args.hold_days)
    tot_nw = newey_west_t(totm["LS"].to_numpy(), tot_nw_lag)
    print(f"Cross-sector (universe-wide) Top {args.top_pct:.0f}% / "
          f"Bot {args.bot_pct:.0f}%  →  "
          f"LS CAGR={tot_ls['ann_comp']:.2f}%  Sharpe={tot_ls['sharpe']:.3f}  "
          f"NW-t={tot_nw:+.2f}  |  EW CAGR={tot_ew['ann_comp']:.2f}%")
    print()

    # Write Excel
    with pd.ExcelWriter(args.out_dir / "by_sector_summary.xlsx",
                         engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="by_sector", index=False)
        pd.DataFrame([{
            "portfolio": "universe_wide",
            "ls_cagr": tot_ls["ann_comp"],
            "ls_sharpe": tot_ls["sharpe"],
            "nw_t_ls": tot_nw,
            "ew_cagr": tot_ew["ann_comp"],
        }]).to_excel(xw, sheet_name="universe", index=False)

    # ── grid plot of cumulative LS per sector ─────────────────────────────
    big = summary[big_enough].sort_values("ls_cagr", ascending=False)
    sectors = big["sector"].tolist()
    n = len(sectors)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows),
                              sharex=True)
    axes = np.atleast_2d(axes)
    for i, sector in enumerate(sectors):
        ax = axes[i // cols, i % cols]
        g = port[port["sector"] == sector].sort_values("end_date")
        monthly = g.set_index("end_date").iloc[::args.hold_days]
        cum = np.exp(monthly[["TOP", "BOT", "LS", "EW"]].cumsum())
        ax.plot(cum.index, cum["TOP"], color="#2e7d32", lw=1.1, label="TOP")
        ax.plot(cum.index, cum["BOT"], color="#c62828", lw=1.1, label="BOT")
        ax.plot(cum.index, cum["EW"],  color="grey",  lw=1.0, ls="--", label="EW")
        ax.plot(cum.index, cum["LS"],  color="tab:blue", lw=1.3, label="LS")
        ax.set_yscale("log")
        stats = summary[summary["sector"] == sector].iloc[0]
        ax.set_title(f"{sector} — LS {stats['ls_cagr']:+.1f}% "
                     f"(NW t={stats['nw_t_ls']:+.2f})", fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")
    # Hide unused subplots
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_visible(False)
    fig.suptitle(f"Within-sector cumulative — Top {args.top_pct:.0f}% / "
                 f"Bot {args.bot_pct:.0f}% — {sector_label}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_pdf = args.out_dir / "by_sector_cumulative.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {args.out_dir / 'by_sector_summary.xlsx'}")


if __name__ == "__main__":
    main()
