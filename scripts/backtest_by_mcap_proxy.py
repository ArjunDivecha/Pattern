"""
=============================================================================
SCRIPT NAME: backtest_by_mcap_proxy.py
=============================================================================

INPUT FILES:
- --predictions PATH    predictions_monthly.parquet (ticker, end_date,
                        forward_return, label, p_up_mean)
- --ohlcv PATH          data/r1000_ohlcv_database.parquet — for dollar-volume
                        proxy (Close × Volume, rolling 60-day mean)

OUTPUT FILES:
- {out_dir}/by_mcap_summary.xlsx     per-tier stats table + universe row
- {out_dir}/by_mcap_portfolios.parquet
- {out_dir}/by_mcap_cumulative.pdf   per-tier TOP/BOT/LS/EW cumulative

DESCRIPTION:
Slice R1000 into LARGE / MID / SMALL buckets at each end_date using a
trailing-60-day dollar-volume PROXY for market cap, then run 50/50
top-half vs bot-half LS within each bucket using the CNN p_up_mean
signal.  Buckets are re-sorted at every month-end so names can move
between tiers over time.

Why the proxy: OHLCV has no MCAP field and pulling historical CUR_MKT_CAP
for 3,011 tickers costs time.  Rolling dollar volume (close × shares
traded) correlates ~0.9 with true MCAP cross-sectionally within an
index — good enough for a first-pass "does the signal change with
size" test.  Confirm with real MCAP if results are interesting.

USAGE:
  python scripts/backtest_by_mcap_proxy.py \
      --predictions runs/expanding/20260419_174908_cdef6809/predictions_monthly.parquet \
      --out-dir runs/expanding/20260419_174908_cdef6809/backtest_by_mcap_proxy \
      --n-buckets 3
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


def compute_dollar_volume(ohlcv: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    df = ohlcv[["Ticker", "Date", "Close", "Volume"]].copy()
    df["dv"] = df["Close"] * df["Volume"]
    df = df.sort_values(["Ticker", "Date"])
    df["dv_60d"] = (
        df.groupby("Ticker")["dv"]
          .transform(lambda s: s.rolling(window, min_periods=max(5, window // 4)).mean())
    )
    return df[["Ticker", "Date", "dv_60d"]].rename(
        columns={"Ticker": "ticker", "Date": "end_date"})


def bucket_by_date(df: pd.DataFrame, col: str, n: int,
                   labels: list[str]) -> pd.Series:
    """Within each end_date, assign `col` into n quantile buckets."""
    def assign(g):
        if len(g) < n:
            return pd.Series([pd.NA] * len(g), index=g.index)
        return pd.qcut(g[col].rank(method="first"), n,
                       labels=labels, duplicates="drop")
    return df.groupby("end_date", group_keys=False).apply(assign)


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


def build_portfolios_50_50(preds: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    rows = []
    for (bucket, date), g in preds.groupby([bucket_col, "end_date"], sort=True):
        if len(g) < 2:
            continue
        r = g["forward_return"].to_numpy()
        p = g["p_up_mean"].to_numpy()
        n = len(g)
        rank = pd.Series(p).rank(method="first") / n
        top_mask = (rank >  0.5).to_numpy()
        bot_mask = (rank <= 0.5).to_numpy()
        top_ret = r[top_mask].mean() if top_mask.any() else np.nan
        bot_ret = r[bot_mask].mean() if bot_mask.any() else np.nan
        rows.append({
            "bucket":   bucket,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ohlcv", type=Path,
                    default=Path("data/r1000_ohlcv_database.parquet"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-buckets", type=int, default=3,
                    help="3 = Large/Mid/Small; 5 = quintiles")
    ap.add_argument("--dv-window", type=int, default=60,
                    help="trailing days for dollar-volume average")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Labels
    if args.n_buckets == 3:
        labels = ["Small", "Mid", "Large"]
    elif args.n_buckets == 5:
        labels = ["Q1-Small", "Q2", "Q3", "Q4", "Q5-Large"]
    else:
        labels = [f"B{i+1}" for i in range(args.n_buckets)]

    print(f"Loading OHLCV {args.ohlcv} …")
    ohlcv = pd.read_parquet(args.ohlcv)
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
    dv = compute_dollar_volume(ohlcv, window=args.dv_window)

    print(f"Loading predictions {args.predictions} …")
    preds = pd.read_parquet(args.predictions)
    preds["end_date"] = pd.to_datetime(preds["end_date"])
    print(f"  preds: {len(preds):,} rows  {preds['end_date'].min().date()} "
          f"→ {preds['end_date'].max().date()}  tickers={preds['ticker'].nunique()}")

    # Merge dollar volume onto (ticker, end_date)
    preds = preds.merge(dv, on=["ticker", "end_date"], how="left")
    pre_n = len(preds)
    preds = preds.dropna(subset=["dv_60d"])
    print(f"  merged dv: {len(preds):,} rows "
          f"(dropped {pre_n - len(preds):,} with no dv)")

    # Bucket within each end_date
    preds["mcap_bucket"] = bucket_by_date(preds, "dv_60d",
                                           args.n_buckets, labels)
    preds["mcap_bucket"] = pd.Categorical(preds["mcap_bucket"],
                                           categories=labels, ordered=True)

    # Mean bucket size per date
    sizes = (preds.groupby(["end_date", "mcap_bucket"]).size()
                   .groupby("mcap_bucket").mean())
    print(f"Mean names per bucket per date:")
    print(sizes.round(1).to_string())
    print()

    # Build per-bucket portfolios
    port = build_portfolios_50_50(preds, "mcap_bucket")
    port.to_parquet(args.out_dir / "by_mcap_portfolios.parquet", index=False)

    # Per-bucket summary
    summary_rows = []
    for bucket in labels:
        g = port[port["bucket"] == bucket].sort_values("end_date")
        if len(g) == 0:
            continue
        tp = ann_stats(g["TOP"])
        bt = ann_stats(g["BOT"])
        ls = ann_stats(g["LS"])
        ew = ann_stats(g["EW"])
        nw_ls  = newey_west_t(g["LS"].to_numpy(),  lag=0)
        nw_top = newey_west_t(g["TOP"].to_numpy(), lag=0)
        nw_bot = newey_west_t(g["BOT"].to_numpy(), lag=0)
        summary_rows.append({
            "bucket":        bucket,
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
    summary = pd.DataFrame(summary_rows)

    print("Within-MCAP-bucket Top 50% / Bot 50%")
    cols = ["bucket", "mean_universe", "mean_n_top", "months",
            "top_cagr", "bot_cagr", "ls_cagr", "ls_sharpe",
            "nw_t_ls", "ew_cagr"]
    print(summary[cols].round(2).to_string(index=False))
    print()

    # Universe-wide cross-bucket for context
    total_rows = []
    for date, g in preds.groupby("end_date"):
        r = g["forward_return"].to_numpy()
        p = g["p_up_mean"].to_numpy()
        n = len(g)
        rank = pd.Series(p).rank(method="first") / n
        top_mask = (rank >  0.5).to_numpy()
        bot_mask = (rank <= 0.5).to_numpy()
        total_rows.append({
            "end_date": date,
            "TOP": r[top_mask].mean(),
            "BOT": r[bot_mask].mean(),
            "EW":  r.mean(),
        })
    tot = pd.DataFrame(total_rows).set_index("end_date").sort_index()
    tot["LS"] = tot["TOP"] - tot["BOT"]
    tot_ls = ann_stats(tot["LS"])
    tot_ew = ann_stats(tot["EW"])
    tot_nw = newey_west_t(tot["LS"].to_numpy(), lag=0)
    print(f"Cross-universe Top 50% / Bot 50%  →  "
          f"LS CAGR={tot_ls['ann_comp']:.2f}%  Sharpe={tot_ls['sharpe']:.3f}  "
          f"NW-t={tot_nw:+.2f}  |  EW CAGR={tot_ew['ann_comp']:.2f}%")
    print()

    # Write Excel
    with pd.ExcelWriter(args.out_dir / "by_mcap_summary.xlsx",
                         engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="by_mcap", index=False)
        pd.DataFrame([{
            "portfolio": "universe_wide",
            "ls_cagr": tot_ls["ann_comp"],
            "ls_sharpe": tot_ls["sharpe"],
            "nw_t_ls": tot_nw,
            "ew_cagr": tot_ew["ann_comp"],
        }]).to_excel(xw, sheet_name="universe", index=False)

    # Plot
    n = len(summary)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.2), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, bucket in zip(axes, summary["bucket"].tolist()):
        g = port[port["bucket"] == bucket].sort_values("end_date")
        cum = np.exp(g.set_index("end_date")[["TOP", "BOT", "LS", "EW"]].cumsum())
        ax.plot(cum.index, cum["TOP"], color="#2e7d32", lw=1.2, label="TOP")
        ax.plot(cum.index, cum["BOT"], color="#c62828", lw=1.2, label="BOT")
        ax.plot(cum.index, cum["EW"],  color="grey",  lw=1.0, ls="--", label="EW")
        ax.plot(cum.index, cum["LS"],  color="tab:blue", lw=1.5, label="LS")
        ax.set_yscale("log")
        stats = summary[summary["bucket"] == bucket].iloc[0]
        ax.set_title(f"{bucket} — LS {stats['ls_cagr']:+.1f}% "
                     f"(NW t={stats['nw_t_ls']:+.2f})", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Within-MCAP-proxy cumulative — Top 50% / Bot 50%  "
                 f"(dv_{args.dv_window}d, {args.n_buckets} buckets)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_pdf = args.out_dir / "by_mcap_cumulative.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {args.out_dir / 'by_mcap_summary.xlsx'}")


if __name__ == "__main__":
    main()
