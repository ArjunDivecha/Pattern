"""
=============================================================================
SCRIPT NAME: backtest_by_feature.py
=============================================================================

INPUT FILES:
- --predictions PATH    predictions_monthly.parquet (ticker, end_date,
                        forward_return, label, p_up_mean, p_up_std, ...)
- --ohlcv PATH          data/r1000_ohlcv_database.parquet (for price/vol
                        features)

OUTPUT FILES:
- {out_dir}/by_{feature}_summary.xlsx   per-bucket stats + universe
- {out_dir}/by_{feature}_portfolios.parquet
- {out_dir}/by_{feature}_cumulative.pdf per-bucket TOP/BOT/LS/EW cumulative

DESCRIPTION:
Generic within-bucket 50/50 long-short backtester.  At each month-end,
bucket the universe by --feature, then inside each bucket rank by
p_up_mean and form an equal-weight top-half / bot-half LS portfolio.
Feature values are recomputed at every month-end from OHLCV or directly
from the predictions file, so names migrate between tiers over time.

Supported features:
  mom_12_1      trailing 252d log-return skipping last 21d
  mom_6_1       trailing 126d log-return skipping last 21d
  rev_1m        trailing 21d log-return (short-term reversal)
  vol_60d       stdev of trailing 60d log-return × sqrt(252)
  drawdown_52w  1 - Close / 252d trailing max
  dv_60d        trailing 60d mean of Close × Volume (MCAP proxy)
  p_up_std      model ensemble disagreement (from predictions)
  abs_p_up      |p_up_mean − 0.5| (signal magnitude, from predictions)

USAGE:
  python scripts/backtest_by_feature.py \
      --predictions runs/expanding/.../predictions_monthly.parquet \
      --feature mom_12_1 --n-buckets 3 \
      --out-dir runs/expanding/.../backtest_by_mom_12_1
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


# ── feature builders ──────────────────────────────────────────────────────────

def f_mom_n_skip(ohlcv: pd.DataFrame, n: int, skip: int) -> pd.DataFrame:
    """Trailing n-day log return skipping the most recent `skip` days."""
    df = ohlcv[["Ticker", "Date", "AdjClose"]].copy()
    df = df.sort_values(["Ticker", "Date"])
    df["logp"] = np.log(df["AdjClose"])
    # momentum from t-(n+skip) to t-skip  =>  shift(skip) - shift(n+skip)
    g = df.groupby("Ticker")["logp"]
    df["feat"] = g.shift(skip) - g.shift(n + skip)
    return df[["Ticker", "Date", "feat"]].rename(
        columns={"Ticker": "ticker", "Date": "end_date"})


def f_rev_1m(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv[["Ticker", "Date", "AdjClose"]].copy()
    df = df.sort_values(["Ticker", "Date"])
    df["logp"] = np.log(df["AdjClose"])
    df["feat"] = df.groupby("Ticker")["logp"].diff(21)
    return df[["Ticker", "Date", "feat"]].rename(
        columns={"Ticker": "ticker", "Date": "end_date"})


def f_vol_60d(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv[["Ticker", "Date", "AdjClose"]].copy()
    df = df.sort_values(["Ticker", "Date"])
    df["ret"] = df.groupby("Ticker")["AdjClose"].pct_change()
    df["feat"] = (df.groupby("Ticker")["ret"]
                    .transform(lambda s: s.rolling(60, min_periods=20).std()
                                          * np.sqrt(252)))
    return df[["Ticker", "Date", "feat"]].rename(
        columns={"Ticker": "ticker", "Date": "end_date"})


def f_drawdown_52w(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv[["Ticker", "Date", "AdjClose"]].copy()
    df = df.sort_values(["Ticker", "Date"])
    df["roll_max"] = (df.groupby("Ticker")["AdjClose"]
                        .transform(lambda s: s.rolling(252, min_periods=60).max()))
    df["feat"] = 1 - df["AdjClose"] / df["roll_max"]
    return df[["Ticker", "Date", "feat"]].rename(
        columns={"Ticker": "ticker", "Date": "end_date"})


def f_dv_60d(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv[["Ticker", "Date", "Close", "Volume"]].copy()
    df = df.sort_values(["Ticker", "Date"])
    df["dv"] = df["Close"] * df["Volume"]
    df["feat"] = (df.groupby("Ticker")["dv"]
                    .transform(lambda s: s.rolling(60, min_periods=15).mean()))
    return df[["Ticker", "Date", "feat"]].rename(
        columns={"Ticker": "ticker", "Date": "end_date"})


OHLCV_FEATURES = {
    "mom_12_1":     lambda o: f_mom_n_skip(o, n=252, skip=21),
    "mom_6_1":      lambda o: f_mom_n_skip(o, n=126, skip=21),
    "rev_1m":       f_rev_1m,
    "vol_60d":      f_vol_60d,
    "drawdown_52w": f_drawdown_52w,
    "dv_60d":       f_dv_60d,
}

PRED_FEATURES = {
    "p_up_std":  "p_up_std",
    "abs_p_up":  "abs_p_up",  # computed from p_up_mean
}


# ── stats helpers ─────────────────────────────────────────────────────────────

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


def bucket_by_date(df: pd.DataFrame, col: str, n: int,
                   labels: list[str]) -> pd.Series:
    def assign(g):
        if len(g) < n or g[col].notna().sum() < n:
            return pd.Series([pd.NA] * len(g), index=g.index)
        return pd.qcut(g[col].rank(method="first"), n,
                       labels=labels, duplicates="drop")
    return df.groupby("end_date", group_keys=False).apply(assign)


def build_portfolios_50_50(preds: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    rows = []
    for (bucket, date), g in preds.groupby([bucket_col, "end_date"],
                                             sort=True, observed=True):
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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ohlcv", type=Path,
                    default=Path("data/r1000_ohlcv_database.parquet"))
    ap.add_argument("--feature", required=True,
                    choices=list(OHLCV_FEATURES) + list(PRED_FEATURES))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-buckets", type=int, default=3)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.n_buckets == 3:
        labels = ["Low", "Mid", "High"]
    elif args.n_buckets == 5:
        labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    else:
        labels = [f"B{i+1}" for i in range(args.n_buckets)]

    print(f"Loading predictions {args.predictions} …")
    preds = pd.read_parquet(args.predictions)
    preds["end_date"] = pd.to_datetime(preds["end_date"])
    print(f"  preds: {len(preds):,} rows  {preds['end_date'].min().date()} "
          f"→ {preds['end_date'].max().date()}  tickers={preds['ticker'].nunique()}")

    # Build feature
    if args.feature in OHLCV_FEATURES:
        print(f"Loading OHLCV {args.ohlcv} …")
        ohlcv = pd.read_parquet(args.ohlcv)
        ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
        print(f"Computing feature: {args.feature}")
        feat = OHLCV_FEATURES[args.feature](ohlcv)
        preds = preds.merge(feat, on=["ticker", "end_date"], how="left")
    elif args.feature == "p_up_std":
        preds["feat"] = preds["p_up_std"]
    elif args.feature == "abs_p_up":
        preds["feat"] = (preds["p_up_mean"] - 0.5).abs()
    else:
        raise ValueError(args.feature)

    pre_n = len(preds)
    preds = preds.dropna(subset=["feat"])
    print(f"  valid feature rows: {len(preds):,} (dropped {pre_n - len(preds):,})")

    # Bucket within each end_date
    preds["bucket"] = bucket_by_date(preds, "feat", args.n_buckets, labels)
    preds["bucket"] = pd.Categorical(preds["bucket"],
                                       categories=labels, ordered=True)
    preds = preds.dropna(subset=["bucket"])

    sizes = (preds.groupby(["end_date", "bucket"], observed=True).size()
                   .groupby("bucket", observed=True).mean())
    print(f"Mean names per bucket per date:")
    print(sizes.round(1).to_string())
    print()

    port = build_portfolios_50_50(preds, "bucket")
    port.to_parquet(args.out_dir / f"by_{args.feature}_portfolios.parquet",
                    index=False)

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

    print(f"Within-{args.feature}-bucket Top 50% / Bot 50%")
    cols = ["bucket", "mean_universe", "mean_n_top", "months",
            "top_cagr", "bot_cagr", "ls_cagr", "ls_sharpe",
            "nw_t_ls", "ew_cagr"]
    print(summary[cols].round(2).to_string(index=False))
    print()

    # Universe-wide
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
            "TOP": r[top_mask].mean() if top_mask.any() else np.nan,
            "BOT": r[bot_mask].mean() if bot_mask.any() else np.nan,
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

    with pd.ExcelWriter(args.out_dir / f"by_{args.feature}_summary.xlsx",
                         engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name=f"by_{args.feature}", index=False)
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
    fig.suptitle(f"Within-{args.feature} cumulative — Top 50% / Bot 50%  "
                 f"({args.n_buckets} buckets)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_pdf = args.out_dir / f"by_{args.feature}_cumulative.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {args.out_dir / f'by_{args.feature}_summary.xlsx'}")


if __name__ == "__main__":
    main()
