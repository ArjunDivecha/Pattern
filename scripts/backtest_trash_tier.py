"""
=============================================================================
SCRIPT NAME: backtest_trash_tier.py
=============================================================================

INPUT FILES:
- --predictions PATH    predictions_monthly.parquet (ticker, end_date,
                        forward_return, label, p_up_mean, p_up_std)
- --ohlcv PATH          data/r1000_ohlcv_database.parquet

OUTPUT FILES:
- {out_dir}/trash_tier_summary.xlsx       stats per filter combination
- {out_dir}/trash_tier_portfolios.parquet per-filter daily returns
- {out_dir}/trash_tier_cumulative.pdf     cumulative LS by filter stack

DESCRIPTION:
Per-date filtering: at each month-end, bucket names into Small/Mid/Large by
60d dollar-volume, Low/Mid/High by 252-21d momentum, Low/Mid/High by 60d
realized vol.  Then progressively intersect the extreme-trash buckets:
  (1) Small only
  (2) Small & high-vol
  (3) Small & recent-loser
  (4) High-vol & recent-loser
  (5) Small & high-vol & recent-loser  (triple)

Inside each filter's per-date subset, rank by p_up_mean and form a 50/50
top-half vs bot-half LS portfolio.  Compare per-month universe size and
annualised stats across filters.

USAGE:
  python scripts/backtest_trash_tier.py \
      --predictions runs/expanding/.../predictions_monthly.parquet \
      --out-dir runs/expanding/.../backtest_trash_tier
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


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv[["Ticker", "Date", "Close", "Volume", "AdjClose"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])
    g = df.groupby("Ticker")
    # Dollar volume (60d)
    df["dv"] = df["Close"] * df["Volume"]
    df["dv_60d"] = g["dv"].transform(
        lambda s: s.rolling(60, min_periods=15).mean())
    # Momentum 12-1 (252d skipping 21d)
    df["logp"] = np.log(df["AdjClose"].replace(0, np.nan))
    df["mom_12_1"] = g["logp"].shift(21) - g["logp"].shift(252 + 21)
    # 60d realized vol
    df["ret"] = g["AdjClose"].pct_change()
    df["vol_60d"] = g["ret"].transform(
        lambda s: s.rolling(60, min_periods=20).std() * np.sqrt(252))
    return df[["Ticker", "Date", "dv_60d", "mom_12_1", "vol_60d"]].rename(
        columns={"Ticker": "ticker", "Date": "end_date"})


def bucket_within_date(df: pd.DataFrame, col: str, n: int = 3) -> pd.Series:
    """Return categorical bucket labels (Low/Mid/High) per end_date."""
    labels = ["Low", "Mid", "High"] if n == 3 else [f"B{i+1}" for i in range(n)]
    def assign(g):
        if g[col].notna().sum() < n:
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
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan
    mu = x.mean(); xc = x - mu
    s = (xc ** 2).mean()
    for k in range(1, lag + 1):
        if k >= n: break
        gk = (xc[k:] * xc[:-k]).mean()
        w = 1 - k / (lag + 1)
        s += 2 * w * gk
    se = np.sqrt(max(s, 0) / n)
    return float(mu / se) if se > 0 else np.nan


def build_ls_50_50(preds: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    rows = []
    sub = preds.loc[mask]
    for date, g in sub.groupby("end_date"):
        if len(g) < 2:
            continue
        r = g["forward_return"].to_numpy()
        p = g["p_up_mean"].to_numpy()
        n = len(g)
        rank = pd.Series(p).rank(method="first") / n
        top_mask = (rank >  0.5).to_numpy()
        bot_mask = (rank <= 0.5).to_numpy()
        rows.append({
            "end_date": date,
            "n": n,
            "n_top": int(top_mask.sum()),
            "n_bot": int(bot_mask.sum()),
            "TOP": r[top_mask].mean() if top_mask.any() else np.nan,
            "BOT": r[bot_mask].mean() if bot_mask.any() else np.nan,
            "EW":  r.mean(),
        })
    df = pd.DataFrame(rows).sort_values("end_date")
    if len(df):
        df["LS"] = df["TOP"] - df["BOT"]
    return df


def summarise(name: str, df: pd.DataFrame) -> dict:
    tp = ann_stats(df["TOP"]); bt = ann_stats(df["BOT"])
    ls = ann_stats(df["LS"]);  ew = ann_stats(df["EW"])
    return dict(
        filter=name,
        months=ls["n"],
        mean_universe=df["n"].mean() if len(df) else np.nan,
        mean_n_top=df["n_top"].mean() if len(df) else np.nan,
        top_cagr=tp["ann_comp"], bot_cagr=bt["ann_comp"],
        ls_cagr=ls["ann_comp"], ls_vol=ls["ann_vol"],
        ls_sharpe=ls["sharpe"], ls_cum_x=ls["cum_x"],
        ew_cagr=ew["ann_comp"],
        nw_t_ls=newey_west_t(df["LS"].to_numpy(), 0),
        nw_t_top=newey_west_t(df["TOP"].to_numpy(), 0),
        nw_t_bot=newey_west_t(df["BOT"].to_numpy(), 0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ohlcv", type=Path,
                    default=Path("data/r1000_ohlcv_database.parquet"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading OHLCV …")
    ohlcv = pd.read_parquet(args.ohlcv)
    feats = compute_features(ohlcv)

    print(f"Loading predictions …")
    preds = pd.read_parquet(args.predictions)
    preds["end_date"] = pd.to_datetime(preds["end_date"])
    preds = preds.merge(feats, on=["ticker", "end_date"], how="left")

    # Per-date tertiles
    print("Bucketing (per end_date tertiles) …")
    preds["mcap_b"] = bucket_within_date(preds, "dv_60d", 3)
    preds["mom_b"]  = bucket_within_date(preds, "mom_12_1", 3)
    preds["vol_b"]  = bucket_within_date(preds, "vol_60d", 3)

    print(f"Preds: {len(preds):,}  "
          f"with all 3 features: "
          f"{preds[['mcap_b','mom_b','vol_b']].notna().all(axis=1).sum():,}")

    # Masks (small-cap, high-vol, recent-loser)
    small = (preds["mcap_b"] == "Low")
    highv = (preds["vol_b"]  == "High")
    loser = (preds["mom_b"]  == "Low")
    univ  = preds[["mcap_b","mom_b","vol_b"]].notna().all(axis=1)

    filter_specs = [
        ("universe (all R1000)",         univ),
        ("small_cap",                    univ & small),
        ("small_cap & high_vol",         univ & small & highv),
        ("small_cap & recent_loser",     univ & small & loser),
        ("high_vol & recent_loser",      univ & highv & loser),
        ("small_cap & high_vol & recent_loser", univ & small & highv & loser),
    ]

    # Build per-filter portfolio
    portfolios = {}
    summary_rows = []
    for name, mask in filter_specs:
        df = build_ls_50_50(preds, mask.to_numpy())
        portfolios[name] = df
        summary_rows.append(summarise(name, df))

    summary = pd.DataFrame(summary_rows)

    print()
    print("Per-filter 50/50 LS inside subset")
    cols = ["filter", "mean_universe", "mean_n_top", "months",
            "top_cagr", "bot_cagr", "ls_cagr", "ls_sharpe",
            "nw_t_ls", "ew_cagr"]
    print(summary[cols].round(2).to_string(index=False))
    print()

    # Save portfolios
    all_port = pd.concat(
        [df.assign(filter=name) for name, df in portfolios.items()],
        ignore_index=True)
    all_port.to_parquet(args.out_dir / "trash_tier_portfolios.parquet",
                         index=False)
    with pd.ExcelWriter(args.out_dir / "trash_tier_summary.xlsx",
                         engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="per_filter", index=False)

    # ── plot stacked cumulatives ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    axes = axes.flatten()
    for ax, (name, _) in zip(axes, filter_specs):
        df = portfolios[name].set_index("end_date")
        if len(df) == 0:
            ax.set_visible(False); continue
        cum = np.exp(df[["TOP","BOT","LS","EW"]].cumsum())
        ax.plot(cum.index, cum["TOP"], color="#2e7d32", lw=1.2, label="TOP")
        ax.plot(cum.index, cum["BOT"], color="#c62828", lw=1.2, label="BOT")
        ax.plot(cum.index, cum["EW"],  color="grey",  lw=1.0, ls="--", label="EW")
        ax.plot(cum.index, cum["LS"],  color="tab:blue", lw=1.6, label="LS")
        ax.set_yscale("log")
        row = summary[summary["filter"] == name].iloc[0]
        ax.set_title(f"{name}\n"
                     f"univ={row['mean_universe']:.0f}  "
                     f"LS {row['ls_cagr']:+.1f}%  "
                     f"Sh {row['ls_sharpe']:+.2f}  t {row['nw_t_ls']:+.2f}",
                     fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Trash-tier filter stack — Top 50% / Bot 50% LS",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf = args.out_dir / "trash_tier_cumulative.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {args.out_dir / 'trash_tier_summary.xlsx'}")


if __name__ == "__main__":
    main()
