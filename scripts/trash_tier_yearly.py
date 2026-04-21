"""
=============================================================================
SCRIPT NAME: trash_tier_yearly.py
=============================================================================

INPUT FILES:
- --predictions PATH    predictions_monthly.parquet
- --ohlcv PATH          data/r1000_ohlcv_database.parquet

OUTPUT FILES:
- {out_dir}/yearly_turnover_returns.xlsx
- {out_dir}/yearly_turnover_returns.pdf

DESCRIPTION:
For each trash-tier filter, compute per-calendar-year:
  - mean monthly turnover (TOP and BOT)
  - mean universe size
  - 50/50 LS return over the 12 months (sum of log returns → CAGR)
  - TOP / BOT / EW returns

Shows whether the high-turnover regime is uniform across time or
concentrated in particular years (e.g., dot-com, GFC, COVID, post-COVID).

USAGE:
  python scripts/trash_tier_yearly.py \
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
    df["dv"] = df["Close"] * df["Volume"]
    df["dv_60d"] = g["dv"].transform(
        lambda s: s.rolling(60, min_periods=15).mean())
    df["logp"] = np.log(df["AdjClose"].replace(0, np.nan))
    df["mom_12_1"] = g["logp"].shift(21) - g["logp"].shift(252 + 21)
    df["ret"] = g["AdjClose"].pct_change()
    df["vol_60d"] = g["ret"].transform(
        lambda s: s.rolling(60, min_periods=20).std() * np.sqrt(252))
    return df[["Ticker", "Date", "dv_60d", "mom_12_1", "vol_60d"]].rename(
        columns={"Ticker": "ticker", "Date": "end_date"})


def bucket_within_date(df: pd.DataFrame, col: str, n: int = 3) -> pd.Series:
    labels = ["Low", "Mid", "High"] if n == 3 else [f"B{i+1}" for i in range(n)]
    def assign(g):
        if g[col].notna().sum() < n:
            return pd.Series([pd.NA] * len(g), index=g.index)
        return pd.qcut(g[col].rank(method="first"), n,
                       labels=labels, duplicates="drop")
    return df.groupby("end_date", group_keys=False).apply(assign)


def build_portfolio(preds: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Monthly TOP/BOT/LS/EW returns + ticker sets."""
    rows = []
    sub = preds.loc[mask]
    for date, g in sub.groupby("end_date"):
        if len(g) < 2:
            continue
        n = len(g)
        rank = g["p_up_mean"].rank(method="first") / n
        top = g[rank >  0.5]
        bot = g[rank <= 0.5]
        rows.append({
            "end_date": date,
            "n":        n,
            "n_top":    len(top),
            "n_bot":    len(bot),
            "TOP":      top["forward_return"].mean(),
            "BOT":      bot["forward_return"].mean(),
            "EW":       g["forward_return"].mean(),
            "top_set":  frozenset(top["ticker"]),
            "bot_set":  frozenset(bot["ticker"]),
        })
    df = pd.DataFrame(rows).sort_values("end_date").reset_index(drop=True)
    if len(df):
        df["LS"] = df["TOP"] - df["BOT"]
        # one-sided turnover: symmetric diff / (|S_t| + |S_{t-1}|)
        top_turn = [np.nan]
        bot_turn = [np.nan]
        for i in range(1, len(df)):
            pt, pb = df.loc[i-1, "top_set"], df.loc[i-1, "bot_set"]
            ct, cb = df.loc[i,   "top_set"], df.loc[i,   "bot_set"]
            top_turn.append(len(ct ^ pt) / max(len(ct) + len(pt), 1))
            bot_turn.append(len(cb ^ pb) / max(len(cb) + len(pb), 1))
        df["turn_top"] = top_turn
        df["turn_bot"] = bot_turn
    return df.drop(columns=["top_set", "bot_set"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ohlcv", type=Path,
                    default=Path("data/r1000_ohlcv_database.parquet"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OHLCV + predictions …")
    ohlcv = pd.read_parquet(args.ohlcv)
    feats = compute_features(ohlcv)
    preds = pd.read_parquet(args.predictions)
    preds["end_date"] = pd.to_datetime(preds["end_date"])
    preds = preds.merge(feats, on=["ticker", "end_date"], how="left")

    print("Bucketing …")
    preds["mcap_b"] = bucket_within_date(preds, "dv_60d", 3)
    preds["mom_b"]  = bucket_within_date(preds, "mom_12_1", 3)
    preds["vol_b"]  = bucket_within_date(preds, "vol_60d", 3)

    small = (preds["mcap_b"] == "Low")
    highv = (preds["vol_b"]  == "High")
    loser = (preds["mom_b"]  == "Low")
    univ  = preds[["mcap_b","mom_b","vol_b"]].notna().all(axis=1)

    filter_specs = [
        ("universe",                 univ),
        ("small",                    univ & small),
        ("small&highvol",            univ & small & highv),
        ("small&loser",              univ & small & loser),
        ("highvol&loser",            univ & highv & loser),
        ("triple",                   univ & small & highv & loser),
    ]

    # Per-filter yearly summary
    yearly_tables = {}
    for name, mask in filter_specs:
        port = build_portfolio(preds, mask.to_numpy())
        if len(port) == 0:
            continue
        port["year"] = port["end_date"].dt.year
        ann = (port.groupby("year")
                    .agg(months=("end_date", "count"),
                         univ=("n", "mean"),
                         n_top=("n_top", "mean"),
                         turn_top=("turn_top", "mean"),
                         turn_bot=("turn_bot", "mean"),
                         top_sum=("TOP", "sum"),
                         bot_sum=("BOT", "sum"),
                         ew_sum=("EW",  "sum"),
                         ls_sum=("LS",  "sum"))
                    .reset_index())
        # Convert monthly log-return sum → annual simple return
        ann["top_ret"] = np.expm1(ann["top_sum"])     * 100  # %
        ann["bot_ret"] = np.expm1(ann["bot_sum"])     * 100
        ann["ls_ret"]  = np.expm1(ann["ls_sum"])      * 100
        ann["ew_ret"]  = np.expm1(ann["ew_sum"])      * 100
        ann["turn_top_pct"] = ann["turn_top"] * 100
        ann["turn_bot_pct"] = ann["turn_bot"] * 100
        ann["filter"] = name
        yearly_tables[name] = ann

    # Write Excel — one sheet per filter
    with pd.ExcelWriter(args.out_dir / "yearly_turnover_returns.xlsx",
                         engine="openpyxl") as xw:
        for name, df in yearly_tables.items():
            out = df[["year", "months", "univ", "n_top",
                      "turn_top_pct", "turn_bot_pct",
                      "top_ret", "bot_ret", "ls_ret", "ew_ret"]]
            out.round(2).to_excel(xw, sheet_name=name[:31], index=False)
        # Combined yearly LS-return × filter crosstab
        combo_ls = pd.concat(
            [d.set_index("year")["ls_ret"].rename(name)
             for name, d in yearly_tables.items()], axis=1).round(1)
        combo_ls.to_excel(xw, sheet_name="yearly_ls_crosstab")
        # Combined yearly turnover (top) crosstab
        combo_turn = pd.concat(
            [d.set_index("year")["turn_top_pct"].rename(name)
             for name, d in yearly_tables.items()], axis=1).round(1)
        combo_turn.to_excel(xw, sheet_name="yearly_turn_top_crosstab")
        combo_turn_b = pd.concat(
            [d.set_index("year")["turn_bot_pct"].rename(name)
             for name, d in yearly_tables.items()], axis=1).round(1)
        combo_turn_b.to_excel(xw, sheet_name="yearly_turn_bot_crosstab")
    print(f"wrote {args.out_dir / 'yearly_turnover_returns.xlsx'}")

    # Print headline crosstabs
    print()
    print("=" * 78)
    print("Yearly LS returns (%) — per filter")
    print("=" * 78)
    print(combo_ls.to_string())
    print()
    print("=" * 78)
    print("Yearly monthly-turnover (%, TOP side) — per filter")
    print("=" * 78)
    print(combo_turn.to_string())
    print()
    print("=" * 78)
    print("Yearly monthly-turnover (%, BOT side) — per filter")
    print("=" * 78)
    print(combo_turn_b.to_string())

    # ── Plot: two-panel (yearly LS returns, yearly turnover) per filter ──
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for name, df in yearly_tables.items():
        axes[0].plot(df["year"], df["ls_ret"], marker="o", label=name, lw=1.1)
        axes[1].plot(df["year"], df["turn_top_pct"], marker="o", label=name, lw=1.1)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel("Annual LS return (%)")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8, ncol=3)
    axes[0].set_title("Yearly LS returns by filter")
    axes[1].set_ylabel("Mean monthly TOP turnover (%)")
    axes[1].set_xlabel("year")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8, ncol=3)
    axes[1].set_title("Yearly TOP-side monthly turnover by filter")
    plt.tight_layout()
    out_pdf = args.out_dir / "yearly_turnover_returns.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
