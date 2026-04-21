"""
=============================================================================
SCRIPT NAME: trash_tier_turnover.py
=============================================================================

INPUT FILES:
- --predictions PATH    predictions_monthly.parquet
- --ohlcv PATH          data/r1000_ohlcv_database.parquet

OUTPUT FILES:
- {out_dir}/turnover_summary.xlsx     per-filter TOP/BOT/LS turnover
- {out_dir}/turnover_monthly.parquet  monthly turnover series

DESCRIPTION:
Measures name-level turnover for each trash-tier filter portfolio.  At
every month-end we track the exact ticker list in the TOP and BOT
halves, then compute month-over-month turnover:

  turnover_t = |S_t  symmetric-diff  S_{t-1}| / (|S_t| + |S_{t-1}|)

where S_t is the set of tickers on the long (or short) side at t.
This gives a one-sided fraction in [0, 1] — 0 means identical
portfolio, 1 means full replacement.  Annual turnover ≈ 12 × mean(t).

Also reports average holding period (months) and portfolio concentration.

USAGE:
  python scripts/trash_tier_turnover.py \
      --predictions runs/expanding/.../predictions_monthly.parquet \
      --out-dir runs/expanding/.../backtest_trash_tier
=============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

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


def ticker_sets_by_date(preds: pd.DataFrame, mask: np.ndarray
                         ) -> dict[pd.Timestamp, tuple[set, set]]:
    """Return {date: (top_set, bot_set)} for the 50/50 split."""
    out = {}
    sub = preds.loc[mask]
    for date, g in sub.groupby("end_date"):
        if len(g) < 2:
            continue
        n = len(g)
        rank = g["p_up_mean"].rank(method="first") / n
        top = set(g.loc[rank >  0.5, "ticker"].tolist())
        bot = set(g.loc[rank <= 0.5, "ticker"].tolist())
        out[date] = (top, bot)
    return out


def turnover_stats(sets_by_date: dict[pd.Timestamp, tuple[set, set]]
                    ) -> pd.DataFrame:
    """One-sided turnover per side each month."""
    dates = sorted(sets_by_date.keys())
    rows = []
    prev_top, prev_bot = None, None
    for d in dates:
        top, bot = sets_by_date[d]
        row = {"end_date": d, "n_top": len(top), "n_bot": len(bot)}
        if prev_top is not None:
            # symmetric-diff / (|S_t| + |S_{t-1}|)  →  fraction replaced
            sym_t = (top ^ prev_top)
            sym_b = (bot ^ prev_bot)
            row["turn_top"] = len(sym_t) / max(len(top) + len(prev_top), 1)
            row["turn_bot"] = len(sym_b) / max(len(bot) + len(prev_bot), 1)
            # Jaccard = intersection / union for sanity
            row["jaccard_top"] = (len(top & prev_top)
                                    / max(len(top | prev_top), 1))
            row["jaccard_bot"] = (len(bot & prev_bot)
                                    / max(len(bot | prev_bot), 1))
        prev_top, prev_bot = top, bot
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ohlcv", type=Path,
                    default=Path("data/r1000_ohlcv_database.parquet"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OHLCV …")
    ohlcv = pd.read_parquet(args.ohlcv)
    feats = compute_features(ohlcv)

    print("Loading predictions …")
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
        ("universe (all R1000)",         univ),
        ("small_cap",                    univ & small),
        ("small_cap & high_vol",         univ & small & highv),
        ("small_cap & recent_loser",     univ & small & loser),
        ("high_vol & recent_loser",      univ & highv & loser),
        ("small_cap & high_vol & recent_loser", univ & small & highv & loser),
    ]

    summary = []
    all_monthly = []
    for name, mask in filter_specs:
        sets = ticker_sets_by_date(preds, mask.to_numpy())
        to = turnover_stats(sets)
        to["filter"] = name
        all_monthly.append(to)

        avg_top_names = to["n_top"].mean()
        avg_bot_names = to["n_bot"].mean()
        tt = to["turn_top"].mean()    # monthly one-sided
        tb = to["turn_bot"].mean()
        # Average holding period ≈ 1 / (2 × monthly turnover) months one-sided
        # (each month you rotate 2×tt names, so avg name stays 1/(2×tt) months)
        hold_top = 1.0 / (2 * tt) if tt > 0 else np.nan
        hold_bot = 1.0 / (2 * tb) if tb > 0 else np.nan
        j_top = to["jaccard_top"].mean()
        j_bot = to["jaccard_bot"].mean()
        summary.append({
            "filter":          name,
            "avg_n_top":       avg_top_names,
            "avg_n_bot":       avg_bot_names,
            "monthly_turn_top": tt,
            "monthly_turn_bot": tb,
            "annual_turn_top": tt * 12,
            "annual_turn_bot": tb * 12,
            "avg_hold_months_top": hold_top,
            "avg_hold_months_bot": hold_bot,
            "jaccard_top":     j_top,
            "jaccard_bot":     j_bot,
        })

    summary_df = pd.DataFrame(summary)
    monthly_df = pd.concat(all_monthly, ignore_index=True)

    print()
    print("Turnover per filter (50/50 split)")
    cols = ["filter", "avg_n_top", "monthly_turn_top", "monthly_turn_bot",
            "annual_turn_top", "annual_turn_bot",
            "avg_hold_months_top", "avg_hold_months_bot"]
    print(summary_df[cols].round(3).to_string(index=False))
    print()

    with pd.ExcelWriter(args.out_dir / "turnover_summary.xlsx",
                         engine="openpyxl") as xw:
        summary_df.to_excel(xw, sheet_name="per_filter", index=False)
        monthly_df.to_excel(xw, sheet_name="monthly", index=False)
    monthly_df.to_parquet(args.out_dir / "turnover_monthly.parquet", index=False)
    print(f"wrote {args.out_dir / 'turnover_summary.xlsx'}")


if __name__ == "__main__":
    main()
