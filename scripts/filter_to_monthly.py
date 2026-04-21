"""
=============================================================================
SCRIPT NAME: filter_to_monthly.py
=============================================================================

INPUT FILES:
- --predictions PATH   predictions.parquet with ticker, end_date, p_up_mean,
                       forward_return, label (+ optional p_up_0..4, window,
                       p_up_std)

OUTPUT FILES:
- --out PATH           monthly-subsampled parquet (same schema).  Default:
                       writes next to the input with '_monthly.parquet'.

DESCRIPTION:
Keeps only the LAST trading day of each calendar month per ticker.  This
converts a daily overlapping-horizon predictions file into a monthly
non-overlapping cadence suitable for buy-and-hold-a-month backtests.

The forward_return column is left as-is — it remains the 20-trading-day
forward log return from each month-end, which is what the model was
trained to predict.

USAGE:
  python scripts/filter_to_monthly.py \
      --predictions runs/expanding/20260419_174908_cdef6809/predictions.parquet
=============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def filter_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows on the CANONICAL last trading day of each calendar
    month.  The canonical day is the latest date appearing anywhere in the
    dataset for that month (i.e., the last exchange trading day).  Tickers
    that do not have a row on that exact date (delisted, IPO mid-month,
    holiday mismatch) are dropped for that month.  This ensures every kept
    date has a broad cross-section, so portfolio construction is clean.
    """
    df = df.copy()
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["_ym"] = df["end_date"].dt.to_period("M")
    canonical = df.groupby("_ym")["end_date"].transform("max")
    keep = df["end_date"] == canonical
    monthly = df.loc[keep].drop(columns="_ym").reset_index(drop=True)
    return monthly


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = args.predictions.with_name(
            args.predictions.stem + "_monthly.parquet"
        )

    df = pd.read_parquet(args.predictions)
    print(f"Input : {args.predictions}")
    print(f"  rows={len(df):,}  tickers={df['ticker'].nunique():,}  "
          f"{df['end_date'].min()} → {df['end_date'].max()}")

    monthly = filter_to_monthly(df)
    print(f"Monthly: rows={len(monthly):,}  tickers={monthly['ticker'].nunique():,}  "
          f"unique dates={monthly['end_date'].nunique():,}")
    print(f"  mean rows/date: {len(monthly) / max(monthly['end_date'].nunique(),1):.1f}")

    # Sanity: check dates are ~21 days apart on average within a ticker
    per_ticker_gaps = (monthly.groupby("ticker")["end_date"]
                             .apply(lambda s: s.diff().dt.days.mean()))
    print(f"  mean gap days (within ticker): {per_ticker_gaps.mean():.1f} "
          f"(expected ~30 days)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
