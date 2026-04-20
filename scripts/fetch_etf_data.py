"""
=============================================================================
SCRIPT NAME: fetch_etf_data.py
=============================================================================

INPUT FILES:
- AssetList.xlsx                     Single column 'Ticker' (34 ETFs)

OUTPUT FILES:
- data/etf_ohlcv.csv                 Ticker, Date, Open, High, Low, Close, Volume, AdjClose
- data/etf_ohlcv.parquet             same

DESCRIPTION:
Pull daily OHLCV for every ticker in AssetList.xlsx from yfinance, max history
available (typically inception → today).  Output schema matches
r1000_ohlcv_database.csv so the existing imaging/renderer/cache pipeline can
consume it unchanged.

USAGE:
  python scripts/fetch_etf_data.py
=============================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]


def fetch_one(ticker: str) -> pd.DataFrame | None:
    df = yf.download(ticker, period="max", interval="1d",
                     auto_adjust=False, progress=False, threads=False)
    if df is None or len(df) == 0:
        print(f"  ✗ {ticker}: no data")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(columns={
        "Date": "Date",
        "Adj Close": "AdjClose",
    })
    keep = ["Date", "Open", "High", "Low", "Close", "Volume", "AdjClose"]
    df = df[keep].copy()
    df["Ticker"] = ticker
    df = df[["Ticker"] + keep]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "AdjClose"])
    print(f"  ✓ {ticker}: {len(df):,} rows   "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-xlsx", type=Path, default=ROOT / "AssetList.xlsx",
                    help="xlsx with a 'Ticker' column")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "data" / "etf_ohlcv.csv")
    ap.add_argument("--out-parquet", type=Path, default=ROOT / "data" / "etf_ohlcv.parquet")
    args = ap.parse_args()

    tickers = pd.read_excel(args.asset_xlsx)["Ticker"].astype(str).str.strip().tolist()
    print(f"Fetching {len(tickers)} tickers from yfinance…\n")

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for t in tickers:
        df = fetch_one(t)
        if df is None:
            missing.append(t)
        else:
            frames.append(df)

    if not frames:
        sys.exit("No data fetched.  Check network / tickers.")

    out = pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Date"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    out.to_parquet(args.out_parquet, index=False)

    print()
    print(f"wrote {args.out_csv}  rows={len(out):,}  tickers={out['Ticker'].nunique()}")
    print(f"wrote {args.out_parquet}")
    if missing:
        print(f"missing: {missing}")


if __name__ == "__main__":
    main()
