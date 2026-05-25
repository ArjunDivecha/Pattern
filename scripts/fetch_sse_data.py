"""
=============================================================================
SCRIPT NAME: fetch_sse_data.py
=============================================================================

INPUT FILES:
- data/sse_pairs_seed.csv   underlying,long_etf,short_etf,leverage,sponsor

OUTPUT FILES:
- data/sse_underlying_ohlcv.csv    OHLCV for each underlying stock (full history)
- data/sse_wrapper_ohlcv.csv       OHLCV for each ETF wrapper (since inception)
- data/sse_pairs.csv               validated pair table (drops delisted)

DESCRIPTION:
Pulls daily OHLCV via yfinance for both the underlying stocks (long history
needed for CNN scoring) and the leveraged ETF wrappers (since inception, used
for realized returns).  Outputs in the same r1000 schema the renderer / loader
expects: Ticker, Date, Open, High, Low, Close, Volume, AdjClose.

USAGE:
  python scripts/fetch_sse_data.py
=============================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]


def fetch_block(tickers, period="max"):
    bulk = yf.download(tickers, period=period, interval="1d",
                       progress=True, auto_adjust=False, threads=False,
                       group_by="ticker")
    out = []
    for t in tickers:
        try:
            sub = bulk[t] if isinstance(bulk.columns, pd.MultiIndex) else bulk
            df = sub.reset_index()
            df.columns = [c if c != "index" else "Date" for c in df.columns]
            keep = {"Date": "Date", "Open": "Open", "High": "High", "Low": "Low",
                    "Close": "Close", "Volume": "Volume", "Adj Close": "AdjClose"}
            df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
            df = df.dropna(subset=["Close"])
            if len(df) == 0:
                print(f"  [warn] {t}: no rows")
                continue
            df.insert(0, "Ticker", t)
            out.append(df)
        except Exception as e:
            print(f"  [skip] {t}: {e}")
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def main():
    pairs = pd.read_csv(ROOT / "data" / "sse_pairs_seed.csv")

    underlyings = sorted(set(pairs["underlying"]))
    wrappers = sorted(set(pairs["long_etf"].dropna()).union(
        pairs["short_etf"].dropna()))

    print(f"Fetching {len(underlyings)} underlyings...")
    und = fetch_block(underlyings, period="max")
    print(f"  -> {len(und):,} rows, {und['Ticker'].nunique()} tickers")

    print(f"\nFetching {len(wrappers)} wrappers...")
    wr = fetch_block(wrappers, period="max")
    print(f"  -> {len(wr):,} rows, {wr['Ticker'].nunique()} tickers")

    # Drop wrappers / underlyings that returned nothing
    valid_und = set(und["Ticker"].unique())
    valid_wr = set(wr["Ticker"].unique())

    # Filter pairs to those whose underlying + long_etf both validate
    pairs_v = pairs[pairs["underlying"].isin(valid_und)
                    & pairs["long_etf"].isin(valid_wr)].copy()
    # Mark short side as missing if delisted
    pairs_v["short_etf_valid"] = pairs_v["short_etf"].apply(
        lambda x: x if (pd.notna(x) and x in valid_wr) else "")

    # Save
    out_und = ROOT / "data" / "sse_underlying_ohlcv.csv"
    out_wr  = ROOT / "data" / "sse_wrapper_ohlcv.csv"
    out_pa  = ROOT / "data" / "sse_pairs.csv"
    und.to_csv(out_und, index=False)
    wr.to_csv(out_wr, index=False)
    pairs_v.to_csv(out_pa, index=False)

    print(f"\nwrote {out_und}")
    print(f"wrote {out_wr}")
    print(f"wrote {out_pa}")

    print(f"\nValid pair-rows: {len(pairs_v)}")
    print(f"  underlyings: {pairs_v['underlying'].nunique()}")
    print(f"  complete pairs (long + short both valid): "
          f"{(pairs_v['short_etf_valid'].str.len() > 0).sum()}")


if __name__ == "__main__":
    main()
