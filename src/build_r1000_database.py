"""
=============================================================================
SCRIPT NAME: build_r1000_database.py
=============================================================================

INPUT FILES:
- None (pulls directly from Bloomberg)

OUTPUT FILES:
- data/constituents/year_YYYY.json      Per-year R1000 member ticker lists
- data/constituents/master_tickers.xlsx Consolidated unique ticker list + metadata
- data/ohlcv_batches/batch_NNN.parquet  Per-batch OHLCV staging files (resume-safe)
- data/r1000_ohlcv_database.parquet     Final combined database (Parquet, snappy)
- data/r1000_ohlcv_database.csv         Final combined database (CSV, PRD pipeline input)
- logs/build_YYYY_MM_DD_HHMMSS.log      Full processing log

VERSION: 1.0
LAST UPDATED: 2026-04-17
AUTHOR: Arjun Divecha

DESCRIPTION:
Builds a comprehensive point-in-time historical database of every stock that
was ever a Russell 1000 member from 1996 to present, including delisted and
inactive companies.

PHASE 1 — Constituents:
  For each year-end date 1996–present, calls Bloomberg INDX_MEMBERS BDS on
  "RIY Index" with END_DT override to get the index membership as of that date.
  Results are saved per-year as JSON for crash recovery.

PHASE 2 — Consolidate:
  Deduplicates all tickers across all years. Pulls NAME, GICS sector, exchange,
  and SEDOL for each unique ticker. Saves master_tickers.xlsx.

PHASE 3 — OHLCV:
  Pulls daily Open/High/Low/Close/Volume + TotalReturnIndex for every unique
  ticker from 19960101 to today. Uses hist_batch() to send 50 tickers per
  Bloomberg request (vs 1000 individual calls). Each batch is saved as a
  staging parquet immediately for crash recovery. Final step combines all
  batches into one master CSV and Parquet file.

DEPENDENCIES:
- blpapi, pandas, pyarrow, openpyxl
- BBGExtended (from BloomTest/src/bbg_extended.py)

USAGE:
  conda run -p "/Users/arjundivecha/Dropbox/AAA Backup/A Working/OpusBloomberg/.venv" \
      python src/build_r1000_database.py

NOTES:
- Resume-safe: already-fetched years and OHLCV batches are skipped
- Bloomberg Terminal must be running on Windows/Parallels before executing
- Expected runtime: 1-3 hours for full universe (~1,300 unique tickers)
- Expected output: ~3-6 GB CSV, ~1-2 GB Parquet
=============================================================================
"""

import sys
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ============================================================================
# PATH SETUP
# ============================================================================

BLOOMTEST_SRC = "/Users/arjundivecha/Dropbox/AAA Backup/A Working/BloomTest/src"
OPUS_BBG_PATH  = "/Users/arjundivecha/Dropbox/AAA Backup/A Working/OpusBloomberg"

sys.path.insert(0, BLOOMTEST_SRC)
sys.path.insert(0, OPUS_BBG_PATH)

from bbg_extended import BBGExtended, bloomberg_setup  # noqa: E402

# ============================================================================
# DIRECTORIES
# ============================================================================

BASE_DIR        = Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
CONST_DIR       = DATA_DIR / "constituents"
BATCH_DIR       = DATA_DIR / "ohlcv_batches"
LOGS_DIR        = BASE_DIR / "logs"

for d in [DATA_DIR, CONST_DIR, BATCH_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

log_file = LOGS_DIR / f"build_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

INDEX_TICKER = "RIY Index"   # Russell 1000 in Bloomberg
START_YEAR   = 1996
END_YEAR     = datetime.now().year
START_DATE   = "19960101"
END_DATE     = datetime.now().strftime("%Y%m%d")

# OHLCV fields to pull for each ticker
OHLCV_FIELDS = [
    "PX_OPEN",                     # Open
    "PX_HIGH",                     # High
    "PX_LOW",                      # Low
    "PX_LAST",                     # Close
    "PX_VOLUME",                   # Share volume
    "TOT_RETURN_INDEX_GROSS_DVDS", # Total-return index (dividends + splits)
]

# How many tickers per hist_batch call (~50 is reliable without timeout)
HIST_BATCH_SIZE = 50

# Metadata fields for Phase 2
META_FIELDS = ["NAME", "GICS_SECTOR_NAME", "GICS_INDUSTRY_NAME",
               "EXCH_CODE", "SEDOL1", "SECURITY_TYP"]

META_BATCH_SIZE = 100


# ============================================================================
# HELPERS
# ============================================================================

def year_end_date(year: int) -> str:
    """Return year-end date as YYYYMMDD (or today for the current year)."""
    if year >= datetime.now().year:
        return datetime.now().strftime("%Y%m%d")
    return f"{year}1231"


def convert_ticker(member_code: str) -> str:
    """
    Convert Bloomberg index member code (e.g. "AAPL UW") to equity ticker
    (e.g. "AAPL US Equity"). Exchange suffix (UW/UN/UQ/UA) is stripped and
    replaced with the standard composite "US Equity" suffix.
    """
    parts = member_code.strip().split()
    if len(parts) >= 2:
        ticker = " ".join(parts[:-1])  # drop exchange suffix
        return f"{ticker} US Equity"
    return f"{member_code} US Equity"


def safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ============================================================================
# PHASE 1: FETCH ANNUAL CONSTITUENTS
# ============================================================================

def phase1_get_constituents() -> Dict[int, List[str]]:
    """
    For each year from START_YEAR to END_YEAR, pull the R1000 members
    as of year-end via Bloomberg INDX_MWEIGHT_HIST with END_DATE_OVERRIDE.

    NOTE: INDX_MEMBERS silently ignores date overrides and always returns
    current members. INDX_MWEIGHT_HIST correctly returns point-in-time
    historical membership when END_DATE_OVERRIDE is supplied.

    Each year is saved to data/constituents/year_YYYY.json immediately.
    Already-cached years are skipped (resume-safe).

    Returns {year: [bbg_equity_ticker, ...]}
    """
    log.info("=" * 65)
    log.info("PHASE 1  —  Annual R1000 constituent snapshots")
    log.info("=" * 65)

    all_years = list(range(START_YEAR, END_YEAR + 1))
    all_constituents: Dict[int, List[str]] = {}

    # Load cached years
    for year in all_years:
        cache = CONST_DIR / f"year_{year}.json"
        if cache.exists():
            with open(cache) as f:
                all_constituents[year] = json.load(f)
            log.info(f"  {year}: loaded from cache  ({len(all_constituents[year])} members)")

    to_fetch = [y for y in all_years if y not in all_constituents]
    if not to_fetch:
        log.info("  All years already cached — skipping Phase 1 fetch.")
        return all_constituents

    log.info(f"  Fetching {len(to_fetch)} years from Bloomberg ...")

    with BBGExtended() as bbg:
        for year in to_fetch:
            dt = year_end_date(year)
            log.info(f"  {year}  (as of {dt}) ...")

            try:
                # INDX_MWEIGHT_HIST + END_DATE_OVERRIDE returns true historical members.
                # Field key is "Index Member" (vs current INDX_MWEIGHT which uses
                # "Member Ticker and Exchange Code").
                rows = bbg.bulk(INDEX_TICKER, "INDX_MWEIGHT_HIST",
                                overrides={"END_DATE_OVERRIDE": dt})

                tickers = []
                for row in rows:
                    code = (row.get("Index Member")
                            or row.get("Member Ticker and Exchange Code", ""))
                    if code.strip():
                        tickers.append(convert_ticker(code))

                all_constituents[year] = tickers

                cache = CONST_DIR / f"year_{year}.json"
                with open(cache, "w") as f:
                    json.dump(tickers, f)

                log.info(f"    → {len(tickers)} members  (saved to {cache.name})")

            except Exception as e:
                log.error(f"    FAILED: {e}")
                all_constituents[year] = []

            time.sleep(0.3)   # polite pause between requests

    total_years = len([y for y in all_years if all_constituents.get(y)])
    log.info(f"Phase 1 complete: {total_years} years with data.")
    return all_constituents


# ============================================================================
# PHASE 2: CONSOLIDATE TICKERS + METADATA
# ============================================================================

def phase2_consolidate(all_constituents: Dict[int, List[str]]) -> pd.DataFrame:
    """
    Build the unique-ticker master list with first/last year membership and
    Bloomberg metadata (name, sector, exchange, SEDOL).

    Saves data/constituents/master_tickers.xlsx.
    Returns DataFrame indexed by Ticker.
    """
    log.info("=" * 65)
    log.info("PHASE 2  —  Consolidate tickers + metadata")
    log.info("=" * 65)

    master_file = CONST_DIR / "master_tickers.xlsx"

    # Build ticker → sorted year list
    ticker_years: Dict[str, List[int]] = {}
    for year, tickers in all_constituents.items():
        for t in tickers:
            ticker_years.setdefault(t, []).append(year)

    unique_tickers = sorted(ticker_years.keys())
    log.info(f"  Unique tickers across all years: {len(unique_tickers)}")

    # Fetch metadata in batches of META_BATCH_SIZE
    meta_rows = []
    n_batches = (len(unique_tickers) + META_BATCH_SIZE - 1) // META_BATCH_SIZE

    with BBGExtended() as bbg:
        for i in range(0, len(unique_tickers), META_BATCH_SIZE):
            batch = unique_tickers[i:i + META_BATCH_SIZE]
            batch_num = i // META_BATCH_SIZE + 1
            log.info(f"  Metadata batch {batch_num}/{n_batches}  ({len(batch)} tickers) ...")

            try:
                result = bbg.ref_batch(batch, META_FIELDS)
            except Exception as e:
                log.error(f"    Metadata batch {batch_num} failed: {e}")
                result = {}

            for ticker in batch:
                data = result.get(ticker, {})
                yrs = sorted(ticker_years[ticker])
                meta_rows.append({
                    "Ticker":       ticker,
                    "Name":         data.get("NAME"),
                    "Sector":       data.get("GICS_SECTOR_NAME"),
                    "Industry":     data.get("GICS_INDUSTRY_NAME"),
                    "Exchange":     data.get("EXCH_CODE"),
                    "SEDOL":        data.get("SEDOL1"),
                    "SecurityType": data.get("SECURITY_TYP"),
                    "FirstYear":    yrs[0],
                    "LastYear":     yrs[-1],
                    "YearsInIndex": len(yrs),
                })

            time.sleep(0.2)

    df = pd.DataFrame(meta_rows).sort_values("Ticker").reset_index(drop=True)
    df.to_excel(master_file, index=False)
    log.info(f"  Saved master_tickers.xlsx  ({len(df)} tickers)")

    return df


# ============================================================================
# PHASE 3: PULL OHLCV VIA hist_batch
# ============================================================================

def phase3_ohlcv(master_df: pd.DataFrame) -> Path:
    """
    Pull daily OHLCV + total-return index for every ticker, 1996 → today.

    Sends HIST_BATCH_SIZE tickers per Bloomberg hist_batch request.
    Each batch result is saved immediately as a staging Parquet file.
    Already-completed batches are skipped (resume-safe).

    Finally combines all staging files into the master CSV + Parquet.
    Returns path to the final CSV.
    """
    log.info("=" * 65)
    log.info("PHASE 3  —  OHLCV pull via hist_batch")
    log.info("=" * 65)

    tickers = master_df["Ticker"].tolist()
    n_total = len(tickers)

    # Split tickers into fixed batches (so batch IDs are stable across runs)
    batches = [tickers[i:i + HIST_BATCH_SIZE]
               for i in range(0, n_total, HIST_BATCH_SIZE)]
    n_batches = len(batches)

    log.info(f"  {n_total} tickers  |  {n_batches} batches of {HIST_BATCH_SIZE}")

    with BBGExtended() as bbg:
        for batch_idx, batch in enumerate(batches):
            batch_num = batch_idx + 1
            batch_file = BATCH_DIR / f"batch_{batch_num:04d}.parquet"

            if batch_file.exists():
                log.info(f"  Batch {batch_num:4d}/{n_batches}  — already done, skipping")
                continue

            log.info(f"  Batch {batch_num:4d}/{n_batches}  ({len(batch)} tickers) ...")

            try:
                raw = bbg.hist_batch(batch, OHLCV_FIELDS, START_DATE, END_DATE)
            except Exception as e:
                log.error(f"    hist_batch failed: {e}")
                # Write empty file so this batch is marked as attempted
                pd.DataFrame().to_parquet(batch_file)
                continue

            # Assemble rows
            rows = []
            for ticker, points in raw.items():
                if not points:
                    continue
                # Skip error responses
                if len(points) == 1 and "error" in points[0]:
                    log.warning(f"    {ticker}: {points[0]['error']}")
                    continue
                for pt in points:
                    row = {"Ticker": ticker, "Date": pt.get("date")}
                    for f in OHLCV_FIELDS:
                        row[f] = safe_float(pt.get(f))
                    rows.append(row)

            if rows:
                df_batch = pd.DataFrame(rows)
                df_batch["Date"] = pd.to_datetime(df_batch["Date"])
                df_batch.to_parquet(batch_file, index=False, compression="snappy")
                log.info(f"    → {len(rows):,} rows saved to {batch_file.name}")
            else:
                log.warning(f"    Batch {batch_num}: no data rows — writing empty file")
                pd.DataFrame().to_parquet(batch_file)

    # ----- Combine all batches ------------------------------------------------
    log.info("")
    log.info("  Combining all batch files into master database ...")

    all_dfs = []
    for batch_file in sorted(BATCH_DIR.glob("batch_*.parquet")):
        try:
            df_b = pd.read_parquet(batch_file)
            if not df_b.empty:
                all_dfs.append(df_b)
        except Exception as e:
            log.error(f"  Failed to read {batch_file.name}: {e}")

    if not all_dfs:
        raise RuntimeError("No batch files to combine — check logs for errors.")

    log.info(f"  Combining {len(all_dfs)} non-empty batch files ...")
    master = pd.concat(all_dfs, ignore_index=True)

    # Rename columns to PRD spec
    master.rename(columns={
        "PX_OPEN":                      "Open",
        "PX_HIGH":                      "High",
        "PX_LOW":                       "Low",
        "PX_LAST":                      "Close",
        "PX_VOLUME":                    "Volume",
        "TOT_RETURN_INDEX_GROSS_DVDS":  "AdjClose",
    }, inplace=True)

    master.sort_values(["Ticker", "Date"], inplace=True)
    master.reset_index(drop=True, inplace=True)

    # Save parquet (primary, efficient)
    parquet_path = DATA_DIR / "r1000_ohlcv_database.parquet"
    master.to_parquet(parquet_path, index=False, compression="snappy")

    # Save CSV (direct input for PRD pipeline)
    csv_path = DATA_DIR / "r1000_ohlcv_database.csv"
    master.to_csv(csv_path, index=False)

    n_tickers  = master["Ticker"].nunique()
    date_min   = master["Date"].min().date()
    date_max   = master["Date"].max().date()
    n_rows     = len(master)
    size_gb    = csv_path.stat().st_size / 1e9

    log.info("")
    log.info("  ── Database summary ──────────────────────────────────")
    log.info(f"  Tickers   : {n_tickers:,}")
    log.info(f"  Date range: {date_min}  →  {date_max}")
    log.info(f"  Total rows: {n_rows:,}")
    log.info(f"  CSV size  : {size_gb:.2f} GB  →  {csv_path}")
    log.info(f"  Parquet   : {parquet_path}")
    log.info("  ──────────────────────────────────────────────────────")

    return csv_path


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    log.info("=" * 65)
    log.info("  R1000 HISTORICAL DATABASE BUILD")
    log.info(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Output  : {DATA_DIR}")
    log.info(f"  Log     : {log_file}")
    log.info("=" * 65)

    # Ensure Bloomberg is reachable before doing anything
    log.info("Setting up Bloomberg connection ...")
    bloomberg_setup(verbose=True)
    log.info("Bloomberg OK\n")

    # Phase 1: constituent snapshots per year
    all_constituents = phase1_get_constituents()

    # Phase 2: unique ticker list + metadata
    master_df = phase2_consolidate(all_constituents)

    # Phase 3: daily OHLCV for every ticker
    csv_path = phase3_ohlcv(master_df)

    log.info("")
    log.info("=" * 65)
    log.info(f"  BUILD COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Final CSV: {csv_path}")
    log.info("=" * 65)
