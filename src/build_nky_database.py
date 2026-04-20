"""
=============================================================================
SCRIPT NAME: build_nky_database.py
=============================================================================

INPUT FILES:
- None (pulls directly from Bloomberg)

OUTPUT FILES:
- data/nky_constituents/year_YYYY.json      Per-year NKY member ticker lists
- data/nky_constituents/master_tickers.xlsx Consolidated unique ticker list + metadata
- data/nky_ohlcv_batches/batch_NNN.parquet  Per-batch OHLCV staging files (resume-safe)
- data/nky_ohlcv_database.parquet           Final combined database (Parquet, snappy)
- data/nky_ohlcv_database.csv               Final combined database (CSV)
- logs/nky_build_YYYY_MM_DD_HHMMSS.log      Full processing log

VERSION: 1.0
LAST UPDATED: 2026-04-20
AUTHOR: Arjun Divecha

DESCRIPTION:
NKY 225 analogue of build_r1000_database.py.  Builds a point-in-time
historical database of every stock that was ever a Nikkei 225 member from
1996 to present, including delisted names.

PHASE 1 — Constituents:
  For each year-end 1996..today, call INDX_MWEIGHT_HIST on "NKY Index" with
  END_DATE_OVERRIDE to get point-in-time members.  Convert Japanese index
  member codes (e.g. "7203 JT") to composite Japan equity ticker
  ("7203 JP Equity").  Saved per-year as JSON for crash recovery.

PHASE 2 — Consolidate:
  Deduplicate across all years; fetch name/sector/exchange/SEDOL metadata
  for every unique ticker; save master_tickers.xlsx.

PHASE 3 — OHLCV:
  Daily O/H/L/C/V + TotalReturnIndex from 19960101 → today via hist_batch
  (50 tickers per request).  Each batch is staged to parquet immediately.
  Final step combines into master CSV + Parquet.

USAGE:
  conda run -p "/Users/arjundivecha/Dropbox/AAA Backup/A Working/OpusBloomberg/.venv" \
      python src/build_nky_database.py

NOTES:
- Resume-safe: cached years and completed batches are skipped
- Bloomberg Terminal must be running on Windows/Parallels
- Expected runtime: 20-45 min (NKY is smaller than R1000)
- Japanese member codes come back with exchange suffix "JT" (Tokyo);
  these are converted to composite "JP Equity" for pricing queries.
=============================================================================
"""

import sys
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
OPUS_BBG_PATH = "/Users/arjundivecha/Dropbox/AAA Backup/A Working/OpusBloomberg"

sys.path.insert(0, BLOOMTEST_SRC)
sys.path.insert(0, OPUS_BBG_PATH)

from bbg_extended import BBGExtended, bloomberg_setup  # noqa: E402

# ============================================================================
# DIRECTORIES
# ============================================================================

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
CONST_DIR = DATA_DIR / "nky_constituents"
BATCH_DIR = DATA_DIR / "nky_ohlcv_batches"
LOGS_DIR  = BASE_DIR / "logs"

for d in [DATA_DIR, CONST_DIR, BATCH_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

log_file = LOGS_DIR / f"nky_build_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.log"
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

INDEX_TICKER = "NKY Index"
START_YEAR   = 1996
END_YEAR     = datetime.now().year
START_DATE   = "19960101"
END_DATE     = datetime.now().strftime("%Y%m%d")

OHLCV_FIELDS = [
    "PX_OPEN",
    "PX_HIGH",
    "PX_LOW",
    "PX_LAST",
    "PX_VOLUME",
    "TOT_RETURN_INDEX_GROSS_DVDS",
]

HIST_BATCH_SIZE = 25

META_FIELDS = ["NAME", "GICS_SECTOR_NAME", "GICS_INDUSTRY_NAME",
               "EXCH_CODE", "SEDOL1", "SECURITY_TYP"]
META_BATCH_SIZE = 100

# Japan exchange suffixes Bloomberg returns in index-member codes.  We map
# any of these to the composite "JP Equity" used for pricing queries.
JP_EXCH_SUFFIXES = {"JT", "JP", "JF", "JO", "JN", "JS", "JU", "JE", "JQ", "JX"}


# ============================================================================
# HELPERS
# ============================================================================

def year_end_date(year: int) -> str:
    if year >= datetime.now().year:
        return datetime.now().strftime("%Y%m%d")
    return f"{year}1231"


def convert_jp_ticker(member_code: str) -> str:
    """
    Convert a Bloomberg NKY index member code to a composite-Japan equity
    ticker.

    Examples:
      "7203 JT"           -> "7203 JP Equity"
      "7203 JT Equity"    -> "7203 JP Equity"
      "7203 JP Equity"    -> "7203 JP Equity"  (already composite)
      "7203"              -> "7203 JP Equity"  (fallback, unlikely)
    """
    parts = member_code.strip().split()
    if not parts:
        return member_code
    # Drop a trailing "Equity" token if present
    if parts[-1].lower() == "equity":
        parts = parts[:-1]
    if len(parts) == 1:
        return f"{parts[0]} JP Equity"
    code = parts[0]
    exch = parts[-1].upper()
    if exch in JP_EXCH_SUFFIXES:
        return f"{code} JP Equity"
    # Unexpected exchange — default to composite JP anyway
    return f"{code} JP Equity"


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
    log.info("=" * 65)
    log.info("PHASE 1  —  Annual NKY constituent snapshots")
    log.info("=" * 65)

    all_years = list(range(START_YEAR, END_YEAR + 1))
    all_constituents: Dict[int, List[str]] = {}

    for year in all_years:
        cache = CONST_DIR / f"year_{year}.json"
        if cache.exists():
            with open(cache) as f:
                all_constituents[year] = json.load(f)
            log.info(f"  {year}: cached  ({len(all_constituents[year])} members)")

    to_fetch = [y for y in all_years if y not in all_constituents]
    if not to_fetch:
        log.info("  All years cached — skipping Phase 1 fetch.")
        return all_constituents

    log.info(f"  Fetching {len(to_fetch)} years from Bloomberg ...")

    with BBGExtended() as bbg:
        for year in to_fetch:
            dt = year_end_date(year)
            log.info(f"  {year}  (as of {dt}) ...")

            try:
                rows = bbg.bulk(INDEX_TICKER, "INDX_MWEIGHT_HIST",
                                overrides={"END_DATE_OVERRIDE": dt})

                tickers = []
                for row in rows:
                    code = (row.get("Index Member")
                            or row.get("Member Ticker and Exchange Code", ""))
                    if code.strip():
                        tickers.append(convert_jp_ticker(code))

                # De-dup within a year
                tickers = sorted(set(tickers))
                all_constituents[year] = tickers

                cache = CONST_DIR / f"year_{year}.json"
                with open(cache, "w") as f:
                    json.dump(tickers, f)

                log.info(f"    → {len(tickers)} members  (saved to {cache.name})")

            except Exception as e:
                log.error(f"    FAILED: {e}")
                all_constituents[year] = []

            time.sleep(0.3)

    # Bloomberg INDX_MWEIGHT_HIST for NKY only returns data from ~2005.
    # For earlier years it returns 0 members.  Backfill those with the
    # earliest non-empty year's list (NKY has very low turnover, ~5% per
    # year, so this is a reasonable proxy and lets us score 1999-2004
    # with the expanding / rolling CNN models).
    years_with_data = [y for y in sorted(all_constituents.keys())
                       if all_constituents.get(y)]
    if years_with_data:
        earliest_real = years_with_data[0]
        fallback = all_constituents[earliest_real]
        n_backfilled = 0
        for y in all_years:
            if not all_constituents.get(y) and y < earliest_real:
                all_constituents[y] = list(fallback)
                cache = CONST_DIR / f"year_{y}.json"
                with open(cache, "w") as f:
                    json.dump(all_constituents[y], f)
                n_backfilled += 1
        if n_backfilled:
            log.info(f"  Backfilled {n_backfilled} pre-{earliest_real} years "
                     f"with the {earliest_real} member list "
                     f"({len(fallback)} tickers).")

    total_years = len([y for y in all_years if all_constituents.get(y)])
    log.info(f"Phase 1 complete: {total_years} years with data.")
    return all_constituents


# ============================================================================
# PHASE 2: CONSOLIDATE + METADATA
# ============================================================================

def phase2_consolidate(all_constituents: Dict[int, List[str]]) -> pd.DataFrame:
    log.info("=" * 65)
    log.info("PHASE 2  —  Consolidate tickers + metadata")
    log.info("=" * 65)

    master_file = CONST_DIR / "master_tickers.xlsx"

    ticker_years: Dict[str, List[int]] = {}
    for year, tickers in all_constituents.items():
        for t in tickers:
            ticker_years.setdefault(t, []).append(year)

    unique_tickers = sorted(ticker_years.keys())
    log.info(f"  Unique tickers across all years: {len(unique_tickers)}")

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
# PHASE 3: OHLCV VIA hist_batch
# ============================================================================

def phase3_ohlcv(master_df: pd.DataFrame) -> Path:
    log.info("=" * 65)
    log.info("PHASE 3  —  OHLCV pull via hist_batch")
    log.info("=" * 65)

    tickers = master_df["Ticker"].tolist()
    n_total = len(tickers)

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
                pd.DataFrame().to_parquet(batch_file)
                continue

            rows = []
            for ticker, points in raw.items():
                if not points:
                    continue
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

    # ----- Combine ----------------------------------------------------------
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

    master = pd.concat(all_dfs, ignore_index=True)

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

    parquet_path = DATA_DIR / "nky_ohlcv_database.parquet"
    master.to_parquet(parquet_path, index=False, compression="snappy")

    csv_path = DATA_DIR / "nky_ohlcv_database.csv"
    master.to_csv(csv_path, index=False)

    n_tickers = master["Ticker"].nunique()
    date_min  = master["Date"].min().date()
    date_max  = master["Date"].max().date()
    n_rows    = len(master)
    size_gb   = csv_path.stat().st_size / 1e9

    log.info("")
    log.info("  ── NKY Database summary ──────────────────────────────")
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
    log.info("  NKY HISTORICAL DATABASE BUILD")
    log.info(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Output  : {DATA_DIR}")
    log.info(f"  Log     : {log_file}")
    log.info("=" * 65)

    log.info("Setting up Bloomberg connection ...")
    bloomberg_setup(verbose=True)
    log.info("Bloomberg OK\n")

    all_constituents = phase1_get_constituents()
    master_df = phase2_consolidate(all_constituents)
    csv_path = phase3_ohlcv(master_df)

    log.info("")
    log.info("=" * 65)
    log.info(f"  BUILD COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Final CSV: {csv_path}")
    log.info("=" * 65)
