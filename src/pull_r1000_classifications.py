"""
=============================================================================
SCRIPT NAME: pull_r1000_classifications.py
=============================================================================

INPUT FILES:
- runs/expanding/20260419_174908_cdef6809/predictions.parquet
  (source of unique tickers)

OUTPUT FILES:
- data/r1000_classifications.xlsx
- data/r1000_classifications.parquet
  Columns: ticker, name, gics_sector, gics_industry_group,
           gics_industry, gics_sub_industry,
           bics_level_1, bics_level_2, bics_level_3,
           industry_sector, industry_group, industry_subgroup,
           country, exchange

VERSION: 1.0
LAST UPDATED: 2026-04-20
AUTHOR: Arjun Divecha

DESCRIPTION:
Pulls sector/industry classifications from Bloomberg for every unique
ticker in the R1000 prediction universe (~3,000 names, including dead
tickers ending with "D US Equity").  We fetch BOTH GICS (MSCI/S&P
standard) and BICS (Bloomberg's own classification) plus basic
identifiers.  Incremental save per batch so a crash / disconnect does
not lose work.

DEPENDENCIES:
- OpusBloomberg (bbg.py)
- pandas, openpyxl

USAGE:
  conda run -p "/Users/arjundivecha/Dropbox/AAA Backup/A Working/OpusBloomberg/.venv" \
      python src/pull_r1000_classifications.py

NOTES:
- Requires Bloomberg Terminal open + logged in on Parallels Windows side.
- ~3,000 tickers  →  ~30 batches of 100  →  ~2-3 min.
=============================================================================
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/Users/arjundivecha/Dropbox/AAA Backup/A Working/OpusBloomberg")

from bbg import BBG, bloomberg_setup

log = logging.getLogger("pull_class")

PRED_PARQUET = ROOT / "runs" / "expanding" / "20260419_174908_cdef6809" / "predictions.parquet"
OUT_XLSX = ROOT / "data" / "r1000_classifications.xlsx"
OUT_PARQUET = ROOT / "data" / "r1000_classifications.parquet"
BATCH_DIR = ROOT / "data" / "classification_batches"
BATCH_SIZE = 100

FIELDS = [
    "NAME",
    # GICS (S&P/MSCI global standard)
    "GICS_SECTOR_NAME",
    "GICS_INDUSTRY_GROUP_NAME",
    "GICS_INDUSTRY_NAME",
    "GICS_SUB_INDUSTRY_NAME",
    # BICS (Bloomberg classification)
    "BICS_LEVEL_1_SECTOR_NAME",
    "BICS_LEVEL_2_INDUSTRY_GROUP_NAME",
    "BICS_LEVEL_3_INDUSTRY_NAME",
    # Generic legacy (should still populate for dead tickers)
    "INDUSTRY_SECTOR",
    "INDUSTRY_GROUP",
    "INDUSTRY_SUBGROUP",
    # Identification
    "EXCH_CODE",
    "COUNTRY_ISO",
]


def load_tickers() -> list[str]:
    df = pd.read_parquet(PRED_PARQUET, columns=["ticker"])
    t = sorted(df["ticker"].unique().tolist())
    log.info(f"Found {len(t)} unique tickers in {PRED_PARQUET.name}")
    return t


def pull_batch(bbg: BBG, tickers: list[str]) -> pd.DataFrame:
    """Pull one batch via ref_batch — returns tidy DataFrame."""
    data = bbg.ref_batch(tickers, FIELDS)
    # ref_batch returns {ticker: {field: value}} or {ticker: dict}
    rows = []
    for t in tickers:
        row = {"ticker": t}
        v = data.get(t, {}) if isinstance(data, dict) else {}
        if isinstance(v, dict):
            for f in FIELDS:
                row[f.lower()] = v.get(f)
        rows.append(row)
    df = pd.DataFrame(rows)
    # Rename long field names to friendly columns
    rename = {
        "name": "name",
        "gics_sector_name": "gics_sector",
        "gics_industry_group_name": "gics_industry_group",
        "gics_industry_name": "gics_industry",
        "gics_sub_industry_name": "gics_sub_industry",
        "bics_level_1_sector_name": "bics_level_1",
        "bics_level_2_industry_group_name": "bics_level_2",
        "bics_level_3_industry_name": "bics_level_3",
        "industry_sector": "industry_sector",
        "industry_group": "industry_group",
        "industry_subgroup": "industry_subgroup",
        "exch_code": "exchange",
        "country_iso": "country",
    }
    df = df.rename(columns=rename)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_tickers()

    # Resume: skip batches already written
    existing = list(BATCH_DIR.glob("batch_*.parquet"))
    done_tickers: set[str] = set()
    for f in existing:
        try:
            d = pd.read_parquet(f)
            done_tickers.update(d["ticker"].tolist())
        except Exception:
            pass
    todo = [t for t in tickers if t not in done_tickers]
    log.info(f"Already fetched: {len(done_tickers):,}  |  todo: {len(todo):,}")

    bloomberg_setup()

    with BBG() as bbg:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            batch_idx = i // BATCH_SIZE
            out = BATCH_DIR / f"batch_{batch_idx:04d}.parquet"
            log.info(f"Batch {batch_idx+1}/{(len(todo)+BATCH_SIZE-1)//BATCH_SIZE}  "
                     f"({len(batch)} tickers)")
            df = pull_batch(bbg, batch)
            df.to_parquet(out, index=False)

    # ── consolidate ────────────────────────────────────────────────────────
    frames = [pd.read_parquet(f) for f in sorted(BATCH_DIR.glob("batch_*.parquet"))]
    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)

    full.to_parquet(OUT_PARQUET, index=False)
    full.to_excel(OUT_XLSX, index=False)

    # Quick summary
    log.info(f"Wrote {OUT_PARQUET}  rows={len(full):,}")
    log.info(f"Wrote {OUT_XLSX}")
    log.info("")
    log.info("GICS Sector coverage:")
    log.info(full["gics_sector"].value_counts(dropna=False).to_string())
    log.info("")
    log.info("Missing GICS sector: "
             f"{full['gics_sector'].isna().sum():,} / {len(full):,}")


if __name__ == "__main__":
    main()
