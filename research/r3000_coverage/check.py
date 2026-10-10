"""
=============================================================================
SCRIPT NAME: check.py  (research/r3000_coverage)
=============================================================================

DESCRIPTION:
    Survivorship-coverage check for the Russell 3000 rebuild of Pattern.
    For every month-end index membership record in the Global Data Mart
    (point-in-time, monthly from 1996), asks whether the per-security price
    cache actually holds that name around that date:
      - priced:    a price row in the 7 calendar days up to the month end
      - chartable: at least 39 sessions with a positive close in the 70
                   calendar days up to the month end (I20 image + 20-day MA)
      - no_cache:  the member has no price-cache file at all
    Reports these shares by year, so missing names (usually delisted ones)
    are visible rather than averaged away. Read-only on the Data Mart; no
    Bloomberg calls.

INPUT FILES:
    --membership (default: full RAY point-in-time membership, 11k names)
      /Users/arjundivecha/Dropbox/AAA Backup/A Working/Visual Pattern Genre/data/ray_membership_pit.parquet
      (the vintage's own membership.parquet is filtered to its 6,784-name
       universe and so omits the October 2026 names -- do not use it here)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/_cache/prices/securities/*.parquet

OUTPUT FILES (in a new timestamped run dir):
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_coverage/runs/<UTC stamp>/coverage_by_year.parquet (+ .xlsx)
    .../member_months.parquet   one row per member-month with the flags
    .../missing_names.xlsx      members never priced while in the index
    .../summary.json, check.log

VERSION: 1.1   LAST UPDATED: 2026-10-10   AUTHOR: Claude (for Arjun Divecha)
DEPENDENCIES: pandas, pyarrow, numpy, openpyxl
USAGE: .venv/bin/python research/r3000_coverage/check.py [--membership <parquet>]
=============================================================================
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

V = Path("/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30")
RAY = Path("/Users/arjundivecha/Dropbox/AAA Backup/A Working/Visual Pattern Genre/data/ray_membership_pit.parquet")
OUT = Path(__file__).resolve().parent / "runs" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
OUT.mkdir(parents=True, exist_ok=False)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "check.log", "a") as f:
        f.write(line + "\n")


def load(path):
    t = pq.read_table(path, columns=["security", "date", "close"]).to_pandas()
    t = t[np.isfinite(t["close"]) & (t["close"] > 0)]
    return t[["security", "date"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", type=Path, default=RAY)
    args = ap.parse_args()
    log(f"membership source {args.membership}")
    mem = pd.read_parquet(args.membership, columns=["date", "security"])
    mem["date"] = pd.to_datetime(mem["date"])
    log(f"membership rows {len(mem):,}, securities {mem.security.nunique():,}, "
        f"{mem.date.min().date()} to {mem.date.max().date()}")

    files = glob.glob(str(V / "_cache/prices/securities/*.parquet"))
    with ThreadPoolExecutor(16) as ex:
        px = pd.concat(ex.map(load, files), ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    log(f"price cache files {len(files):,}, positive-close rows {len(px):,}, "
        f"securities {px.security.nunique():,}")

    cached = set(px.security.unique())
    mem["no_cache"] = ~mem.security.isin(cached)

    # per security sorted date arrays -> counts in windows via searchsorted
    by = {s: g.values for s, g in px.groupby("security")["date"]}
    priced = np.zeros(len(mem), bool)
    chartable = np.zeros(len(mem), bool)
    d7, d70 = np.timedelta64(7, "D"), np.timedelta64(70, "D")
    for i, (s, d) in enumerate(zip(mem.security.values, mem.date.values)):
        a = by.get(s)
        if a is None:
            continue
        hi = np.searchsorted(a, d, side="right")
        priced[i] = hi > 0 and a[hi - 1] > d - d7
        chartable[i] = hi - np.searchsorted(a, d - d70, side="right") >= 39
    mem["priced"], mem["chartable"] = priced, chartable
    mem.to_parquet(OUT / "member_months.parquet")

    mem["year"] = mem.date.dt.year
    by_year = mem.groupby("year").agg(
        member_months=("security", "size"),
        avg_members=("security", lambda s: len(s) / mem.loc[s.index, "date"].nunique()),
        no_cache=("no_cache", "mean"), priced=("priced", "mean"),
        chartable=("chartable", "mean")).reset_index()
    by_year.to_parquet(OUT / "coverage_by_year.parquet")
    by_year.to_excel(OUT / "coverage_by_year.xlsx", index=False)
    log("\n" + by_year.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    never = (mem.groupby("security")
             .agg(first=("date", "min"), last=("date", "max"), months=("date", "size"),
                  priced_months=("priced", "sum"), no_cache=("no_cache", "first"))
             .reset_index())
    never = never[never.priced_months == 0].sort_values("first")
    never.to_excel(OUT / "missing_names.xlsx", index=False)

    since = mem[mem.date >= "2000-01-01"]
    summary = {
        "run_dir": str(OUT),
        "membership_source": str(args.membership),
        "membership_securities": int(mem.security.nunique()),
        "cache_securities": len(cached),
        "since_2000": {k: float(since[k].mean()) for k in ["no_cache", "priced", "chartable"]},
        "pre_2000": {k: float(mem[mem.date < "2000-01-01"][k].mean())
                     for k in ["no_cache", "priced", "chartable"]},
        "members_never_priced": int(len(never)),
        "members_never_priced_since_2000": int((never["last"] >= "2000-01-01").sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
