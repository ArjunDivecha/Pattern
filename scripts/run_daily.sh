#!/bin/zsh
# =============================================================================
# SCRIPT NAME: run_daily.sh
# =============================================================================
#
# INPUT FILES:
# - /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_ohlcv_database.parquet
#       Bloomberg history (read-only, never rewritten)
# - /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/expanding/20260419_174908_cdef6809/w27_ensemble_{0..4}.pt
# - /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r1000_I20_monthly/
#       reference pixel stats for live scoring
#
# OUTPUT FILES:
# - data/r1000_ohlcv_spliced.{parquet,csv}   Bloomberg + yfinance panel
# - data/r1000_tradeable_universe.csv        staleness manifest
# - runs/live/live_predictions.parquet       latest p_up
# - data/tracker/pattern-trash/{nav_daily.csv,holdings_latest.csv,meta.json}
# - data/tracker/pattern-megacap/{nav_daily.csv,holdings_latest.csv,meta.json}
# - logs/daily/YYYY-MM-DD.log                full run log (Overseer reads this)
#
# DESCRIPTION:
# The Pattern daily job. Publishes the artifacts Tracker reads. Tracker is
# read-only and never runs this -- if a Pattern row shows STALE on the hub, fix
# THIS job, not Tracker.
#
# Scheduled 13:20 PT on trading days (com.arjundivecha.pattern-daily), after the
# 13:00 PT US close so yfinance carries same-day closes, and comfortably before
# Tracker's 13:45 PT consolidation. Takes roughly 8-12 minutes, dominated by
# image rendering in score_live.
#
# Steps:
#   1. splice_daily_update.py  -- extend the Bloomberg panel with yfinance
#   2. score_live.py           -- w27 ensemble -> today's p_up (tradeable only)
#   3. mark_book.py            -- build/mark both sleeves, publish
#   4. verify                  -- REFUSE to report success on stale artifacts
#
# Step 4 exists because steps 1-3 can each exit 0 while publishing yesterday's
# numbers (e.g. yfinance returns nothing and the splice adds no rows). The
# success marker therefore proves the ACTION, not just the exit code: it is
# printed only when both sleeves' meta.json as_of equals the panel's own last
# date. That string is what the Overseer keys on.
#
# USAGE:
#   zsh scripts/run_daily.sh
# =============================================================================
set -uo pipefail

ROOT="/Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern"
PY="/opt/homebrew/bin/python3"
DAY="$(date +%Y-%m-%d)"
LOG_DIR="$ROOT/logs/daily"
LOG="$LOG_DIR/$DAY.log"

# launchd gives almost no environment -- set PATH explicitly.
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "Pattern daily job — $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "=========================================================="

fail=0
step() {
  local name="$1"; shift
  echo ""
  echo "--- $name ---"
  if "$@"; then
    echo "[ok] $name"
  else
    echo "[FAIL] $name (exit $?)"
    fail=1
  fi
}

step "splice prices" \
  "$PY" scripts/splice_daily_update.py --csv

step "score live" \
  "$PY" scripts/score_live.py \
    --data-csv data/r1000_ohlcv_spliced.csv \
    --cache-dir cache/r1000_I20_monthly \
    --tickers-csv data/r1000_tradeable_universe.csv \
    --out-parquet runs/live/live_predictions.parquet \
    --tail-days 160 --cadence monthly --device mps

step "mark book" \
  "$PY" scripts/mark_book.py --start 1999-01-01

# ── 4. verify the published artifacts are actually current ──────────────────
echo ""
echo "--- verify published artifacts ---"
if [ "$fail" -eq 0 ]; then
  "$PY" - <<'PYEOF'
import json, sys
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern")
SLEEVES = ["pattern-trash", "pattern-megacap"]

panel_end = pd.read_parquet(ROOT / "data" / "r1000_ohlcv_spliced.parquet",
                            columns=["Date"])["Date"].max()
panel_end = str(pd.Timestamp(panel_end).date())
print(f"  panel last date: {panel_end}")

bad = []
for sid in SLEEVES:
    mp = ROOT / "data" / "tracker" / sid / "meta.json"
    if not mp.is_file():
        print(f"  {sid:18s} MISSING meta.json")
        bad.append(sid)
        continue
    m = json.loads(mp.read_text())
    ok = m.get("as_of") == panel_end
    print(f"  {sid:18s} as_of={m.get('as_of')}  nav={m.get('nav'):.2f}  "
          f"positions={m.get('n_positions')}  paper_days={m.get('paper_days')}  "
          f"{'OK' if ok else 'STALE vs panel'}")
    if not ok:
        bad.append(sid)

if bad:
    print(f"  VERIFY FAILED: {', '.join(bad)}")
    sys.exit(1)
print(f"PATTERN DAILY OK — published {len(SLEEVES)} sleeves as_of={panel_end}")
PYEOF
  if [ $? -ne 0 ]; then
    fail=1
    echo "[FAIL] verify"
  else
    echo "[ok] verify"
  fi
else
  echo "  skipped — an earlier step failed"
fi

echo ""
echo "=========================================================="
if [ "$fail" -eq 0 ]; then
  echo "Status: ok  $(date '+%H:%M:%S')"
else
  echo "Status: FAILED  $(date '+%H:%M:%S')  -- see $LOG"
fi
echo "=========================================================="
exit "$fail"
