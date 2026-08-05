#!/usr/bin/env bash
# =============================================================================
# SCRIPT NAME: run_daily.sh
# =============================================================================
#
# INPUT FILES:
# - data/r1000_ohlcv_database.parquet     Bloomberg history (read-only)
# - runs/expanding/20260419_174908_cdef6809/w27_ensemble_{0..4}.pt
# - cache/r1000_I20_monthly/              reference pixel stats for live scoring
#
# OUTPUT FILES:
# - data/r1000_ohlcv_spliced.{parquet,csv}    Bloomberg + yfinance panel
# - data/r1000_tradeable_universe.csv         staleness manifest
# - runs/live/live_predictions.parquet        latest p_up
# - data/tracker/pattern-trash/{nav_daily.csv,holdings_latest.csv,meta.json}
# - data/tracker/pattern-megacap/{nav_daily.csv,holdings_latest.csv,meta.json}
# - logs/daily/YYYY-MM-DD.log                 full run log
#
# DESCRIPTION:
# The Pattern daily job. Publishes the artifacts Tracker reads. Tracker is
# read-only and never runs this -- if a Pattern row shows STALE on the hub,
# fix THIS job, not Tracker.
#
# Must complete before Tracker's 13:45 PT consolidation.
# Takes roughly 8-12 minutes, dominated by image rendering in score_live.
#
# USAGE:
#   bash scripts/run_daily.sh
# =============================================================================
set -uo pipefail

ROOT="/Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern"
PY="$(command -v python3)"
RUN="$ROOT/runs/expanding/20260419_174908_cdef6809"
DAY="$(date +%Y-%m-%d)"
LOG_DIR="$ROOT/logs/daily"
LOG="$LOG_DIR/$DAY.log"
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

# 1. Extend the Bloomberg panel with yfinance. Bloomberg is never rewritten.
step "splice prices" \
  "$PY" scripts/splice_daily_update.py --csv

# 2. Refresh the CNN signal. Only the month-end signal drives a rebalance --
#    mark_book ignores mid-month prediction dates -- but re-scoring daily keeps
#    the latest month-end current and costs a few minutes.
step "score live" \
  "$PY" scripts/score_live.py \
    --data-csv data/r1000_ohlcv_spliced.csv \
    --cache-dir cache/r1000_I20_monthly \
    --tickers-csv data/r1000_tradeable_universe.csv \
    --out-parquet runs/live/live_predictions.parquet \
    --tail-days 160 --cadence monthly --device mps

# 3. Build + mark both sleeves and publish what Tracker reads.
step "mark book" \
  "$PY" scripts/mark_book.py --start 1999-01-01

echo ""
echo "=========================================================="
if [ "$fail" -eq 0 ]; then
  echo "Status: ok  $(date '+%H:%M:%S')"
  for s in pattern-trash pattern-megacap; do
    "$PY" - "$s" <<'PYEOF'
import json, sys
from pathlib import Path
sid = sys.argv[1]
p = Path("/Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/tracker") / sid / "meta.json"
if p.is_file():
    m = json.loads(p.read_text())
    print(f"  {sid:18s} as_of={m['as_of']}  nav={m['nav']:.2f}  "
          f"positions={m['n_positions']}  paper_days={m['paper_days']}")
else:
    print(f"  {sid:18s} MISSING meta.json")
PYEOF
  done
else
  echo "Status: FAILED  $(date '+%H:%M:%S')  -- see $LOG"
fi
echo "=========================================================="
exit "$fail"
