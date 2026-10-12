#!/usr/bin/env bash
# Script: cloud_timing.sh
# Version: 1.0 (2026-10-11)
#
# Description:
#   Timing test for the R3000 headline trainer on a rented CUDA machine. Runs the real
#   train.py on one fold (default test year 2013, a mid-sized fold) with K concurrent
#   processes (one seed each, sharing one run directory) for a fixed number of seconds,
#   samples each seed's progress.json at the start and end of a steady-state window, and
#   records aggregate training images/second, GPU utilisation and CPU load. Then kills the
#   processes. Nothing it produces is a research result.
#
# Input files (on the remote machine, mirrored absolute paths):
#   /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json and the v2 snapshot files
#   /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/{index.parquet,state.json}
#   <run dir>/white_pixels.npy (copied from the local run so start-up skips recomputing it)
# Output files:
#   <run dir>/timing_K<k>.json, <run dir>/train_K<k>_seed<s>.log, <run dir>/gpu_K<k>.csv
#
# Usage: cloud_timing.sh RUN_DIR K WARMUP_SECONDS MEASURE_SECONDS [YEAR]
set -u
RUN="$1"; K="$2"; WARM="$3"; MEAS="$4"; YEAR="${5:-2013}"
ROOT="/Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern"
PY=/root/venv/bin/python
cd "$ROOT"
pids=()
for s in $(seq 0 $((K-1))); do
  rm -rf "$RUN/Y$YEAR/seed_$s"
  $PY -u research/r3000_headline/train.py --run-dir "$RUN" --device cuda --years $YEAR --seeds $s > "$RUN/train_K${K}_seed$s.log" 2>&1 &
  pids+=($!)
done
# wait until every seed has started training (progress.json in train phase)
for i in $(seq 1 600); do
  n=0; for s in $(seq 0 $((K-1))); do grep -q '"phase": "train"' "$RUN/Y$YEAR/seed_$s/progress.json" 2>/dev/null && n=$((n+1)); done
  [ $n -eq $K ] && break; sleep 2
done
sleep "$WARM"
snap() { $PY - "$RUN" "$YEAR" "$K" <<'EOF'
import json,sys
run,year,k=sys.argv[1],sys.argv[2],int(sys.argv[3])
out={}
for s in range(k):
    p=json.load(open(f'{run}/Y{year}/seed_{s}/progress.json'))
    out[s]={'epoch':p.get('epoch'),'phase':p.get('phase'),'processed':p.get('processed'),'elapsed':p.get('elapsed_seconds')}
print(json.dumps(out))
EOF
}
A=$(snap); T0=$(date +%s)
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 5 > "$RUN/gpu_K$K.csv" &
GP=$!
sleep "$MEAS"
B=$(snap); T1=$(date +%s)
kill $GP
LOAD=$(cut -d' ' -f1 /proc/loadavg)
$PY - "$RUN" "$K" "$T1" "$T0" "$LOAD" "$A" "$B" <<'EOF'
import json,sys
run,k,t1,t0,load=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),float(sys.argv[5])
a,b=json.loads(sys.argv[6]),json.loads(sys.argv[7])
per={}
for s in a:
    ok=a[s]['epoch']==b[s]['epoch'] and a[s]['phase']==b[s]['phase']=='train'
    per[s]={'images_per_s':(b[s]['processed']-a[s]['processed'])/(b[s]['elapsed']-a[s]['elapsed']) if ok else None,'same_epoch_train_phase':ok}
gpu=[float(l.split(',')[1].strip().rstrip(' %')) for l in open(f'{run}/gpu_K{k}.csv') if l.strip()]
res={'processes':k,'window_s':t1-t0,'per_seed':per,'aggregate_images_per_s':sum(v['images_per_s'] or 0 for v in per.values()),
     'all_valid':all(v['same_epoch_train_phase'] for v in per.values()),'gpu_util_mean':sum(gpu)/len(gpu),'loadavg_1m':load}
json.dump(res,open(f'{run}/timing_K{k}.json','w'),indent=2); print(json.dumps(res))
EOF
kill "${pids[@]}" 2>/dev/null; sleep 3; kill -9 "${pids[@]}" 2>/dev/null
rm -f "$RUN"/Y$YEAR/seed_*/RUNNING.lock
