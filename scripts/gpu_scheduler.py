"""
=============================================================================
SCRIPT NAME: gpu_scheduler.py
=============================================================================

INPUT:
- --config    : path to pathway YAML (expanding/rolling)
- --run-dir   : run directory (pre-created)
- --n-windows : total number of windows to schedule (0..N-1)
- --n-gpus    : total GPU count on the box (default 8)

DESCRIPTION:
Work-stealing scheduler that shares GPUs with another running workload.
Polls nvidia-smi every 30s; whenever a GPU has no compute apps attached,
pops the next pending window index from the queue and spawns

    CUDA_VISIBLE_DEVICES=g python -m pattern.cli train \
        --config <cfg> --window-indices <w> --run-dir <run>

on that GPU.  Exits when the queue is empty and all launched shards have
finished.  Per-window logs go to shard_gpu{g}_w{w:02d}.log inside the run
directory; scheduler stdout is the single event stream for monitoring.

USAGE:
  python scripts/gpu_scheduler.py --config configs/prod_rolling.yaml \
      --run-dir /data/Pattern/runs/rolling/<ts> --n-windows 28
=============================================================================
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _gpu_pids(gpu_id: int) -> list[str]:
    """Return PIDs currently running compute on GPU gpu_id via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-compute-apps=pid", "--format=csv,noheader"],
            timeout=10,
        ).decode().strip()
    except Exception as e:
        print(f"[sched] nvidia-smi error on gpu{gpu_id}: {e}", flush=True)
        return ["?"]
    return [p.strip() for p in out.splitlines() if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--n-windows", required=True, type=int)
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--poll-seconds", type=int, default=30)
    args = ap.parse_args()

    pending: list[int] = list(range(args.n_windows))
    active: dict[int, tuple[int, subprocess.Popen]] = {}
    done: list[int] = []
    failed: list[int] = []

    print(f"[sched] config={args.config.name}  run_dir={args.run_dir}")
    print(f"[sched] queue=[{args.n_windows} windows]  n_gpus={args.n_gpus}")
    sys.stdout.flush()

    while pending or active:
        # Reap finished shards.
        for g in list(active):
            w, proc = active[g]
            rc = proc.poll()
            if rc is not None:
                (done if rc == 0 else failed).append(w)
                tag = "ok" if rc == 0 else f"FAIL rc={rc}"
                print(f"[sched] GPU {g} window {w} finished {tag}  "
                      f"done={len(done)}/{args.n_windows}  failed={len(failed)}")
                sys.stdout.flush()
                del active[g]

        # Schedule pending windows on idle GPUs.
        for g in range(args.n_gpus):
            if g in active or not pending:
                continue
            pids = _gpu_pids(g)
            if pids:
                continue  # GPU busy with another workload
            w = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(g)
            env["PYTHONUNBUFFERED"] = "1"
            log_path = args.run_dir / f"shard_gpu{g}_w{w:02d}.log"
            cmd = [
                sys.executable, "-m", "pattern.cli", "train",
                "--config", str(args.config),
                "--window-indices", str(w),
                "--run-dir", str(args.run_dir),
            ]
            print(f"[sched] GPU {g} starting window {w}  → {log_path.name}  "
                  f"(pending={len(pending)}  active={len(active)+1})")
            sys.stdout.flush()
            f_out = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=f_out, stderr=subprocess.STDOUT,
                                    env=env, cwd=str(REPO_ROOT))
            active[g] = (w, proc)

        if pending or active:
            time.sleep(args.poll_seconds)

    print(f"[sched] all done  done={sorted(done)}  failed={sorted(failed)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
