"""
=============================================================================
SCRIPT NAME: run_multi_gpu.py
=============================================================================

INPUT FILES:
- configs/prod_expanding.yaml or configs/prod_rolling.yaml
- /data/Pattern/cache/prod_I20/images.npy + index.parquet (built on first use)

OUTPUT FILES:
- {output_dir}/<timestamp>_<hash>/window_NN_predictions.parquet (one per window)
- {output_dir}/<timestamp>_<hash>/window_NN_features.npz
- {output_dir}/<timestamp>_<hash>/window_stats.csv
- {output_dir}/<timestamp>_<hash>/predictions.parquet (merged at the end)

DESCRIPTION:
8-GPU parallelization driver.  Enumerates every (window_idx) for a given
config, assigns them round-robin to GPUs 0..n_gpus-1, then spawns one
subprocess per GPU with CUDA_VISIBLE_DEVICES pinned and --window-indices
set to that GPU's shard.  Each subprocess sees only its one device as
`cuda:0` so the existing training code needs no GPU-index logic.

After all shards finish, the driver concatenates the per-window prediction
parquets into the final predictions.parquet and prints a quick AUC check.

Wall-clock savings scale with the number of windows and GPUs.  At 27 windows
and 8 GPUs the schedule is 4 waves × ~5 min ≈ 20 min per pathway instead of
~135 min sequential.

USAGE:
  python scripts/run_multi_gpu.py --config configs/prod_expanding.yaml
  python scripts/run_multi_gpu.py --config configs/prod_rolling.yaml --n-gpus 8

NOTES:
- Run this from the repo root so the relative paths in the config resolve.
- The image cache is built once on first use by the shard that sees an
  empty cache; subsequent shards will reuse it (no races because the cache
  is a pure read after build, and concurrent shards starting against an
  already-built cache is the intended steady state).  If you are running
  the very first time, run one shard alone (--n-gpus 1) until the cache is
  built, then re-run with all GPUs.  Simpler: pre-build the cache via the
  helper below (--prebuild-cache).
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _count_windows(cfg_path: Path) -> int:
    """Compute the number of windows for this config without running training."""
    sys.path.insert(0, str(REPO_ROOT))
    from pattern.config import Config
    from pattern.data.loader import compute_labels, load_data
    from pattern.data.splits import get_splits

    cfg = Config.from_yaml(cfg_path)
    df = load_data(cfg.data.csv_path, cfg.data.date_format, cfg.data.min_history_days)
    labelled = compute_labels(df, cfg.label.horizon)
    purge = cfg.image.window + cfg.label.horizon - 1
    return len(get_splits(labelled, cfg.split, purge_days=purge))


def _make_run_dir(cfg_path: Path) -> Path:
    """Mirror cli._make_run_dir so the driver picks the same dir naming scheme."""
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg_hash = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = Path(cfg.get("output_dir", "./runs"))
    run_dir  = out_dir / f"{stamp}_{cfg_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _assign_windows(n_windows: int, n_gpus: int) -> list[list[int]]:
    """Round-robin assignment: GPU g gets windows g, g+n_gpus, g+2n_gpus, ..."""
    return [[w for w in range(n_windows) if w % n_gpus == g] for g in range(n_gpus)]


def _prebuild_cache(cfg_path: Path) -> None:
    """Run cli on GPU 0 with --window-indices -1 (no-op) just to trigger cache build."""
    # Easier: call build_cache directly.
    sys.path.insert(0, str(REPO_ROOT))
    from pattern.config import Config
    from pattern.data.loader import compute_labels, load_data
    from pattern.imaging.cache import build_cache

    cfg = Config.from_yaml(cfg_path)
    df = load_data(cfg.data.csv_path, cfg.data.date_format, cfg.data.min_history_days)
    labelled = compute_labels(df, cfg.label.horizon)
    build_cache(labelled, cfg.image, cfg.label, cfg.image.cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to pathway config YAML")
    parser.add_argument("--n-gpus", type=int, default=8,
                        help="Number of GPUs to fan out over (default 8)")
    parser.add_argument("--run-dir", default=None,
                        help="Resume into an existing run dir instead of creating one")
    parser.add_argument("--prebuild-cache", action="store_true",
                        help="Build the image cache serially before launching shards, "
                             "then exit (avoids first-shard cache-build race).")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    if args.prebuild_cache:
        print(f"[driver] pre-building image cache for {cfg_path.name}")
        _prebuild_cache(cfg_path)
        print("[driver] cache build done")
        return

    run_dir = Path(args.run_dir).resolve() if args.run_dir else _make_run_dir(cfg_path)
    print(f"[driver] run dir: {run_dir}")

    n_windows = _count_windows(cfg_path)
    print(f"[driver] detected {n_windows} windows for {cfg_path.name}")

    shards = _assign_windows(n_windows, args.n_gpus)
    for g, wins in enumerate(shards):
        print(f"[driver]  GPU {g}: windows {wins}")

    # Spawn one subprocess per GPU.
    procs: list[tuple[int, subprocess.Popen, Path]] = []
    for g, wins in enumerate(shards):
        if not wins:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        env["PYTHONUNBUFFERED"] = "1"
        log_path = run_dir / f"shard_gpu{g}.log"
        spec = ",".join(str(w) for w in wins)
        cmd = [
            sys.executable, "-m", "pattern.cli", "train",
            "--config", str(cfg_path),
            "--window-indices", spec,
            "--run-dir", str(run_dir),
        ]
        print(f"[driver] spawn GPU {g}: {' '.join(cmd)}  →  {log_path.name}")
        f_out = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=f_out, stderr=subprocess.STDOUT, env=env,
                                cwd=str(REPO_ROOT))
        procs.append((g, proc, log_path))

    # Wait for all shards.
    t0 = time.time()
    failed: list[int] = []
    for g, proc, log_path in procs:
        rc = proc.wait()
        print(f"[driver] GPU {g} finished rc={rc}  log={log_path}")
        if rc != 0:
            failed.append(g)

    print(f"[driver] all shards done in {time.time() - t0:.1f}s  failed={failed}")
    if failed:
        raise SystemExit(f"Shards failed on GPUs: {failed}")

    # Final merge — concatenate all per-window predictions parquets.
    files = sorted(run_dir.glob("window_*_predictions.parquet"))
    if not files:
        raise SystemExit("No per-window predictions written; something went wrong.")
    pred_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out_path = run_dir / "predictions.parquet"
    pred_df.to_parquet(out_path, index=False)
    print(f"[driver] merged {len(files)} window parquets → {out_path}  "
          f"({len(pred_df):,} rows)")

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(pred_df["label"], pred_df["p_up_mean"])
        print(f"[driver] overall Test AUC = {auc:.4f}")
    except Exception as e:
        print(f"[driver] AUC computation skipped: {e}")


if __name__ == "__main__":
    main()
