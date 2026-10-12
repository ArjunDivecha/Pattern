"""
Script: cloud_agent.py
Version: 1.0 (2026-10-11)

Description:
    Small helper that cloud_run.py copies to each rented GPU machine (and uses directly
    for its local test host). It starts one train.py unit (one test year x one seed)
    detached from the SSH session, and reports the state of the units it was asked
    about. It never decides anything: scheduling, retries and copying results home are
    cloud_run.py's job. Standard library only, so it runs before any packages exist.

    launch: removes a stale RUNNING.lock left by a dead process, sets aside an earlier
            FAILED.json, then starts
              CUDA_VISIBLE_DEVICES=<gpu> <python> -u research/r3000_headline/train.py
                  --run-dir <run dir> --device <device> --years <year> --seeds <seed> [--smoke]
            in a new session, appending to <run dir>/logs/Y<year>_s<seed>.log. Prints {"pid": n}.
    poll:   reads a JSON list of {"year","seed","pid"} on stdin and prints, per unit, whether
            the process is alive, scores.npy / FAILED.json exist, the progress.json content,
            the epochs finished and last.pt's modification time, plus host load and GPU use.

Input files (on the machine it runs on):
    <run dir>/Y<year>/seed_<seed>/{progress.json, epochs.csv, last.pt, scores.npy, FAILED.json, RUNNING.lock}
Output files:
    <run dir>/logs/Y<year>_s<seed>.log (train.py output, appended per attempt)
    <run dir>/Y<year>/seed_<seed>/FAILED_<utc>.json (an earlier failure, renamed on relaunch)

Usage:
    python cloud_agent.py launch --root R --run-dir D --python P --device cuda --gpu 0 --year 2013 --seed 0 [--smoke]
    echo '[{"year":2013,"seed":0,"pid":123}]' | python cloud_agent.py poll --run-dir D
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, PermissionError, ValueError):
        return False
    cmd = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
    return 'train.py' in cmd


def seed_dir(run_dir, year, seed):
    return Path(run_dir) / f'Y{year}' / f'seed_{seed}'


def launch(a):
    sd = seed_dir(a.run_dir, a.year, a.seed)
    sd.mkdir(parents=True, exist_ok=True)
    lock = sd / 'RUNNING.lock'
    if lock.exists():
        owner = lock.read_text().strip()
        if alive(owner.rsplit(':', 1)[-1]):
            print(json.dumps({'error': f'unit already running: {owner}'}))
            return 1
        lock.unlink()
    if (sd / 'FAILED.json').exists():
        (sd / 'FAILED.json').replace(sd / f"FAILED_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    logs = Path(a.run_dir) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = open(logs / f'Y{a.year}_s{a.seed}.log', 'a')
    log.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} gpu {a.gpu} device {a.device} ===\n")
    log.flush()
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    if a.device == 'cuda':
        env['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    cmd = [a.python, '-u', 'research/r3000_headline/train.py', '--run-dir', a.run_dir, '--device', a.device,
           '--years', str(a.year), '--seeds', str(a.seed)] + (['--smoke'] if a.smoke else [])
    p = subprocess.Popen(cmd, cwd=a.root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
    print(json.dumps({'pid': p.pid}))
    return 0


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def poll(a):
    units = json.loads(sys.stdin.read() or '[]')
    out = {'units': {}, 'host': {'loadavg': os.getloadavg()[0], 'utc': time.time()}}
    for u in units:
        sd = seed_dir(a.run_dir, u['year'], u['seed'])
        last = sd / 'last.pt'
        epochs = sd / 'epochs.csv'
        rec = {'alive': alive(u.get('pid')), 'done': (sd / 'scores.npy').exists(),
               'failed': (sd / 'FAILED.json').exists(), 'progress': read_json(sd / 'progress.json'),
               'epochs_done': max(0, len(epochs.read_text().splitlines()) - 1) if epochs.exists() else 0,
               'last_mtime': last.stat().st_mtime if last.exists() else None}
        if rec['failed']:
            rec['error'] = (read_json(sd / 'FAILED.json') or {}).get('error', '')[-1500:]
        out['units'][f"Y{u['year']}_s{u['seed']}"] = rec
    try:
        g = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20)
        out['host']['gpus'] = [line.split(', ') for line in g.stdout.strip().splitlines()]
    except (OSError, subprocess.TimeoutExpired):
        out['host']['gpus'] = None
    print(json.dumps(out))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    lp = sub.add_parser('launch')
    for k in ['--root', '--run-dir', '--python', '--device']:
        lp.add_argument(k, required=True)
    lp.add_argument('--gpu', type=int, default=0)
    lp.add_argument('--year', type=int, required=True)
    lp.add_argument('--seed', type=int, required=True)
    lp.add_argument('--smoke', action='store_true')
    pp = sub.add_parser('poll')
    pp.add_argument('--run-dir', required=True)
    a = ap.parse_args()
    return launch(a) if a.cmd == 'launch' else poll(a)


if __name__ == '__main__':
    sys.exit(main())
