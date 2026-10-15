"""
Script: cloud_run.py
Version: 1.2 (2026-10-15)

Description:
    Runs the R3000 headline expanding-window training (train.py, unchanged) on rented
    Verda GPU machines, driven from this Mac. The work is 135 units: 27 test years
    (2000-2026) x 5 seeds. One unit runs per GPU at a time (the A100 timing test showed
    one process saturates a GPU). Longest units go first.

    What it does, in order:
      prepare   make the home run directory, copy the precomputed white-pixel counts,
                record the SHA-256 of every file the machines need. Spends nothing.
      plan      unit list, estimated A100-hours, cost and wall time, and live Verda
                prices and spot availability. Read-only; spends nothing.
      start     THE START BUTTON. Rents up to --gpus GPUs (spot by default), installs
                clang + a pinned Python stack, uploads code and data (from the Mac to the
                first machine, then machine-to-machine), checks every file's hash, and
                dispatches units. Every minute it polls each machine; finished units are
                copied home and checked; each running unit's last.pt is copied home every
                15 minutes so a unit lost to spot preemption resumes on another machine
                from its last finished epoch. Lost machines are detected (unreachable, or
                Verda reports them gone) and their units requeued. A unit that raises an
                error twice is marked failed, never silently dropped. Stops renting at
                --budget-usd. When every unit is home it writes the ensembles (train.py on
                CPU, which only assembles since every seed is done) and runs report.py.
      resume    restart the dispatcher with the saved settings (e.g. after the Mac
                restarts); running units on the machines are untouched.
      status    print the heartbeat.
      teardown  delete this run's machines and their disks, only those it created, then
                sweep and verify that no disk of this run is left (see v1.2 below).
      assemble  write ensembles for complete years and run report.py.
      localtest end-to-end test on this Mac's CPU with two fake local machines, smoke-
                sized units, a simulated machine loss, and a resume-from-checkpoint check.

    v1.2: deleting a machine does not reliably delete its disk. In the 2026-10-12 run, all 14
    machines deleted after a failed setup or a spot reclaim kept their 100 GB OS disk (billed
    $20/month each) although Verda accepted every delete call; only machines deleted while
    running lost theirs. The dispatcher now deletes each recorded disk itself once its
    machine is gone (every 5 minutes while running, and at the end), then re-lists Verda's
    volumes until none of this run's disks remain, and logs DISK CLEANUP FAILED otherwise.
    It only ever touches disks whose id it recorded for a machine it created.

    Machines are deleted automatically only if --auto-delete was given (Arjun's
    authorisation; Verda reuses IPs, so deletion needs his explicit say-so). Without it,
    idle machines keep billing until teardown is run.

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/.env.txt (VERDA_CLIENT_ID, VERDA_CLIENT_SECRET)
    ~/.ssh/id_ed25519 (Verda SSH key id cd39268d-3a85-4316-9970-a29bf990ed48)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json and the snapshot it names
        (manifest.json, prices.parquet, quality.parquet, sessions.parquet)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/{index.parquet, state.json}
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/pattern/ (package code), research/r3000_headline/{prepare,train,cloud_agent,report}.py
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_20261010T184018Z/white_pixels.npy
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/cloud_timing_A100_20261011/fold_workload.csv
Output files (home run dir, default research/r3000_headline/runs/headline_I20R20_v2_cloud_<UTC>/):
    Y<year>/seed_<s>/{best.pt, last.pt, epochs.csv, normalization.json, progress.json, scores.npy, cloud_provenance.json}
    Y<year>/{sample_counts.json, ensemble_scores.parquet, COMPLETE.json}, logs/Y<year>_s<s>.log,
    config.json, environment.json, folds.csv, white_pixels.npy, assemble.log, report_<UTC>/ (report.py), report.log
    cloud/{settings.json, hosts.json, units.json, status.json, dispatch.log, upload_manifest.json,
           fold_workload.csv, relay_key(.pub), partial/<unit>/, failures/<unit>_<n>/, hosts/<name>/}
    ~/.ssh/pattern_cloud/<run name>/<machine>_known_hosts (one per machine; Verda reuses IPs)

Usage (repo venv):
    .venv/bin/python research/r3000_headline/cloud_run.py prepare
    .venv/bin/python research/r3000_headline/cloud_run.py plan --run-dir DIR --gpus 4
    nohup caffeinate -is .venv/bin/python research/r3000_headline/cloud_run.py start --run-dir DIR --gpus 4 \
        --budget-usd 200 --confirm-spend [--auto-delete] [--types 1A100.22V 2A100.44V] > DIR/cloud/start.out 2>&1 &
    .venv/bin/python research/r3000_headline/cloud_run.py status --run-dir DIR
    .venv/bin/python research/r3000_headline/cloud_run.py teardown --run-dir DIR --confirm-delete
    .venv/bin/python research/r3000_headline/cloud_run.py localtest [--ssh-shim]

Dependencies: numpy (verification of copied scores); everything else is the standard library,
ssh and tar. Remote machines: Verda image 24.04.cuda13.2, packages pinned in PIP_PINS.
"""
from pathlib import Path
import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
VENV_PY = ROOT / '.venv/bin/python'
ENV_FILE = Path('/Users/arjundivecha/Dropbox/AAA Backup/.env.txt')
SSH_KEY = Path.home() / '.ssh/id_ed25519'
SSH_KEY_ID = 'cd39268d-3a85-4316-9970-a29bf990ed48'
VERDA = 'https://api.verda.com/v1'
IMAGE = '24.04.cuda13.2'
REMOTE_PY = '/root/venv/bin/python'
PIP_PINS = 'torch==2.14.1 numpy==2.4.4 pandas==3.0.2 pyarrow==23.0.1 pydantic==2.13.3 pyyaml==6.0.3'
WHITE_SRC = RUNS / 'headline_I20R20_v2_20261010T184018Z/white_pixels.npy'
WORKLOAD_SRC = RUNS / 'cloud_timing_A100_20261011/fold_workload.csv'
INDEX_DIR = ROOT / 'cache/r3000_I20_M20_v2_index'
DEFAULT_TYPES = ['1A100.22V', '2A100.44V']     # A100 80GB: the hardware the timing test measured
YEARS, SEEDS = list(range(2000, 2027)), [0, 1, 2, 3, 4]
# Unit-time model from the A100 timing test (2026-10-11): training 17,000 images/s,
# validation and scoring 60,700/s, 6.125 epochs per seed observed, ~45 s start-up.
TRAIN_RATE, EVAL_RATE, EPOCHS, STARTUP_S = 17000, 60700, 6.125, 45
STALE_S = 40 * 60          # no progress for this long -> the unit is killed and requeued
LIVE = ('creating', 'booting', 'setup', 'ready')
SETUP_TRIES = 3   # a machine gets this many setup attempts (ssh drops, first-boot reboots) before it is given up
# Verda's first-boot cloud-init runs apt for a few minutes; installing alongside it hits the apt lock.
WAIT_FOR_BOOT = ('cloud-init status --wait >/dev/null 2>&1; '
                 'for i in $(seq 1 360); do '
                 'pgrep -x apt-get >/dev/null || pgrep -x apt >/dev/null || pgrep -x dpkg >/dev/null || '
                 'pgrep -f "/usr/bin/unattended-upgrade( |$)" >/dev/null || exit 0; sleep 5; done; '
                 'echo "apt still busy after 30 min" >&2; exit 1')


def now():
    return time.time()


def utc(t=None):
    return datetime.fromtimestamp(t or now(), timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def unit_name(year, seed):
    return f'Y{year}_s{seed}'


# ----------------------------------------------------------------------------- Verda API
class Verda:
    def __init__(self):
        creds = {}
        for line in ENV_FILE.read_text().splitlines():
            k, _, v = line.partition('=')
            k = k.strip()
            if k in ('VERDA_CLIENT_ID', 'VERDA_CLIENT_SECRET') and k not in creds:
                creds[k] = v.strip().strip('"\'')
        if len(creds) != 2:
            raise RuntimeError(f'VERDA_CLIENT_ID / VERDA_CLIENT_SECRET not found in {ENV_FILE}')
        self.creds, self.tok, self.exp = creds, None, 0

    def _call(self, method, path, body=None, auth=True):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(VERDA + path, data=data, method=method)
        req.add_header('Content-Type', 'application/json')
        if auth:
            req.add_header('Authorization', f'Bearer {self.token()}')
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'Verda {method} {path} -> HTTP {e.code}: {e.read().decode()[:500]}') from None
        try:
            return json.loads(raw) if raw.strip() else None
        except ValueError:
            return raw.strip()

    def token(self):
        if not self.tok or now() > self.exp - 120:
            r = self._call('POST', '/oauth2/token', {'grant_type': 'client_credentials',
                                                     'client_id': self.creds['VERDA_CLIENT_ID'],
                                                     'client_secret': self.creds['VERDA_CLIENT_SECRET']}, auth=False)
            self.tok, self.exp = r['access_token'], now() + float(r.get('expires_in', 3600))
        return self.tok

    def types(self):
        return {t['instance_type']: t for t in self._call('GET', '/instance-types')}

    def availability(self, spot):
        return {x['location_code']: x['availabilities'] for x in self._call('GET', f'/instance-availability?is_spot={str(spot).lower()}')}

    def instances(self):
        return self._call('GET', '/instances')

    def instance(self, iid):
        return self._call('GET', f'/instances/{iid}')

    def create(self, itype, location, hostname, spot, description):
        r = self._call('POST', '/instances', {'instance_type': itype, 'image': IMAGE, 'ssh_key_ids': [SSH_KEY_ID],
                                              'location_code': location, 'hostname': hostname, 'description': description,
                                              'is_spot': spot, 'os_volume': {'name': f'{hostname}-os', 'size': 100}})
        iid = r if isinstance(r, str) else (r or {}).get('id')
        if not iid:
            raise RuntimeError(f'Verda create returned no instance id: {r!r}')
        return iid.strip('"')

    def delete(self, iid, volume_ids):
        return self._call('PUT', '/instances', {'id': iid, 'action': 'delete', 'volume_ids': volume_ids,
                                                'delete_permanently': True})

    def volumes(self):
        return self._call('GET', '/volumes') or []

    def delete_volumes(self, ids):
        return self._call('PUT', '/volumes', {'id': list(ids), 'action': 'delete', 'is_permanent': True})


# ----------------------------------------------------------------------------- run setup
def code_and_data_files(home):
    """Paths relative to ROOT that every machine needs."""
    ref = json.loads((ROOT / 'data/frozen/CURRENT.json').read_text())
    snap = Path(ref['path']).relative_to(ROOT)
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'pattern').rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and p.name != '.DS_Store')
    files += [f'research/r3000_headline/{n}.py' for n in ('prepare', 'train', 'cloud_agent')]
    files += ['data/frozen/CURRENT.json'] + [str(snap / n) for n in ('manifest.json', 'prices.parquet', 'quality.parquet', 'sessions.parquet')]
    files += [str((INDEX_DIR / n).relative_to(ROOT)) for n in ('index.parquet', 'state.json')]
    files += [str((home / 'white_pixels.npy').relative_to(ROOT))]
    return files


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def unit_hours(workload):
    est = {}
    for _, r in workload.iterrows():
        est[int(r.year)] = (r.train_bal * EPOCHS / TRAIN_RATE + r.val * EPOCHS / EVAL_RATE + r.test / EVAL_RATE + STARTUP_S) / 3600
    return est


def prepare(home=None):
    import pandas as pd
    home = Path(home) if home else RUNS / f"headline_I20R20_v2_cloud_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    cloud = home / 'cloud'
    cloud.mkdir(parents=True, exist_ok=True)
    index_rows = pd.read_parquet(INDEX_DIR / 'index.parquet', columns=['row_id']).shape[0]
    white = np.load(WHITE_SRC, mmap_mode='r')
    if len(white) != index_rows:
        raise ValueError(f'white_pixels.npy has {len(white)} rows, index has {index_rows}')
    if not (home / 'white_pixels.npy').exists():
        shutil.copy2(WHITE_SRC, home / 'white_pixels.npy')
    shutil.copy2(WORKLOAD_SRC, cloud / 'fold_workload.csv')
    files = code_and_data_files(home)
    manifest = {f: sha256(ROOT / f) for f in files}
    atomic_json(cloud / 'upload_manifest.json', {'root': str(ROOT), 'files': manifest,
                                                 'bytes': sum((ROOT / f).stat().st_size for f in files), 'utc': utc()})
    if not (cloud / 'relay_key').exists():
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', f'pattern-relay-{home.name}', '-f',
                        str(cloud / 'relay_key')], check=True)
    print(f'prepared {home}\n  {len(files)} files, {sum((ROOT / f).stat().st_size for f in files) / 1e9:.2f} GB to upload per machine')
    return home


# ----------------------------------------------------------------------------- dispatcher
class Dispatcher:
    def __init__(self, home, settings):
        self.home = Path(home).resolve()
        self.cloud = self.home / 'cloud'
        self.s = settings
        self.lock = threading.RLock()
        self.hosts = read_json(self.cloud / 'hosts.json', [])
        self.units = read_json(self.cloud / 'units.json')
        self.verda = Verda() if settings['transport'] == 'ssh' else None
        self.last_sweep = 0.0
        self.manifest = read_json(self.cloud / 'upload_manifest.json')
        self.files = list(self.manifest['files'])
        self.last_fill = 0
        self.stop = False
        self.chaos_done = False
        self.mac_lock = threading.Lock()   # one upload from the Mac at a time; later machines relay from the first
        self.mac_lock_holder = None
        if self.units is None:
            import pandas as pd
            est = unit_hours(pd.read_csv(self.cloud / 'fold_workload.csv'))
            self.units = {unit_name(y, s): {'year': y, 'seed': s, 'est_hours': round(est[y], 4), 'state': 'queued',
                                            'attempts': 0, 'errors': 0, 'deaths': 0, 'preemptions': 0, 'history': []}
                          for y in settings['years'] for s in settings['seeds']}
            for u in self.units.values():   # already copied home (a resumed run)
                if (self.seed_dir(u) / 'scores.npy').exists():
                    u['state'] = 'done'
        self.save()

    # --- bookkeeping
    def log(self, msg):
        line = f'[{utc()}] {msg}'
        print(line, flush=True)
        with open(self.cloud / 'dispatch.log', 'a') as f:
            f.write(line + '\n')

    def save(self):
        with self.lock:
            atomic_json(self.cloud / 'hosts.json', self.hosts)
            atomic_json(self.cloud / 'units.json', self.units)

    def seed_dir(self, u):
        return self.home / f"Y{u['year']}" / f"seed_{u['seed']}"

    def host(self, name):
        return next(h for h in self.hosts if h['name'] == name)

    def spent(self):
        return sum(h.get('price_per_hour', 0) * ((h.get('ended') or now()) - h['created']) / 3600 for h in self.hosts)

    def burn(self):
        return sum(h.get('price_per_hour', 0) for h in self.hosts if h['state'] in LIVE)

    # --- transport
    def ssh_cmd(self, h):
        # ssh splits UserKnownHostsFile on whitespace, so this path must not contain spaces;
        # one file per machine because Verda reuses IPs.
        kh = Path.home() / '.ssh' / 'pattern_cloud' / self.home.name / f"{h['name']}_known_hosts"
        kh.parent.mkdir(parents=True, exist_ok=True)
        return ['ssh', '-i', str(SSH_KEY), '-o', f'UserKnownHostsFile={kh}', '-o', 'StrictHostKeyChecking=accept-new',
                '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4',
                '-o', 'BatchMode=yes', '-o', 'LogLevel=ERROR', f"root@{h['ip']}"]

    def remote_shell(self, h, cmd):
        """A bash command line that runs cmd on host h."""
        if 'fake_dir' in h and (Path(h['fake_dir']) / 'DOWN').exists():
            return '(echo "test host is down" >&2; exit 255)'
        if h['transport'] == 'local':
            return f'bash -c {shlex.quote(cmd)}'   # one command, like ssh, so pipes bind correctly
        return shlex.join(self.ssh_cmd(h) + [cmd])

    def run(self, h, cmd, input=None, timeout=180):
        try:
            return subprocess.run(['bash', '-c', 'set -o pipefail; ' + self.remote_shell(h, cmd)], input=input,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, 124, '', f'timeout after {timeout}s')

    def pull(self, h, items, dest, timeout=900):
        """Copy items (relative to the host's run dir) into dest."""
        dest.mkdir(parents=True, exist_ok=True)
        src = f"cd {shlex.quote(h['run_dir'])} && tar cf - {' '.join(shlex.quote(i) for i in items)}"
        pipe = f'{self.remote_shell(h, src)} | tar xf - -C {shlex.quote(str(dest))}'
        try:
            return subprocess.run(['bash', '-c', 'set -o pipefail; ' + pipe], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(pipe, 124, '', 'timeout')

    def push(self, h, src_dir, remote_dir, timeout=900):
        dst = f'mkdir -p {shlex.quote(remote_dir)} && tar xf - -C {shlex.quote(remote_dir)}'
        pipe = f'COPYFILE_DISABLE=1 tar cf - -C {shlex.quote(str(src_dir))} . | {self.remote_shell(h, dst)}'
        try:
            return subprocess.run(['bash', '-c', 'set -o pipefail; ' + pipe], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(pipe, 124, '', 'timeout')

    def agent(self, h):
        return f"cd {shlex.quote(h['root'])} && {shlex.quote(h['python'])} research/r3000_headline/cloud_agent.py"

    # --- provisioning
    def create_hosts(self):
        """Rent GPUs up to the target, within budget. Spends money."""
        live_gpus = sum(len(h['gpus']) for h in self.hosts if h['state'] in LIVE and not h.get('quarantined'))
        open_units = sum(u['state'] in ('queued', 'running') for u in self.units.values())
        need = min(self.s['gpus'], open_units) - live_gpus
        if need <= 0 or self.s['transport'] != 'ssh':
            return
        if self.spent() >= 0.9 * self.s['budget_usd']:
            self.log(f'not renting: spent ${self.spent():.2f} of ${self.s["budget_usd"]} budget')
            return
        types = self.verda.types()
        tries = [(True, t) for t in self.s['types']] + ([(False, t) for t in self.s['types']] if self.s['allow_on_demand'] else [])
        avail = {sp: self.verda.availability(sp) for sp in {sp for sp, _ in tries}}
        for spot, t in tries:
            n = types[t]['gpu']['number_of_gpus']
            for loc, offered in sorted(avail[spot].items()):
                if need < n or t not in offered:
                    continue
                price = float(types[t]['spot_price' if spot else 'price_per_hour'])
                if self.spent() + self.burn() + price > self.s['budget_usd']:
                    self.log(f'not renting {t}: would pass the ${self.s["budget_usd"]} budget')
                    return
                k = 1 + max([int(h['name'].rsplit('-', 1)[1]) for h in self.hosts] or [0])
                name = f"{self.s['tag']}-{k:02d}"
                try:
                    iid = self.verda.create(t, loc, name, spot, f'Pattern R3000 headline cloud run {self.home.name} (Claude)')
                except RuntimeError as e:
                    self.log(f'create {t} in {loc} failed: {e}')
                    continue
                h = {'name': name, 'instance_id': iid, 'type': t, 'location': loc, 'spot': spot, 'price_per_hour': price,
                     'gpus': list(range(n)), 'transport': 'ssh', 'root': str(ROOT), 'run_dir': str(self.home),
                     'python': REMOTE_PY, 'device': 'cuda', 'state': 'creating', 'created': now(), 'successes': 0,
                     'errors': 0, 'poll_failures': 0}
                with self.lock:
                    self.hosts.append(h)
                self.save()
                self.log(f'rented {name}: {t} {"spot" if spot else "on-demand"} in {loc} at ${price}/h (instance {iid})')
                threading.Thread(target=self.provision, args=(name,), daemon=True).start()
                need -= n
                if need <= 0:
                    return

    def set_state(self, h, state, **extra):
        with self.lock:
            h['state'] = state
            h.update(extra)
        self.save()

    def provision(self, name):
        h = self.host(name)
        try:
            if 'fake_dir' in h:   # test machine on this Mac: code and data are already here
                Path(h['run_dir']).mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.home / 'white_pixels.npy', Path(h['run_dir']) / 'white_pixels.npy')
                self.set_state(h, 'ready', ready=now())
                return
            self.set_state(h, 'booting')
            t0 = now()
            while True:
                info = self.verda.instance(h['instance_id'])
                if info.get('status') == 'running' and info.get('ip'):
                    break
                if now() - t0 > 1800:
                    raise RuntimeError(f"not running after 30 min (status {info.get('status')})")
                time.sleep(15)
            with self.lock:
                h['ip'], h['os_volume_id'] = info['ip'], info.get('os_volume_id')
            self.save()
            self.set_state(h, 'setup')
            for attempt in range(1, SETUP_TRIES + 1):
                try:
                    gpu_names = self.setup_once(h)
                    break
                except Exception as e:  # noqa: BLE001 - retried, then reported below
                    status = self.verda.instance(h['instance_id']).get('status')
                    if attempt == SETUP_TRIES or status != 'running':
                        raise
                    self.log(f"{name} setup attempt {attempt} failed ({str(e)[-400:]}); instance is {status}, retrying")
                    time.sleep(30)
            self.set_state(h, 'ready', ready=now(), gpu_names=gpu_names)
            self.log(f"{name} ready after {(now() - h['created']) / 60:.1f} min")
        except Exception as e:  # noqa: BLE001 - every provisioning failure is reported, never hidden
            self.set_state(h, 'setup_failed', error=str(e)[-1500:])
            self.log(f'{name} setup FAILED: {e}')
            if self.s['auto_delete']:
                self.delete_host(h, 'setup failed')
        finally:
            self.release_mac_lock(h)

    def setup_once(self, h):
        """One setup attempt over ssh; safe to repeat after a dropped connection or a reboot."""
        t0 = now()
        while self.run(h, 'true', timeout=30).returncode != 0:
            if now() - t0 > 900:
                raise RuntimeError('ssh not reachable after 15 min')
            time.sleep(15)
        r = self.run(h, WAIT_FOR_BOOT, timeout=2400)
        if r.returncode:
            raise RuntimeError(f'waiting for first-boot scripts failed: {(r.stdout + r.stderr)[-300:]}')
        key = (self.cloud / 'relay_key').read_text()
        pub = shlex.quote((self.cloud / 'relay_key.pub').read_text().strip())
        r = self.run(h, f'umask 077; mkdir -p /root/.ssh; cat > /root/.ssh/relay_key; '
                        f'grep -qxF {pub} /root/.ssh/authorized_keys || echo {pub} >> /root/.ssh/authorized_keys', input=key)
        if r.returncode:
            raise RuntimeError(f'relay key install failed: {r.stderr[-500:]}')
        setup_err = {}
        th = threading.Thread(target=self.install_stack, args=(h, setup_err))
        th.start()
        try:
            self.upload(h)
        finally:
            th.join()
        if setup_err:
            raise RuntimeError(setup_err['error'])
        self.verify_files(h)
        r = self.run(h, 'nvidia-smi -L')
        if r.returncode or len(r.stdout.strip().splitlines()) != len(h['gpus']):
            raise RuntimeError(f'GPU check failed: {r.stdout} {r.stderr}')
        self.run(h, f"mkdir -p {shlex.quote(h['run_dir'] + '/logs')}")
        return r.stdout.strip().splitlines()

    def release_mac_lock(self, h):
        with self.lock:
            if self.mac_lock_holder == h['name']:
                self.mac_lock_holder = None
                self.mac_lock.release()

    def install_stack(self, h, err):
        script = ('set -e; export DEBIAN_FRONTEND=noninteractive; '
                  'n=0; until apt-get -o DPkg::Lock::Timeout=900 update -qq; do n=$((n+1)); [ $n -lt 10 ] || exit 1; sleep 20; done; '
                  'n=0; until apt-get -o DPkg::Lock::Timeout=900 install -y -qq clang python3-venv; do n=$((n+1)); [ $n -lt 10 ] || exit 1; sleep 20; done; '
                  '[ -x /root/venv/bin/python ] || python3 -m venv /root/venv; '
                  '/root/venv/bin/pip install -q --upgrade pip; '
                  f'/root/venv/bin/pip install -q {PIP_PINS}; '
                  '/root/venv/bin/python -c "import torch; assert torch.cuda.is_available(); '
                  'print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"')
        r = self.run(h, f'bash -c {shlex.quote(script)} > /root/setup.log 2>&1; rc=$?; tail -3 /root/setup.log; exit $rc', timeout=2700)
        (self.cloud / 'hosts' / h['name']).mkdir(parents=True, exist_ok=True)
        (self.cloud / 'hosts' / h['name'] / 'setup_tail.txt').write_text(r.stdout + r.stderr)
        if r.returncode:
            err['error'] = f'software setup failed (rc {r.returncode}): {(r.stdout + r.stderr)[-800:]}'

    def upload(self, h):
        root = shlex.quote(h['root'])
        files = ' '.join(shlex.quote(f) for f in self.files)
        t0 = now()
        tried = set()
        while True:
            sources = [x for x in self.hosts if x['state'] == 'ready' and x['transport'] == 'ssh'
                       and x['name'] != h['name'] and x['name'] not in tried]
            for src in sources:   # machine-to-machine inside the data centre first
                tried.add(src['name'])
                pull = (f"ssh -i /root/.ssh/relay_key -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/root/.ssh/relay_known "
                        f"-o BatchMode=yes -o ConnectTimeout=15 root@{src['ip']} {shlex.quote(f'cd {root} && tar cf - {files}')} "
                        f"| tar xf - -C {root}")
                r = self.run(h, f'set -o pipefail; mkdir -p {root} && {pull}', timeout=3600)
                if r.returncode == 0:
                    self.log(f"{h['name']}: copied files from {src['name']} in {(now() - t0) / 60:.1f} min")
                    return
                self.log(f"{h['name']}: relay from {src['name']} failed ({r.stderr.strip()[-200:]}), trying next source")
            if self.mac_lock_holder == h['name']:
                break
            if self.mac_lock.acquire(timeout=20):
                # the previous holder may have just become ready: relay from it rather than upload again
                if any(x['state'] == 'ready' and x['transport'] == 'ssh' and x['name'] != h['name']
                       and x['name'] not in tried for x in self.hosts):
                    self.mac_lock.release()
                    continue
                with self.lock:
                    self.mac_lock_holder = h['name']
                break
        pipe = (f'cd {shlex.quote(str(ROOT))} && COPYFILE_DISABLE=1 tar --no-xattrs -cf - {files} | '
                + self.remote_shell(h, f'mkdir -p {root} && tar xf - -C {root}'))
        r = subprocess.run(['bash', '-c', 'set -o pipefail; ' + pipe], capture_output=True, text=True, timeout=7200)
        if r.returncode:
            err = '\n'.join(l for l in r.stderr.splitlines() if 'unknown extended header' not in l)
            raise RuntimeError(f'upload from the Mac failed: {err[-500:]}')
        self.log(f"{h['name']}: uploaded {self.manifest['bytes'] / 1e9:.2f} GB from the Mac in {(now() - t0) / 60:.1f} min")

    def verify_files(self, h):
        r = self.run(h, f"cd {shlex.quote(h['root'])} && sha256sum -- {' '.join(shlex.quote(f) for f in self.files)}", timeout=900)
        got = {}
        for line in r.stdout.splitlines():
            digest, _, path = line.partition('  ')
            got[path] = digest
        bad = [f for f, d in self.manifest['files'].items() if got.get(f) != d]
        if r.returncode or bad:
            raise RuntimeError(f'{len(bad)} files fail the hash check, e.g. {bad[:3]} ({r.stderr[-200:]})')
        self.log(f"{h['name']}: all {len(self.files)} file hashes match")

    def delete_host(self, h, why):
        if 'fake_dir' in h:
            self.set_state(h, 'deleted', ended=h.get('ended') or now())
            return
        if not self.s['auto_delete']:
            return
        try:
            info = self.verda.instance(h['instance_id'])
            if info.get('hostname') != h['name']:
                raise RuntimeError(f"hostname mismatch ({info.get('hostname')} != {h['name']}); not deleting")
            vols = [info['os_volume_id']] if info.get('os_volume_id') else []
            self.verda.delete(h['instance_id'], vols)
            self.set_state(h, 'deleted', ended=h.get('ended') or now(), deleted_why=why)
            self.log(f"deleted {h['name']} ({why}); its disk is removed and verified by the disk sweep")
        except Exception as e:  # noqa: BLE001
            self.log(f"delete of {h['name']} FAILED: {e}")

    def sweep_disks(self, final=False, wait_s=900):
        """Delete the recorded OS disk of every machine of this run that is deleted; with final=True,
        keep re-listing until none of this run's disks remain. Returns the names still present."""
        if self.s['transport'] != 'ssh' or not self.s['auto_delete']:
            return []
        ours = {h['os_volume_id']: h['name'] for h in self.hosts if h.get('os_volume_id')}
        t0 = now()
        while True:
            try:
                vols = [v for v in self.verda.volumes() if v['id'] in ours]
            except Exception as e:  # noqa: BLE001
                self.log(f'disk sweep: listing volumes failed: {e}')
                vols = None
            if vols is not None:
                gone_hosts = {h['name'] for h in self.hosts if h['state'] == 'deleted'}
                free = [v for v in vols if ours[v['id']] in gone_hosts and not v.get('instance_id')]
                if free:
                    try:
                        self.verda.delete_volumes([v['id'] for v in free])
                        self.log(f"disk sweep: deleted {len(free)} leftover disk(s): {sorted(v['name'] for v in free)}")
                    except Exception as e:  # noqa: BLE001
                        self.log(f'disk sweep: delete failed: {e}')
                if not final or not vols:
                    return []
            if now() - t0 > wait_s:
                left = sorted(v['name'] for v in (vols or []))
                self.log(f'DISK CLEANUP FAILED: still present after {wait_s}s: {left}')
                return left
            time.sleep(30)

    # --- units
    def running_on(self, h):
        return [u for u in self.units.values() if u['state'] == 'running' and u['host'] == h['name']]

    def launch(self, h, gpu, name):
        u = self.units[name]
        rdir = f"{h['run_dir']}/Y{u['year']}/seed_{u['seed']}"
        partial = self.cloud / 'partial' / name
        resumed = False
        if partial.exists() and u.get('host') != h['name']:
            r = self.push(h, partial, rdir)
            if r.returncode:
                self.log(f'{name}: could not copy its checkpoint to {h["name"]} ({r.stderr[-200:]}); starting fresh')
            else:
                resumed = True
        cmd = (f"{self.agent(h)} launch --root {shlex.quote(h['root'])} --run-dir {shlex.quote(h['run_dir'])} "
               f"--python {shlex.quote(h['python'])} --device {h['device']} --gpu {gpu} --year {u['year']} --seed {u['seed']}"
               + (' --smoke' if self.s['smoke'] else ''))
        r = self.run(h, cmd)
        try:
            pid = json.loads(r.stdout.strip().splitlines()[-1])['pid']
        except (ValueError, IndexError, KeyError):
            self.log(f"{name}: launch on {h['name']} failed: {r.stdout[-300:]} {r.stderr[-300:]}")
            h['launch_failures'] = h.get('launch_failures', 0) + 1
            if h['launch_failures'] >= 3:
                h['quarantined'] = True
                self.log(f"{h['name']} quarantined: three failed launches")
            return
        with self.lock:
            u.update(state='running', host=h['name'], gpu=gpu, pid=pid, launched=now(), attempts=u['attempts'] + 1,
                     resumed_from_checkpoint=resumed, partial_synced=None)
        self.save()
        self.log(f"{name} -> {h['name']} gpu {gpu} (attempt {u['attempts']}{', resuming from checkpoint' if resumed else ''})")

    def requeue(self, u, why, counter):
        with self.lock:
            u['history'].append({'host': u.get('host'), 'why': why, 'utc': utc()})
            u[counter] += 1
            limit = {'errors': 2, 'deaths': 3, 'preemptions': 10}[counter]
            u['state'] = 'failed' if u[counter] >= limit else 'queued'
            u['pid'] = None
        self.save()
        self.log(f"{unit_name(u['year'], u['seed'])}: {why} -> {u['state']}")

    def collect(self, h, name):
        u = self.units[name]
        y, s = u['year'], u['seed']
        r = self.pull(h, [f'Y{y}/seed_{s}', f'Y{y}/sample_counts.json', f'logs/Y{y}_s{s}.log'], self.home)
        if r.returncode:
            self.log(f'{name}: copy home failed ({r.stderr[-200:]}); will retry')
            return
        sd = self.seed_dir(u)
        counts = read_json(self.home / f'Y{y}' / 'sample_counts.json', {})
        try:
            sc = np.load(sd / 'scores.npy')
            ok = len(sc) == counts.get('test_all') and np.isfinite(sc).all()
        except (OSError, ValueError):
            ok = False
        if not ok:
            self.log(f'{name}: copied scores fail the check (length or non-finite); will retry the copy')
            return
        hd = self.cloud / 'hosts' / h['name']
        if not (hd / 'config.json').exists():
            self.pull(h, ['config.json', 'environment.json'], hd)
        atomic_json(sd / 'cloud_provenance.json', {'host': h['name'], 'instance_type': h.get('type'), 'location': h.get('location'),
                                                   'gpu': (h.get('gpu_names') or [None])[0], 'attempts': u['attempts'],
                                                   'resumed_from_checkpoint': u.get('resumed_from_checkpoint'),
                                                   'history': u['history'], 'launched_utc': utc(u['launched']), 'collected_utc': utc()})
        shutil.rmtree(self.cloud / 'partial' / name, ignore_errors=True)
        with self.lock:
            u.update(state='done', finished=now(), hours=round((now() - u['launched']) / 3600, 3))
            h['successes'] = h.get('successes', 0) + 1
        self.save()
        self.log(f"{name} done on {h['name']} in {u['hours'] * 60:.1f} min (est {u['est_hours'] * 60:.0f})")

    def sync_partial(self, h, name, mtime):
        u = self.units[name]
        tmp = self.cloud / 'partial' / f'{name}.tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        r = self.pull(h, [f"Y{u['year']}/seed_{u['seed']}/{f}" for f in ('last.pt', 'best.pt', 'epochs.csv', 'normalization.json')], tmp)
        if r.returncode:
            return
        final = self.cloud / 'partial' / name
        shutil.rmtree(final, ignore_errors=True)
        (tmp / f"Y{u['year']}" / f"seed_{u['seed']}").rename(final)
        shutil.rmtree(tmp, ignore_errors=True)
        with self.lock:
            u['partial_synced'], u['partial_synced_at'] = mtime, now()

    def poll_host(self, h):
        units = self.running_on(h)
        payload = json.dumps([{'year': u['year'], 'seed': u['seed'], 'pid': u['pid']} for u in units])
        r = self.run(h, f"{self.agent(h)} poll --run-dir {shlex.quote(h['run_dir'])}", input=payload, timeout=90)
        try:
            res = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            h['poll_failures'] = h.get('poll_failures', 0) + 1
            self.log(f"{h['name']}: poll failed ({h['poll_failures']} in a row): {r.stderr.strip()[-200:]}")
            if h['poll_failures'] >= 3:
                self.check_lost(h)
            return
        h['poll_failures'] = 0
        h['last_seen'], h['host_info'] = now(), res['host']
        for u in units:
            name = unit_name(u['year'], u['seed'])
            rec = res['units'].get(name, {})
            u['last_poll'] = rec
            if rec.get('done'):
                self.collect(h, name)
            elif rec.get('failed'):
                fd = self.cloud / 'failures' / f"{name}_{u['attempts']}"
                self.pull(h, [f"Y{u['year']}/seed_{u['seed']}/FAILED.json", f"logs/Y{u['year']}_s{u['seed']}.log"], fd)
                h['errors'] = h.get('errors', 0) + 1
                self.requeue(u, f"raised an error on {h['name']}: {rec.get('error', '')[-300:]}", 'errors')
                if h['errors'] >= 2 and not h.get('successes'):
                    h['quarantined'] = True
                    self.log(f"{h['name']} quarantined: two failures and no successes")
            elif not rec.get('alive'):
                self.requeue(u, f"process died on {h['name']} without a result", 'deaths')
            else:
                prog = rec.get('progress') or {}
                try:
                    last = max(u['launched'], datetime.fromisoformat(prog['updated_utc']).timestamp())
                except (KeyError, ValueError, TypeError):
                    last = u['launched']
                if now() - last > STALE_S:
                    self.run(h, f"kill {u['pid']}")
                    self.requeue(u, f"no progress for {STALE_S // 60} min on {h['name']}; killed", 'deaths')
                elif rec.get('last_mtime') and rec['last_mtime'] != u.get('partial_synced') and \
                        now() - (u.get('partial_synced_at') or 0) >= self.s['partial_sync_s']:
                    self.sync_partial(h, name, rec['last_mtime'])
        self.save()

    def check_lost(self, h):
        gone = h['poll_failures'] >= 15
        if 'fake_dir' not in h:
            try:
                st = self.verda.instance(h['instance_id']).get('status')
            except RuntimeError as e:
                st = f'lookup failed: {e}'
            h['verda_status'] = st
            gone = gone or st not in ('running', 'provisioning', 'ordered', 'validating')
        else:
            gone = True   # local test host that is marked down
        if not gone:
            return
        self.log(f"{h['name']} LOST (Verda status {h.get('verda_status')}, {h['poll_failures']} failed polls); requeueing its units")
        self.set_state(h, 'lost', ended=now())
        for u in self.running_on(h):
            self.requeue(u, f"machine {h['name']} lost", 'preemptions')
        self.delete_host(h, 'lost')

    def dispatch(self):
        if self.spent() >= self.s['budget_usd']:
            return
        for h in self.hosts:
            if h['state'] != 'ready' or h.get('quarantined'):
                continue
            busy = {u['gpu'] for u in self.running_on(h)}
            for gpu in h['gpus']:
                if gpu in busy:
                    continue
                queued = [n for n, u in self.units.items() if u['state'] == 'queued']
                if not queued:
                    return
                bad = {n for n in queued if any(x['host'] == h['name'] and 'error' in x['why'] for x in self.units[n]['history'])}
                pick = sorted(queued, key=lambda n: (n in bad, -self.units[n]['est_hours'], n))[0]
                self.launch(h, gpu, pick)

    def retire_idle(self):
        if any(u['state'] == 'queued' for u in self.units.values()):
            return
        for h in self.hosts:
            if h['state'] == 'ready' and not self.running_on(h):
                self.set_state(h, 'idle', ended=now()) if not self.s['auto_delete'] and h['transport'] == 'ssh' else None
                self.delete_host(h, 'no work left')

    def write_status(self):
        c = {}
        for u in self.units.values():
            c[u['state']] = c.get(u['state'], 0) + 1
        running = [{'unit': unit_name(u['year'], u['seed']), 'host': u['host'], 'gpu': u['gpu'],
                    'minutes': round((now() - u['launched']) / 60, 1), 'progress': (u.get('last_poll') or {}).get('progress'),
                    'epochs_done': (u.get('last_poll') or {}).get('epochs_done')}
                   for u in self.units.values() if u['state'] == 'running']
        slots = sum(len(h['gpus']) for h in self.hosts if h['state'] in LIVE and not h.get('quarantined')) or 1
        left = sum(u['est_hours'] for u in self.units.values() if u['state'] == 'queued') + \
            sum(max(0.0, u['est_hours'] - (now() - u['launched']) / 3600) for u in self.units.values() if u['state'] == 'running')
        atomic_json(self.cloud / 'status.json', {
            'state': self.state, 'units': c, 'total_units': len(self.units), 'running': running,
            'hosts': [{k: h.get(k) for k in ('name', 'type', 'location', 'spot', 'state', 'price_per_hour', 'quarantined',
                                             'successes', 'errors', 'poll_failures', 'error')} for h in self.hosts],
            'spent_usd': round(self.spent(), 2), 'burn_usd_per_hour': round(self.burn(), 3), 'budget_usd': self.s['budget_usd'],
            'remaining_gpu_hours_est': round(left, 1), 'eta_hours_est': round(left / slots, 1), 'updated_utc': utc()})

    # --- main loop
    def loop(self, chaos=None):
        self.state = 'running'
        for h in self.hosts:   # resumed dispatcher: finish provisioning anything half-built
            if h['state'] in ('creating', 'booting', 'setup'):
                threading.Thread(target=self.provision, args=(h['name'],), daemon=True).start()
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop', True))
        self.log(f"dispatcher up: {len(self.units)} units, {sum(u['state'] == 'done' for u in self.units.values())} already done")
        while not self.stop:
            if self.s['transport'] == 'ssh' and now() - self.last_fill >= self.s['fill_interval_s']:
                self.last_fill = now()
                try:
                    self.create_hosts()
                except Exception as e:  # noqa: BLE001
                    self.log(f'renting step failed: {e}')
            for h in list(self.hosts):
                if h['state'] == 'ready':
                    self.poll_host(h)
            if chaos:
                chaos(self)
            self.dispatch()
            self.retire_idle()
            if now() - self.last_sweep >= 300:
                self.last_sweep = now()
                self.sweep_disks()
            if self.spent() >= self.s['budget_usd'] and self.state == 'running':
                self.state = 'budget_exhausted'
                self.log(f'BUDGET REACHED (${self.spent():.2f}); no new units will start')
            self.write_status()
            open_units = [u for u in self.units.values() if u['state'] in ('queued', 'running')]
            if not open_units:
                break
            if self.state == 'budget_exhausted' and not any(u['state'] == 'running' for u in self.units.values()):
                break
            if not any(h['state'] in LIVE for h in self.hosts) and self.s['transport'] == 'local':
                self.log('no live test hosts left')
                break
            time.sleep(self.s['poll_s'])
        failed = [n for n, u in self.units.items() if u['state'] != 'done']
        self.state = 'stopped' if self.stop else ('complete' if not failed else 'finished_with_failures')
        self.write_status()
        self.log(f'dispatcher finished: {self.state}; not done: {failed[:10]}{" ..." if len(failed) > 10 else ""}')
        return failed


def assemble(home, smoke=False, report_args=()):
    """Ensembles for every complete year (train.py on CPU only assembles here), then report.py."""
    home = Path(home)
    complete = [y for y in YEARS if all((home / f'Y{y}' / f'seed_{s}' / 'scores.npy').exists() for s in SEEDS)]
    if not complete:
        raise RuntimeError('no complete year to assemble')
    for hd in (home / 'cloud' / 'hosts').glob('*'):
        cfg = read_json(hd / 'config.json')
        if cfg is not None and (home / 'config.json').exists() and cfg != read_json(home / 'config.json'):
            raise RuntimeError(f'{hd.name} ran with a different config.json than the home run')
    with open(home / 'assemble.log', 'a') as log:
        r = subprocess.run([str(VENV_PY), '-u', str(HERE / 'train.py'), '--run-dir', str(home), '--device', 'cpu',
                            '--years', *map(str, complete)] + (['--smoke'] if smoke else []), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode or not all((home / f'Y{y}' / 'COMPLETE.json').exists() for y in complete):
        raise RuntimeError(f'ensemble assembly failed; see {home / "assemble.log"}')
    for hd in (home / 'cloud' / 'hosts').glob('*'):
        cfg = read_json(hd / 'config.json')
        if cfg is not None and cfg != read_json(home / 'config.json'):
            raise RuntimeError(f'{hd.name} ran with a different config.json than the home run')
    args = list(report_args) + (['--allow-partial'] if len(complete) < len(YEARS) and '--allow-partial' not in report_args else [])
    with open(home / 'report.log', 'a') as log:
        r = subprocess.run([str(VENV_PY), '-u', str(HERE / 'report.py'), '--run-dir', str(home), *args], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f'report.py failed; see {home / "report.log"}')
    return complete


# ----------------------------------------------------------------------------- commands
def cmd_plan(a):
    import pandas as pd
    home = Path(a.run_dir)
    est = unit_hours(pd.read_csv(home / 'cloud' / 'fold_workload.csv'))
    hours = sum(est[y] for y in YEARS) * len(SEEDS)
    print(f'{len(YEARS) * len(SEEDS)} units, {hours:.0f} A100-hours estimated at {EPOCHS} epochs per seed; '
          f'longest unit {max(est.values()):.2f} h, shortest {min(est.values()):.2f} h')
    v = Verda()
    types = v.types()
    for t in a.types:
        print(f"  {t}: {types[t]['gpu']['description']}, spot ${types[t]['spot_price']}/h, on demand ${types[t]['price_per_hour']}/h")
    for spot in (True, False):
        av = v.availability(spot)
        print(f"  {'spot' if spot else 'on demand'} now: " + '; '.join(f'{loc} {[t for t in ts if t in a.types]}' for loc, ts in sorted(av.items())))
    price = float(types[a.types[0]]['spot_price'])
    print(f'  {a.gpus} GPUs: ~{hours / a.gpus:.0f} h wall (+ ~0.5 h setup), ~${hours * price:.0f} at ${price}/GPU-h spot')
    print(f"  current Verda instances: {[i['hostname'] for i in v.instances()]}")


def cmd_start(a):
    if not a.confirm_spend:
        sys.exit('start rents GPUs and spends money; add --confirm-spend once Arjun has said go')
    home = Path(a.run_dir).resolve()
    if (home / 'cloud' / 'settings.json').exists():
        sys.exit('this run was already started; use resume')
    settings = {'transport': 'ssh', 'gpus': a.gpus, 'budget_usd': a.budget_usd, 'types': a.types,
                'allow_on_demand': a.allow_on_demand, 'auto_delete': a.auto_delete, 'smoke': False,
                'years': YEARS, 'seeds': SEEDS, 'poll_s': 60, 'partial_sync_s': 900, 'fill_interval_s': 600,
                'tag': 'pr3k-' + home.name.rsplit('_', 1)[-1][-7:-1].lower(), 'started_utc': utc()}
    atomic_json(home / 'cloud' / 'settings.json', settings)
    finish(home, Dispatcher(home, settings))


def cmd_resume(a):
    home = Path(a.run_dir).resolve()
    settings = read_json(home / 'cloud' / 'settings.json')
    if a.auto_delete:
        settings['auto_delete'] = True
        atomic_json(home / 'cloud' / 'settings.json', settings)
    finish(home, Dispatcher(home, settings))


def finish(home, d):
    failed = d.loop()
    if not failed:
        d.log('all units home; assembling ensembles and running report.py')
        try:
            assemble(home)
            d.log('ensembles and report written')
        except RuntimeError as e:
            d.log(f'ASSEMBLY FAILED: {e}')
    left = [h['name'] for h in d.hosts if h['state'] in LIVE + ('idle',)]
    if left:
        d.log(f'machines still rented (no --auto-delete): {left}; run teardown --confirm-delete when Arjun agrees')
    elif d.s['auto_delete']:
        disks = d.sweep_disks(final=True)
        d.log('disk sweep: no disks of this run remain' if not disks else f'DISKS STILL BILLING: {disks}')


def cmd_status(a):
    st = read_json(Path(a.run_dir) / 'cloud' / 'status.json')
    print(json.dumps(st, indent=2) if st else 'no status yet')


def cmd_teardown(a):
    if not a.confirm_delete:
        sys.exit('teardown deletes machines; add --confirm-delete once Arjun has said so')
    home = Path(a.run_dir).resolve()
    s = read_json(home / 'cloud' / 'settings.json')
    s['auto_delete'] = True
    d = Dispatcher(home, s)
    for h in d.hosts:
        if h['state'] not in ('deleted',):
            d.delete_host(h, 'teardown')
    disks = d.sweep_disks(final=True)
    d.log('teardown: no disks of this run remain' if not disks else f'teardown: DISKS STILL BILLING: {disks}')
    if disks:
        sys.exit(f'disks still present: {disks}')


def cmd_localtest(a):
    """Full dispatcher test on CPU: 2 fake machines, years 2000-2001 x 5 seeds, smoke-sized."""
    home = prepare(RUNS / f"cloud_localtest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    years = [2000, 2001]
    s = {'transport': 'local', 'gpus': 2, 'budget_usd': 1e9, 'types': [], 'allow_on_demand': False, 'auto_delete': True,
         'smoke': True, 'years': years, 'seeds': SEEDS, 'poll_s': 5, 'partial_sync_s': 0, 'fill_interval_s': 1e9, 'tag': 'local'}
    atomic_json(home / 'cloud' / 'settings.json', s)

    if a.ssh_shim:   # stand-in ssh: same options as real ssh, runs the remote command the way sshd does
        shim = home / 'cloud' / 'ssh_shim'
        shim.mkdir()
        (shim / 'ssh').write_text('#!/bin/bash\n# test stand-in for ssh: skip options, drop the host, run the rest as sshd would\n'
                                  'while [ $# -gt 0 ]; do case "$1" in -i|-o|-p|-l) shift 2;; -*) shift;; *) break;; esac; done\n'
                                  'shift\nexec bash -c "$*"\n')
        (shim / 'ssh').chmod(0o755)
        os.environ['PATH'] = f"{shim}:{os.environ['PATH']}"

    def fake(name):
        fd = home / 'cloud' / 'fakehosts' / name
        return {'name': name, 'transport': 'ssh' if a.ssh_shim else 'local', 'ip': 'shim', 'root': str(ROOT), 'run_dir': str(fd / 'run'), 'fake_dir': str(fd),
                'python': str(VENV_PY), 'device': 'cpu', 'gpus': [0], 'state': 'creating', 'created': now(),
                'price_per_hour': 0, 'successes': 0, 'errors': 0, 'poll_failures': 0}
    d = Dispatcher(home, s)
    d.hosts = [fake('local-a'), fake('local-b')]
    d.save()
    for h in d.hosts:
        d.provision(h['name'])

    def chaos(disp):  # take local-b down once it has a unit past start-up
        if disp.chaos_done:
            return
        b = disp.host('local-b')
        for u in disp.running_on(b):
            if (u.get('last_poll') or {}).get('progress'):
                disp.log(f"TEST: taking local-b down while {unit_name(u['year'], u['seed'])} runs")
                os.kill(u['pid'], signal.SIGKILL)
                (Path(b['fake_dir']) / 'DOWN').touch()
                disp.chaos_done = True
    failed = d.loop(chaos=chaos)
    assert not failed, f'units not done: {failed}'
    lost = [u for u in d.units.values() if u['preemptions']]
    assert d.host('local-b')['state'] in ('lost', 'deleted') and lost, 'machine loss was not exercised'
    d.log(f'TEST: machine loss handled; requeued and finished: {[unit_name(u["year"], u["seed"]) for u in lost]}')

    # Resume-from-checkpoint: rebuild one finished unit on a fresh machine from its checkpoint only.
    probe = d.units['Y2001_s3']
    sd = d.seed_dir(probe)
    original = np.load(sd / 'scores.npy')
    part = home / 'cloud' / 'partial' / 'Y2001_s3'
    part.mkdir(parents=True)
    for f in ('last.pt', 'best.pt', 'epochs.csv', 'normalization.json'):
        shutil.copy2(sd / f, part / f)
    shutil.rmtree(sd)
    for f in ('COMPLETE.json', 'ensemble_scores.parquet'):
        (home / 'Y2001' / f).unlink(missing_ok=True)
    probe.update(state='queued', host='local-a')
    d.hosts.append(fake('local-c'))
    d.host('local-a')['state'] = 'deleted'
    d.save()
    d.provision('local-c')
    d.stop = False
    failed = d.loop()
    assert not failed
    again = np.load(d.seed_dir(probe) / 'scores.npy')
    log = (home / 'logs' / 'Y2001_s3.log').read_text()
    resumed = d.units['Y2001_s3']['resumed_from_checkpoint'] and 'EPOCH' not in log.split('=== launch')[-1]
    same = np.array_equal(original, again)
    d.log(f'TEST: resume from checkpoint on a new machine: resumed without retraining={resumed}, scores identical={same}')
    assert resumed and same
    complete = assemble(home, smoke=True, report_args=['--allow-partial', '--min-names', '1'])
    d.log(f'TEST: ensembles and report written for {complete}')
    atomic_json(home / 'cloud' / 'localtest_result.json', {'passed': True, 'ssh_shim': a.ssh_shim, 'units': len(d.units), 'machine_loss_requeued':
                [unit_name(u['year'], u['seed']) for u in lost], 'resume_identical_scores': same, 'assembled_years': complete, 'utc': utc()})
    print(f'LOCALTEST PASSED: {home}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prepare').add_argument('--run-dir')
    p = sub.add_parser('plan')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--gpus', type=int, default=4)
    p.add_argument('--types', nargs='+', default=DEFAULT_TYPES)
    p = sub.add_parser('start')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--gpus', type=int, required=True)
    p.add_argument('--budget-usd', type=float, required=True)
    p.add_argument('--types', nargs='+', default=DEFAULT_TYPES)
    p.add_argument('--allow-on-demand', action='store_true')
    p.add_argument('--auto-delete', action='store_true')
    p.add_argument('--confirm-spend', action='store_true')
    p = sub.add_parser('resume')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--auto-delete', action='store_true')
    sub.add_parser('status').add_argument('--run-dir', required=True)
    p = sub.add_parser('teardown')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--confirm-delete', action='store_true')
    sub.add_parser('assemble').add_argument('--run-dir', required=True)
    sub.add_parser('localtest').add_argument('--ssh-shim', action='store_true',
                                             help='route every machine command through the real ssh code path (stand-in ssh)')
    a = ap.parse_args()
    if a.cmd == 'prepare':
        prepare(a.run_dir)
    elif a.cmd == 'assemble':
        print(assemble(a.run_dir))
    else:
        {'plan': cmd_plan, 'start': cmd_start, 'resume': cmd_resume, 'status': cmd_status,
         'teardown': cmd_teardown, 'localtest': cmd_localtest}[a.cmd](a)


if __name__ == '__main__':
    main()
