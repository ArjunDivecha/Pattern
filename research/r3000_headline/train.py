"""
Script: train.py
Version: 1.0 (2026-10-10)

Description:
    Expanding-window annual retraining for the R3000 headline rerun (I20/R20, 5-seed
    ChartCNN ensemble), reading the v2 frozen snapshot and the sample index built by
    prepare.py. Images are rendered on the fly by the compiled renderer (no 83 GB cache);
    rendering is checked against the reference renderer at start-up.

    Folds (one per test year, 2000 to 2026 by default):
      test        every session of the year (all samples scored, known label or not)
      validation  the latest 30% of sessions before the test year, ending GAP sessions
                  before the first test session
      train       all earlier sessions from 1996, ending GAP sessions before validation
      GAP = 58 sessions = 38 sessions of chart+MA history + 20-session horizon, so no
      training label overlaps a validation input window, and likewise val -> test.
    Model recipe follows the original R1000 headline (configs/prod_expanding.yaml):
    ChartCNN defaults (3 blocks, 64/128/256), Adam lr 1e-5, batch 128, early stopping
    on validation loss with patience 2, max 100 epochs, train-only 50/50 undersampling.
    Label = 1 if the simple 20-session TRI return > 0 (sign identical to the log return).

    Work unit = (test year, seed). Each unit writes scores.npy; when all seeds of a year
    exist the ensemble mean is written to ensemble_scores.parquet. Units already done
    are skipped, so the run resumes and can be sharded by --years/--seeds across GPUs
    or machines that share the run directory.

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010)
    <index dir>/index.parquet and state.json from prepare.py (default
        /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/)
Output files (in --run-dir):
    config.json, environment.json, folds.csv, white_pixels.npy, status.json,
    Y<year>/seed_<s>/{normalization.json, progress.json, epochs.csv, best.pt, last.pt, scores.npy},
    Y<year>/ensemble_scores.parquet, Y<year>/COMPLETE.json, train.log (tee'd by caller)

Usage:
    .venv/bin/python research/r3000_headline/train.py --run-dir DIR [--device mps|cuda]
        [--years 2000 2001 ...] [--seeds 0 1 2 3 4] [--smoke]
    --smoke: tiny subsample (20k train, 5k val, 5k test, 1 epoch) to test the pipeline.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import threading
import queue
import time
import traceback

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pattern.imaging.fast import load_renderer  # noqa: E402
from pattern.imaging.renderer import render_window  # noqa: E402
from pattern.research import full_train  # noqa: E402
from pattern.research.full_train import fit, score  # noqa: E402
from prepare import resolve_dataset, START, WINDOW, MA, H, HISTORY  # noqa: E402

GAP = HISTORY - 1 + H
CFG = {'window': WINDOW, 'ma': MA, 'horizon': H, 'gap_sessions': GAP, 'val_fraction': 0.30,
       'train_start': str(START.date()), 'batch_size': 128, 'learning_rate': 1e-5, 'max_epochs': 100,
       'patience': 2, 'seeds': [0, 1, 2, 3, 4], 'first_test': 2000, 'last_test': 2026,
       'model': 'ChartCNN defaults (3 blocks, 64/128/256)', 'target': 'sign of simple 20-session TRI return',
       'class_balance': 'undersample majority, train only'}
T0 = time.time()


def log(msg):
    print(f'[{time.time()-T0:8.1f}s] {msg}', flush=True)


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


class OnTheFlyImages:
    """images[ids] -> uint8 (n,1,64,60), rendered from one global OHLCV array."""

    def __init__(self, ohlcv, tri, ends, render):
        self.ohlcv, self.tri, self.ends, self.render = ohlcv, tri, ends, render

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, ids):
        return self.render(self.ohlcv, self.tri, self.ends[np.asarray(ids)], WINDOW, MA)


def prefetching_batches(images, ids, labels, mean, std, batch_size, shuffle_seed=None):
    """Same batches as full_train.batches, rendered one step ahead in a thread."""
    q = queue.Queue(maxsize=4)

    def work():
        for item in _plain_batches(images, ids, labels, mean, std, batch_size, shuffle_seed):
            q.put(item)
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


_plain_batches = full_train.batches


def load_panel(root, securities):
    filters = [('date', '>=', START), ('security', 'in', securities)]
    p = pd.read_parquet(root / 'prices.parquet', filters=filters,
                        columns=['security', 'date', 'open', 'high', 'low', 'close', 'volume', 'total_return_index_gross_dividends'])
    q = pd.read_parquet(root / 'quality.parquet', columns=['security', 'date', 'off_exchange_calendar'], filters=filters)
    p = p.merge(q, on=['security', 'date'], validate='one_to_one')
    p = p[~p.off_exchange_calendar].sort_values(['security', 'date']).reset_index(drop=True)
    p['security'] = p.security.astype(str)
    return p


def folds(sessions, index_dates, first, last):
    cal = pd.DatetimeIndex(sessions)
    begin = cal.searchsorted(index_dates.min())
    out = []
    for year in range(first, last + 1):
        test = np.flatnonzero(cal.year == year)
        test = test[cal[test] <= index_dates.max()]
        if not len(test):
            continue
        tv_stop = test[0] - GAP              # exclusive end of validation
        n_tv = tv_stop - begin
        n_val = int(round(n_tv * CFG['val_fraction']))
        val_start = tv_stop - n_val
        train_stop = val_start - GAP          # exclusive end of train
        if train_stop - begin < 252:
            raise ValueError(f'Insufficient training history for {year}')
        out.append({'year': year, 'train': (cal[begin], cal[train_stop - 1]),
                    'validation': (cal[val_start], cal[tv_stop - 1]), 'test': (cal[test[0]], cal[test[-1]])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--index-dir', default=str(ROOT / 'cache/r3000_I20_M20_v2_index'))
    ap.add_argument('--device', default='mps')
    ap.add_argument('--years', type=int, nargs='*')
    ap.add_argument('--seeds', type=int, nargs='*')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root, ref = resolve_dataset()
    index_dir = Path(a.index_dir)
    istate = json.loads((index_dir / 'state.json').read_text())
    if istate['state'] != 'complete' or istate['config']['manifest_sha256'] != ref['manifest_sha256'] or istate['config']['limit']:
        raise ValueError('Sample index incomplete, partial, or built from another snapshot')
    cfg = dict(CFG, smoke=a.smoke, dataset=ref, index_state=istate['config'],
               code_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                            ['research/r3000_headline/train.py', 'research/r3000_headline/prepare.py',
                             'pattern/research/full_train.py', 'pattern/models/cnn.py', 'pattern/models/blocks.py',
                             'pattern/config.py', 'pattern/imaging/fast_renderer.c']})
    if (out / 'config.json').exists():
        if json.loads((out / 'config.json').read_text()) != json.loads(json.dumps(cfg, default=str)):
            raise ValueError('Run configuration or code changed; start a new run directory')
    else:
        atomic_json(out / 'config.json', cfg)
    device = torch.device(a.device)
    if a.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no silent CPU fallback')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    atomic_json(out / 'environment.json', {'python': sys.version, 'torch': torch.__version__, 'device': str(device),
                                           'pid': os.getpid(), 'host': os.uname().nodename})
    full_train.batches = prefetching_batches  # fit/score call module-level batches

    index = pd.read_parquet(index_dir / 'index.parquet', columns=['row_id', 'security', 'date', 'session_position', 'forward_return_20'])
    log(f'index: {len(index):,} samples')
    p = load_panel(root, sorted(index.security.unique()))
    p['row'] = np.arange(len(p), dtype='int64')
    ends = index[['security', 'date']].merge(p[['security', 'date', 'row']], on=['security', 'date'], how='left', validate='one_to_one').row
    if ends.isna().any():
        raise ValueError('Sample index rows missing from panel')
    ends = ends.to_numpy('int64')
    ohlcv = np.ascontiguousarray(p[['open', 'high', 'low', 'close', 'volume']].to_numpy(float))
    tri = np.ascontiguousarray(p.total_return_index_gross_dividends.to_numpy(float))
    sec_codes = p.security.astype('category').cat.codes.to_numpy()
    if not ((ends >= HISTORY - 1).all() and (sec_codes[ends] == sec_codes[ends - HISTORY + 1]).all()):
        raise ValueError('A sample window crosses a security boundary')
    del p
    render = load_renderer(out / 'compiled')
    images = OnTheFlyImages(ohlcv, tri, ends, render)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(ends), 300, replace=False):
        e = ends[i]
        if not np.array_equal(images[[i]][0], render_window(ohlcv[e - HISTORY + 1:e + 1], tri[e - HISTORY + 1:e + 1],
                                                            window=WINDOW, width=3 * WINDOW, ma_length=MA)):
            raise AssertionError(f'On-the-fly render parity failure at sample {i}')
    log('on-the-fly renderer matches reference on 300 random samples')

    wp_path = out / 'white_pixels.npy'
    if wp_path.exists():
        white = np.load(wp_path)
    else:
        white = np.empty(len(ends), dtype='int16')
        for k in range(0, len(ends), 100_000):
            white[k:k + 100_000] = np.count_nonzero(images[np.arange(k, min(k + 100_000, len(ends)))], axis=(1, 2, 3))
        np.save(str(wp_path) + '.tmp.npy', white)
        Path(str(wp_path) + '.tmp.npy').replace(wp_path)
    index['white_pixels'] = white
    log('white-pixel counts ready')

    sessions = pd.read_parquet(root / 'sessions.parquet').date
    plan = folds(sessions, index.date, CFG['first_test'], CFG['last_test'])
    pd.DataFrame([{'year': f['year'], **{f'{k}_{e}': f[k][j] for k in ['train', 'validation', 'test'] for j, e in enumerate(['start', 'end'])}}
                  for f in plan]).to_csv(out / 'folds.csv', index=False)
    years = set(a.years) if a.years else {f['year'] for f in plan}
    seeds = a.seeds if a.seeds is not None else CFG['seeds']
    returns = index.forward_return_20.to_numpy()
    pos = index.session_position.to_numpy()
    dates = index.date
    for f in plan:
        if f['year'] not in years:
            continue
        dest = out / f"Y{f['year']}"
        dest.mkdir(exist_ok=True)
        if (dest / 'COMPLETE.json').exists():
            continue
        train_ids = np.flatnonzero(dates.between(*f['train']).to_numpy() & np.isfinite(returns))
        val_ids = np.flatnonzero(dates.between(*f['validation']).to_numpy() & np.isfinite(returns))
        test_ids = np.flatnonzero(dates.between(*f['test']).to_numpy())
        assert pos[train_ids].max() + H < pos[val_ids].min() - (HISTORY - 1)
        assert pos[val_ids].max() + H < pos[test_ids].min() - (HISTORY - 1)
        if a.smoke:
            r = np.random.default_rng(f['year'])
            train_ids = np.sort(r.choice(train_ids, 20_000, replace=False))
            val_ids = np.sort(r.choice(val_ids, 5_000, replace=False))
            test_ids = np.sort(r.choice(test_ids, 5_000, replace=False))
        atomic_json(dest / 'sample_counts.json', {'train_known': len(train_ids), 'validation_known': len(val_ids),
                                                  'test_all': len(test_ids), 'test_unknown': int(np.isnan(returns[test_ids]).sum()),
                                                  'fold': {k: [str(x.date()) for x in v] for k, v in f.items() if k != 'year'}})
        for seed in seeds:
            sd = dest / f'seed_{seed}'
            sd.mkdir(exist_ok=True)
            if (sd / 'scores.npy').exists():
                continue
            lock = sd / 'RUNNING.lock'
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f'{os.uname().nodename}:{os.getpid()}'.encode())
                os.close(fd)
            except FileExistsError:
                log(f"skip Y{f['year']} seed {seed}: locked by {lock.read_text()}")
                continue
            try:
                atomic_json(out / f'status_{os.uname().nodename}_{os.getpid()}.json',
                            {'state': 'training', 'year': f['year'], 'seed': seed, 'progress': str(sd / 'progress.json'),
                             'updated_utc': pd.Timestamp.now(tz='UTC').isoformat()})
                log(f"Y{f['year']} seed {seed}: train {len(train_ids):,} val {len(val_ids):,} test {len(test_ids):,}")
                model, mean, std = fit(images, index, returns, train_ids, val_ids, sd, seed, device, CFG['batch_size'],
                                       1 if a.smoke else CFG['max_epochs'], CFG['patience'], CFG['learning_rate'])
                prob = score(model, images, test_ids, mean, std, device, 512)
                if not np.isfinite(prob).all():
                    raise ValueError('Non-finite scores')
                with open(str(sd / 'scores.npy') + '.tmp', 'wb') as fh:
                    np.save(fh, prob)
                Path(str(sd / 'scores.npy') + '.tmp').replace(sd / 'scores.npy')
                del model
                if device.type == 'mps':
                    torch.mps.empty_cache()
            except BaseException:
                atomic_json(sd / 'FAILED.json', {'error': traceback.format_exc(), 'utc': pd.Timestamp.now(tz='UTC').isoformat()})
                raise
            finally:
                lock.unlink(missing_ok=True)
        if all((dest / f'seed_{s}' / 'scores.npy').exists() for s in CFG['seeds']):
            probs = np.stack([np.load(dest / f'seed_{s}' / 'scores.npy') for s in CFG['seeds']])
            assert probs.shape[1] == len(test_ids)
            sc = index.iloc[test_ids][['row_id', 'security', 'date', 'session_position']].copy()
            sc['score'] = probs.mean(0)
            for s, pr in zip(CFG['seeds'], probs):
                sc[f'score_seed{s}'] = pr
            sc['fit_year'] = f['year']
            sc.to_parquet(dest / 'ensemble_scores.parquet', index=False)
            atomic_json(dest / 'COMPLETE.json', {'rows': len(sc), 'seeds': CFG['seeds'], 'utc': pd.Timestamp.now(tz='UTC').isoformat()})
            log(f"Y{f['year']} ensemble complete: {len(sc):,} scored samples")
    log('DONE')


if __name__ == '__main__':
    main()
