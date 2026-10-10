"""
Script: score_fixed.py
Version: 1.0 (2026-10-10)

Description:
    Train-once variant of the R3000 headline: takes the five seed models that train.py fit
    for one test year (default 2000: trained 1996-02-26 to 1998-06-16, validated to
    1999-10-08) and scores every later year with them, never retraining. This is the
    original JKX design (train once, test on everything after) and isolates how much the
    annual retraining adds. All test samples are after the source fold's validation end,
    so the scores are out of sample.

    Each image is rendered once and passed through all five models; each seed keeps its
    own training-set normalization, exactly as train.py's scorer does. At start-up the
    source year is re-scored and must reproduce the stored seed scores (max abs diff
    1e-4), which proves the models, normalization and rendering are loaded identically.
    Work unit = one test year; finished years are skipped, so the run resumes.

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/{index.parquet, state.json}
    <source run>/Y<fit year>/seed_<s>/{best.pt, normalization.json, scores.npy}, <source run>/white_pixels.npy
        (default source run /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_20261010T184018Z)
Output files (in --run-dir):
    config.json, verification.json, status.json, Y<year>/ensemble_scores.parquet (same columns as
    train.py, fit_year = the source year), Y<year>/COMPLETE.json, score.log (tee'd by caller).
    report.py runs on this directory unchanged.

Usage:
    .venv/bin/python research/r3000_headline/score_fixed.py --run-dir DIR [--source-run DIR]
        [--fit-year 2000] [--device mps] [--years 2001 ...]

Dependencies: torch, numpy, pandas; imports train.py and prepare.py helpers (unmodified).
"""
from pathlib import Path
import argparse
import hashlib
import json
import queue
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pattern.config import ModelConfig, ImageConfig  # noqa: E402
from pattern.imaging.fast import load_renderer  # noqa: E402
from pattern.models.cnn import ChartCNN  # noqa: E402
from prepare import resolve_dataset, HISTORY  # noqa: E402
from train import CFG, OnTheFlyImages, load_panel, folds, atomic_json  # noqa: E402

DEFAULT_SOURCE = ROOT / 'research/r3000_headline/runs/headline_I20R20_v2_20261010T184018Z'
BATCH = 1024
T0 = time.time()


def log(msg):
    print(f'[{time.time()-T0:8.1f}s] {msg}', flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_all(models, norms, images, ids, device):
    """(n_seeds, len(ids)) probabilities; each batch rendered once, in a prefetch thread."""
    q = queue.Queue(maxsize=4)

    def work():
        for k in range(0, len(ids), BATCH):
            q.put(images[ids[k:k + BATCH]])
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    out = np.empty((len(models), len(ids)), dtype='float32')
    k = 0
    with torch.no_grad():
        while (raw := q.get()) is not None:
            x = torch.from_numpy(raw.astype('float32'))
            for s, (m, (mean, std)) in enumerate(zip(models, norms)):
                p = torch.softmax(m(((x - mean) / std).to(device)), 1)[:, 1].cpu().numpy()
                if not np.isfinite(p).all():
                    raise RuntimeError('Non-finite probabilities')
                out[s, k:k + len(p)] = p
            k += len(raw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--source-run', default=str(DEFAULT_SOURCE))
    ap.add_argument('--fit-year', type=int, default=2000)
    ap.add_argument('--index-dir', default=str(ROOT / 'cache/r3000_I20_M20_v2_index'))
    ap.add_argument('--device', default='mps')
    ap.add_argument('--years', type=int, nargs='*')
    a = ap.parse_args()
    out, src = Path(a.run_dir).resolve(), Path(a.source_run).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fit_dir = src / f'Y{a.fit_year}'
    if not (fit_dir / 'COMPLETE.json').exists():
        raise ValueError(f'Source year {a.fit_year} is not complete in {src}')
    root, ref = resolve_dataset()
    src_cfg = json.loads((src / 'config.json').read_text())
    if src_cfg['dataset']['manifest_sha256'] != ref['manifest_sha256']:
        raise ValueError('Source run used another snapshot')
    seeds = CFG['seeds']
    cfg = {'design': f'train once on the {a.fit_year} fold, score every later year without retraining',
           'source_run': str(src), 'fit_year': a.fit_year, 'source_fold': json.loads((fit_dir / 'sample_counts.json').read_text())['fold'],
           'seeds': seeds, 'model_sha256': {s: sha(fit_dir / f'seed_{s}' / 'best.pt') for s in seeds},
           'dataset': ref, 'code_sha256': sha(Path(__file__))}
    if (out / 'config.json').exists():
        if json.loads((out / 'config.json').read_text()) != json.loads(json.dumps(cfg, default=str)):
            raise ValueError('Configuration or code changed; start a new run directory')
    else:
        atomic_json(out / 'config.json', cfg)
    device = torch.device(a.device)
    if a.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no silent CPU fallback')

    index = pd.read_parquet(Path(a.index_dir) / 'index.parquet', columns=['row_id', 'security', 'date', 'session_position'])
    log(f'index: {len(index):,} samples')
    p = load_panel(root, sorted(index.security.unique()))
    p['row'] = np.arange(len(p), dtype='int64')
    ends = index[['security', 'date']].merge(p[['security', 'date', 'row']], on=['security', 'date'], how='left', validate='one_to_one').row
    if ends.isna().any():
        raise ValueError('Sample index rows missing from panel')
    ends = ends.to_numpy('int64')
    ohlcv = np.ascontiguousarray(p[['open', 'high', 'low', 'close', 'volume']].to_numpy(float))
    tri = np.ascontiguousarray(p.total_return_index_gross_dividends.to_numpy(float))
    del p
    images = OnTheFlyImages(ohlcv, tri, ends, load_renderer(out / 'compiled'))

    models, norms = [], []
    for s in seeds:
        m = ChartCNN(ModelConfig(), ImageConfig())
        m.load_state_dict(torch.load(fit_dir / f'seed_{s}' / 'best.pt', map_location='cpu', weights_only=True))
        models.append(m.to(device).eval())
        n = json.loads((fit_dir / f'seed_{s}' / 'normalization.json').read_text())
        norms.append((n['mean'], n['std']))

    sessions = pd.read_parquet(root / 'sessions.parquet').date
    plan = {f['year']: f for f in folds(sessions, index.date, CFG['first_test'], CFG['last_test'])}
    dates = index.date

    # Verification: re-score the source year and match the stored seed scores.
    vpath = out / 'verification.json'
    if not vpath.exists():
        f = plan[a.fit_year]
        ids = np.flatnonzero(dates.between(*f['test']).to_numpy())
        got = score_all(models, norms, images, ids, device)
        diffs = {s: float(np.abs(got[i] - np.load(fit_dir / f'seed_{s}' / 'scores.npy')).max()) for i, s in enumerate(seeds)}
        ok = max(diffs.values()) < 1e-4
        atomic_json(vpath, {'rescored_year': a.fit_year, 'samples': len(ids), 'max_abs_diff_by_seed': diffs, 'passed': ok})
        if not ok:
            raise AssertionError(f'Re-scored {a.fit_year} does not match stored scores: {diffs}')
        log(f'verification passed: {a.fit_year} re-scored, max abs diff {max(diffs.values()):.2e}')

    years = sorted(set(a.years) if a.years else {y for y in plan if y >= a.fit_year})
    for y in years:
        dest = out / f'Y{y}'
        dest.mkdir(exist_ok=True)
        if (dest / 'COMPLETE.json').exists():
            continue
        atomic_json(out / 'status.json', {'state': 'scoring', 'year': y, 'updated_utc': pd.Timestamp.now(tz='UTC').isoformat()})
        ids = np.flatnonzero(dates.between(*plan[y]['test']).to_numpy())
        if y == a.fit_year:   # identical models: reuse the verified stored scores
            probs = np.stack([np.load(fit_dir / f'seed_{s}' / 'scores.npy') for s in seeds])
        else:
            probs = score_all(models, norms, images, ids, device)
        sc = index.iloc[ids][['row_id', 'security', 'date', 'session_position']].copy()
        sc['score'] = probs.mean(0)
        for s, pr in zip(seeds, probs):
            sc[f'score_seed{s}'] = pr
        sc['fit_year'] = a.fit_year
        tmp = dest / 'ensemble_scores.parquet.tmp'
        sc.to_parquet(tmp, index=False)
        tmp.replace(dest / 'ensemble_scores.parquet')
        atomic_json(dest / 'COMPLETE.json', {'rows': len(sc), 'seeds': seeds, 'fit_year': a.fit_year,
                                             'utc': pd.Timestamp.now(tz='UTC').isoformat()})
        log(f'Y{y} scored: {len(sc):,} samples')
    atomic_json(out / 'status.json', {'state': 'done', 'updated_utc': pd.Timestamp.now(tz='UTC').isoformat()})
    log('DONE')


if __name__ == '__main__':
    main()
