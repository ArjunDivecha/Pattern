"""
Script: score_all_db.py
Version: 1.1 (2026-10-16)

Description:
    Extends the R3000 headline's out-of-sample CNN scores from Russell 3000 member days to
    every stock-day in the frozen research database that can be charted, 2000-01-03 to
    2026-06-30, so the full history can be published to the Global Data Mart's alpha
    library.

    The headline run (train.py on the cloud) scored only sessions on which a name was a
    point-in-time Russell 3000 member (prepare.py's membership rule). This script applies
    prepare.py's chart-validity rule unchanged, minus the membership test: a (security,
    date t) is scored when the 39 sessions ending at t are consecutive NYSE sessions, every
    bar passes bar_ok_on_calendar, and all lie in one calendar_segment. Each stock-day is
    scored by the five seed models of ITS test year (the fold trained and validated only on
    data ending at least 58 sessions before 1 January of that year), so every score is out
    of sample in time. 1996-1999 are not scored: no fold exists whose training ended before
    them.

    Member rows reuse the stored cloud ensemble scores. Non-member rows are scored here on
    MPS with the same renderer, models and per-seed training normalization. Proof that the
    two agree: in every year a random sample of member rows is re-scored locally and must
    match the stored scores: every seed within 5e-4, the 5-seed mean within 1e-4, and rank
    correlation of the mean at least 0.99999. (The cloud A100s ran convolutions at PyTorch's
    default TF32 precision, so seeds differ from MPS float32 by up to ~1.7e-4; a real fault
    -- wrong model, normalization or image -- moves scores by ~1e-2 and scrambles ranks.)
    The run also asserts that every stored member row is present in the full index. Work unit = one year, written
    atomically; finished years are skipped, so the run resumes.

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/pattern_daily_v2_20261010/
        prices.parquet, quality.parquet, calendar_quality.parquet, membership_pit.parquet, manifest.json
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z/
        config.json, folds.csv, Y<year>/ensemble_scores.parquet, Y<year>/seed_<s>/{best.pt, normalization.json}

Output files (in --out, default <run>/full_db_scores_<UTC timestamp>/):
    config.json       inputs, hashes, rules
    Y<year>.parquet   security, date, session_position, fit_year, r3000_member, scored_by
                      ('cloud_run' | 'local_mps'), score (5-seed mean probability of a positive
                      20-session return), score_seed0..4
    verification.json per-year re-score check of member rows
    summary.json      rows by year, member / non-member split, names
    status.json       heartbeat
    score_all_db.log  (tee'd by the caller)

Usage:
    .venv/bin/python research/r3000_headline/score_all_db.py [--out DIR] [--years 2000 ...]
        [--limit N] [--device mps] [--check-rows 20000]
    --limit N keeps the first N securities (alphabetical) for a smoke test.

Dependencies: torch, numpy, pandas; imports prepare.py (sample_index, resolve_dataset) and
the pattern package (ChartCNN, fast renderer), all unmodified.
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
from prepare import resolve_dataset, sample_index, START, WINDOW, MA  # noqa: E402

RUN = ROOT / 'research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z'
SEEDS = [0, 1, 2, 3, 4]
BATCH = 2048
SEED_TOL, ENS_TOL, RANK_MIN = 5e-4, 1e-4, 0.99999
T0 = time.time()


def log(msg):
    print(f'[{time.time()-T0:8.1f}s] {msg}', flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def score(models, norms, render, ohlcv, tri, ends, device):
    """(n_seeds, len(ends)) probabilities; each batch rendered once, in a prefetch thread."""
    q = queue.Queue(maxsize=4)

    def work():
        for k in range(0, len(ends), BATCH):
            q.put(render(ohlcv, tri, ends[k:k + BATCH], WINDOW, MA))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    out = np.empty((len(models), len(ends)), dtype='float32')
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
    ap.add_argument('--run', default=str(RUN))
    ap.add_argument('--out', default=None)
    ap.add_argument('--years', type=int, nargs='*')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--check-rows', type=int, default=20000)
    a = ap.parse_args()
    run = Path(a.run).resolve()
    out = Path(a.out).resolve() if a.out else run / f"full_db_scores_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True, exist_ok=True)
    root, ref = resolve_dataset()
    run_cfg = json.loads((run / 'config.json').read_text())
    if run_cfg['dataset']['manifest_sha256'] != ref['manifest_sha256']:
        raise ValueError('Run used another snapshot')
    folds = pd.read_csv(run / 'folds.csv', parse_dates=['test_start', 'test_end'])
    years = sorted(set(a.years) if a.years else set(folds.year))
    cfg = {'run': str(run), 'dataset': ref, 'limit': a.limit, 'check_rows': a.check_rows,
           'tolerance': {'seed_max_abs': SEED_TOL, 'ensemble_max_abs': ENS_TOL, 'rank_corr_min': RANK_MIN},
           'rule': 'prepare.sample_index validity rule with every month treated as a member month; '
                   'each stock-day scored by its own test-year fold',
           'code_sha256': sha(Path(__file__)), 'prepare_sha256': sha(Path(__file__).parent / 'prepare.py'),
           'renderer_sha256': sha(ROOT / 'pattern/imaging/fast_renderer.c')}
    if (out / 'config.json').exists():
        if json.loads((out / 'config.json').read_text()) != json.loads(json.dumps(cfg, default=str)):
            raise ValueError('Configuration or code changed; use a new --out directory')
    else:
        atomic_json(out / 'config.json', cfg)
    device = torch.device(a.device)
    if a.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no silent CPU fallback')

    # Full panel, exactly as prepare.py loads it, for every priced security.
    mem = pd.read_parquet(root / 'membership_pit.parquet', columns=['date'])
    month_ends = np.sort(mem.date.unique()).astype('datetime64[ns]')
    all_months = np.arange(len(month_ends))
    last_vintage_date = pd.Timestamp(json.loads((root / 'manifest.json').read_text())['last_date'])
    filters = [('date', '>=', START)]
    if a.limit:
        names = sorted(pd.read_parquet(root / 'prices.parquet', columns=['security']).security.astype(str).unique())[:a.limit]
        filters.append(('security', 'in', names))
    p = pd.read_parquet(root / 'prices.parquet', filters=filters,
                        columns=['security', 'date', 'open', 'high', 'low', 'close', 'volume', 'total_return_index_gross_dividends'])
    for name, cols in [('quality.parquet', ['security', 'date', 'bar_ok_on_calendar', 'return_endpoint_ok', 'off_exchange_calendar']),
                       ('calendar_quality.parquet', ['security', 'date', 'session_index', 'calendar_segment'])]:
        p = p.merge(pd.read_parquet(root / name, columns=cols, filters=filters), on=['security', 'date'], validate='one_to_one')
    p = p[~p.off_exchange_calendar].sort_values(['security', 'date']).reset_index(drop=True)
    p['security'] = p.security.astype(str)
    log(f'panel: {len(p):,} on-calendar rows, {p.security.nunique():,} securities')

    parts, offset = [], 0
    for security, g in p.groupby('security', sort=True):
        g = g.reset_index(drop=True)
        res = sample_index(g, all_months, month_ends, last_vintage_date)
        if res is not None:
            idx, ends = res
            parts.append(pd.DataFrame({'security': security, 'date': idx.date.to_numpy(),
                                       'session_position': idx.session_position.to_numpy(), 'row': ends + offset}))
        offset += len(g)
    index = pd.concat(parts, ignore_index=True)
    index = index[index.date >= folds.test_start.min()].reset_index(drop=True)
    log(f'chartable stock-days from {folds.test_start.min().date()}: {len(index):,} over {index.security.nunique():,} securities')
    ohlcv = np.ascontiguousarray(p[['open', 'high', 'low', 'close', 'volume']].to_numpy(float))
    tri = np.ascontiguousarray(p.total_return_index_gross_dividends.to_numpy(float))
    del p
    render = load_renderer(out / 'compiled')

    vpath = out / 'verification.json'
    verification = json.loads(vpath.read_text()) if vpath.exists() else {}
    for y in years:
        dest = out / f'Y{y}.parquet'
        if dest.exists():
            continue
        f = folds.loc[folds.year == y].iloc[0]
        atomic_json(out / 'status.json', {'state': 'scoring', 'year': y, 'updated_utc': pd.Timestamp.now(tz='UTC').isoformat()})
        yi = index[index.date.between(f.test_start, f.test_end)].reset_index(drop=True)
        stored = pd.read_parquet(run / f'Y{y}' / 'ensemble_scores.parquet',
                                 columns=['security', 'date', 'score'] + [f'score_seed{s}' for s in SEEDS])
        stored['security'] = stored.security.astype(str)
        if a.limit:
            stored = stored[stored.security.isin(set(yi.security))]
        yi = yi.merge(stored, on=['security', 'date'], how='left', validate='one_to_one', indicator=True)
        missing = len(stored) - int((yi._merge == 'both').sum())
        if missing:
            raise AssertionError(f'Y{y}: {missing} stored member rows are not in the full index')
        member = (yi._merge == 'both').to_numpy()
        yi = yi.drop(columns='_merge')

        models, norms = [], []
        for s in SEEDS:
            sd = run / f'Y{y}' / f'seed_{s}'
            m = ChartCNN(ModelConfig(), ImageConfig())
            m.load_state_dict(torch.load(sd / 'best.pt', map_location='cpu', weights_only=True))
            models.append(m.to(device).eval())
            n = json.loads((sd / 'normalization.json').read_text())
            norms.append((n['mean'], n['std']))

        # Check: re-score a random sample of member rows and match the stored cloud scores.
        mpos = np.flatnonzero(member)
        if not len(mpos):
            raise AssertionError(f'Y{y}: no member rows to verify against (raise --limit)')
        pick = np.sort(np.random.default_rng(y).choice(mpos, min(a.check_rows, len(mpos)), replace=False))
        got = score(models, norms, render, ohlcv, tri, yi.row.to_numpy('int64')[pick], device)
        diffs = {s: float(np.abs(got[i] - yi[f'score_seed{s}'].to_numpy()[pick]).max()) for i, s in enumerate(SEEDS)}
        ens_stored, ens_local = yi.score.to_numpy()[pick], got.mean(0)
        ens_diff = np.abs(ens_local - ens_stored)
        rank_corr = float(np.corrcoef(pd.Series(ens_stored).rank(), pd.Series(ens_local).rank())[0, 1])
        ok = max(diffs.values()) < SEED_TOL and ens_diff.max() < ENS_TOL and rank_corr >= RANK_MIN
        verification[str(y)] = {'rows_checked': len(pick), 'max_abs_diff_by_seed': diffs,
                                'ensemble_max_abs_diff': float(ens_diff.max()), 'ensemble_mean_abs_diff': float(ens_diff.mean()),
                                'ensemble_rank_corr': rank_corr, 'passed': ok}
        atomic_json(vpath, verification)
        if not ok:
            raise AssertionError(f'Y{y}: local re-score does not match stored scores: {verification[str(y)]}')

        new = np.flatnonzero(~member)
        t = time.time()
        probs = score(models, norms, render, ohlcv, tri, yi.row.to_numpy('int64')[new], device)
        for i, s in enumerate(SEEDS):
            col = yi[f'score_seed{s}'].to_numpy('float32').copy()
            col[new] = probs[i]
            yi[f'score_seed{s}'] = col
        sc = yi[[f'score_seed{s}' for s in SEEDS]].to_numpy('float32')
        yi['score'] = np.where(member, yi.score.to_numpy('float32'), sc.mean(1)).astype('float32')
        yi['fit_year'] = y
        yi['r3000_member'] = member
        yi['scored_by'] = np.where(member, 'cloud_run', 'local_mps')
        yi = yi[['security', 'date', 'session_position', 'fit_year', 'r3000_member', 'scored_by', 'score']
                + [f'score_seed{s}' for s in SEEDS]]
        if not np.isfinite(yi[['score'] + [f'score_seed{s}' for s in SEEDS]].to_numpy()).all():
            raise AssertionError(f'Y{y}: non-finite scores')
        tmp = Path(str(dest) + '.tmp')
        yi.to_parquet(tmp, index=False)
        tmp.replace(dest)
        log(f'Y{y}: {len(yi):,} rows ({member.sum():,} member reused, {len(new):,} scored at '
            f'{len(new)/max(time.time()-t, 1e-9):,.0f}/s), check seed max {max(diffs.values()):.1e}, mean max {ens_diff.max():.1e}, rank corr {rank_corr:.7f}')

    done = sorted(out.glob('Y*.parquet'))
    if len(done) == len(set(folds.year)) or a.years:
        rows = []
        for fp in done:
            d = pd.read_parquet(fp, columns=['security', 'r3000_member'])
            rows.append({'year': int(fp.stem[1:]), 'rows': len(d), 'member_rows': int(d.r3000_member.sum()),
                         'non_member_rows': int((~d.r3000_member).sum()), 'securities': int(d.security.nunique())})
        by_year = pd.DataFrame(rows)
        atomic_json(out / 'summary.json', {'rows': int(by_year.rows.sum()), 'member_rows': int(by_year.member_rows.sum()),
                                           'non_member_rows': int(by_year.non_member_rows.sum()),
                                           'by_year': rows, 'verification_all_passed': all(v['passed'] for v in verification.values()),
                                           'elapsed_seconds': time.time() - T0})
        print(by_year.to_string(index=False), flush=True)
    atomic_json(out / 'status.json', {'state': 'done', 'updated_utc': pd.Timestamp.now(tz='UTC').isoformat()})
    log('DONE')


if __name__ == '__main__':
    main()
