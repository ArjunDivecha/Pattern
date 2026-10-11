"""
Script: verify_fixed.py
Version: 1.1 (2026-10-10)

Description:
    Independent checks of the train-once result (score_fixed.py + report.py), written
    without reusing report.py's portfolio code:
      A. Recompute the decile long-short from the raw scores and index returns and match
         report.py's cohort returns (all names, bankrupt at last price).
      B. Alignment: re-render 600 random scored samples from 2001-2026 with the reference
         Python renderer, score them on CPU with each seed's model and normalization, and
         match the stored per-seed scores. Catches any row/score misalignment.
      C. Timing: every cohort date is after the source fold's validation end plus the
         58-session gap; cohort forward windows do not overlap.
      D. Placebo: shuffle scores within each date 200 times; the real long-short t-stat
         is compared with the placebo distribution.
      E. Drift and what the model has learned: per-year score dispersion, and rank
         correlation of the score with the past 20-session return (short-term reversal).
      F. A plain 20-session reversal book on the same cohorts and names, by five-year block,
         to compare its decay with the CNN's.

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/index.parquet
    <run dir>/config.json, <run dir>/Y<year>/ensemble_scores.parquet, <run dir>/<report dir>/cohort_returns.parquet
    <source run>/Y<fit year>/seed_<s>/{best.pt, normalization.json}
Output files (in <run dir>/verify_<UTC timestamp>/):
    verification.json, by_year.parquet, by_block.parquet, by_block.xlsx, placebo_tstats.npy

Usage:
    .venv/bin/python research/r3000_headline/verify_fixed.py --run-dir DIR --report-dir DIR

Dependencies: numpy, pandas, torch (CPU), scipy-free.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pattern.config import ModelConfig, ImageConfig  # noqa: E402
from pattern.imaging.renderer import render_window  # noqa: E402
from pattern.models.cnn import ChartCNN  # noqa: E402
from prepare import resolve_dataset, WINDOW, MA, H, HISTORY  # noqa: E402
from train import load_panel  # noqa: E402

PPY = 252 / H


def nw_t(x, lags=2):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n, m = len(x), x.mean()
    e = x - m
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return m / np.sqrt(v / n)


def sharpe(x):
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(PPY)


def ls_by_date(d, col):
    """Top-minus-bottom decile by date; deciles over all names, returns over known ones."""
    out = {}
    for date, g in d.groupby('date', sort=True):
        g = g.sort_values([col, 'security'])
        dec = np.arange(len(g)) * 10 // len(g) + 1
        r = g.ret.to_numpy()
        top, bot = r[dec == 10], r[dec == 1]
        if np.isfinite(top).any() and np.isfinite(bot).any():
            out[date] = np.nanmean(top) - np.nanmean(bot)
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--report-dir', required=True)
    ap.add_argument('--index-dir', default=str(ROOT / 'cache/r3000_I20_M20_v2_index'))
    a = ap.parse_args()
    run, rep = Path(a.run_dir).resolve(), Path(a.report_dir).resolve()
    out = run / ('verify_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    cfg = json.loads((run / 'config.json').read_text())
    root, ref = resolve_dataset()
    res = {'run_dir': str(run), 'report_dir': str(rep)}

    years = sorted(int(p.name[1:]) for p in run.glob('Y*') if (p / 'COMPLETE.json').exists())
    sc = pd.concat([pd.read_parquet(run / f'Y{y}' / 'ensemble_scores.parquet') for y in years])
    idx = pd.read_parquet(Path(a.index_dir) / 'index.parquet',
                          columns=['row_id', 'security', 'date', 'session_position', 'forward_return_20', 'terminal_within_20', 'return_to_last'])
    assert sc.row_id.is_unique
    m = sc.merge(idx, on='row_id', validate='one_to_one', suffixes=('', '_i'))
    assert (m.security == m.security_i).all() and (m.date == m.date_i).all() and (m.session_position == m.session_position_i).all()
    known = m.forward_return_20.notna()
    term = m.terminal_within_20 & m.return_to_last.notna() & ~known
    m['ret'] = np.where(known, m.forward_return_20, np.where(term, m.return_to_last, np.nan))
    first = m.session_position.min()
    coh = m[(m.session_position - first) % H == 0].copy()

    # A. independent recompute
    mine = ls_by_date(coh, 'score')
    theirs = pd.read_parquet(rep / 'cohort_returns.parquet')
    theirs = theirs[theirs.variant == 'all, bankrupt at last price'].set_index('date').long_short.dropna()  # last cohort has no forward returns yet
    j = pd.concat([mine.rename('mine'), theirs.rename('report')], axis=1).dropna()
    res['A_recompute'] = {'cohorts': len(j), 'max_abs_diff': float((j.mine - j.report).abs().max()),
                          'sharpe_mine': float(sharpe(mine)), 'nw_t_mine': float(nw_t(mine)),
                          'passed': bool(len(j) == len(theirs) and (j.mine - j.report).abs().max() < 1e-10)}
    print('A', res['A_recompute'], flush=True)

    # B. alignment via reference renderer on CPU
    rng = np.random.default_rng(7)
    pick = m[m.date.dt.year >= 2001].sample(600, random_state=7)
    p = load_panel(root, sorted(pick.security.unique()))
    p['row'] = np.arange(len(p))
    e = pick[['security', 'date']].merge(p[['security', 'date', 'row']], on=['security', 'date'], how='left').row.to_numpy()
    assert not np.isnan(e).any()
    e = e.astype(int)
    ohlcv = p[['open', 'high', 'low', 'close', 'volume']].to_numpy(float)
    tri = p.total_return_index_gross_dividends.to_numpy(float)
    secs = p.security.to_numpy()
    assert all(secs[k] == secs[k - HISTORY + 1] for k in e)
    imgs = np.stack([render_window(ohlcv[k - HISTORY + 1:k + 1], tri[k - HISTORY + 1:k + 1], window=WINDOW, width=3 * WINDOW, ma_length=MA) for k in e])
    fit_dir = Path(cfg['source_run']) / f"Y{cfg['fit_year']}"
    diffs = {}
    for s in cfg['seeds']:
        model = ChartCNN(ModelConfig(), ImageConfig())
        model.load_state_dict(torch.load(fit_dir / f'seed_{s}' / 'best.pt', map_location='cpu', weights_only=True))
        model.eval()
        n = json.loads((fit_dir / f'seed_{s}' / 'normalization.json').read_text())
        with torch.no_grad():
            pr = torch.softmax(model(torch.from_numpy((imgs.astype('float32') - n['mean']) / n['std'])), 1)[:, 1].numpy()
        st = pick[f'score_seed{s}'].to_numpy()
        diffs[s] = {'max_abs_diff': float(np.abs(pr - st).max()), 'share_within_1e-5': float((np.abs(pr - st) < 1e-5).mean()),
                    'corr': float(np.corrcoef(pr, st)[0, 1]), 'corr_if_shifted_one_row': float(np.corrcoef(pr, np.roll(st, 1))[0, 1])}
    # The compiled renderer breaks exact .5 pixel ties away from zero, Python's round() to even, so
    # ~0.5% of images differ by a few pixels (identically in training and scoring). Alignment is
    # judged on the share of exact matches and the correlation, not the worst sample.
    res['B_alignment'] = {'samples': 600, 'years': [int(pick.date.dt.year.min()), int(pick.date.dt.year.max())], 'by_seed': diffs,
                          'passed': bool(all(d['share_within_1e-5'] > 0.97 and d['corr'] > 0.999 for d in diffs.values()))}
    print('B', res['B_alignment'], flush=True)

    # C. timing
    sessions = pd.read_parquet(root / 'sessions.parquet').date
    val_end = pd.Timestamp(cfg['source_fold']['validation'][1])
    vpos = int(np.searchsorted(sessions.to_numpy(), val_end.to_datetime64()))
    cpos = np.sort(coh.session_position.unique())
    res['C_timing'] = {'source_validation_end': str(val_end.date()), 'first_cohort': str(coh.date.min().date()),
                       'sessions_between': int(cpos.min() - vpos), 'min_cohort_spacing': int(np.diff(cpos).min()),
                       'passed': bool(cpos.min() - vpos >= 58 and np.diff(cpos).min() >= H)}
    print('C', res['C_timing'], flush=True)

    # D. placebo
    real_t = nw_t(mine)
    pl = []
    c2 = coh[['date', 'security', 'ret', 'score']].copy()
    for b in range(200):
        c2['shuf'] = c2.groupby('date').score.transform(lambda s: rng.permutation(s.to_numpy()))
        pl.append(nw_t(ls_by_date(c2, 'shuf')))
    pl = np.array(pl)
    np.save(out / 'placebo_tstats.npy', pl)
    res['D_placebo'] = {'real_t': float(real_t), 'placebo_mean': float(pl.mean()), 'placebo_sd': float(pl.std()),
                        'placebo_max': float(pl.max()), 'share_placebo_ge_real': float((pl >= real_t).mean())}
    print('D', res['D_placebo'], flush=True)

    # E/F. reversal comparison
    past = idx[['security', 'session_position', 'forward_return_20']].rename(columns={'forward_return_20': 'past20'})
    past['session_position'] = past.session_position + H
    coh = coh.merge(past, on=['security', 'session_position'], how='left')
    coh['rev'] = -coh.past20
    coh['year'] = coh.date.dt.year
    by_year = []
    for y, g in coh.groupby('year'):
        rc = g.groupby('date').apply(lambda d: d.score.rank().corr(d.past20.rank()), include_groups=False)
        by_year.append({'year': y, 'score_sd': g.groupby('date').score.std().mean(), 'score_mean': g.score.mean(),
                        'rank_corr_score_past20': rc.mean(), 'past20_coverage': g.past20.notna().mean()})
    by_year = pd.DataFrame(by_year)
    rev = ls_by_date(coh.dropna(subset=['rev']), 'rev')
    cnn = mine
    blocks = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2026)]
    rows = []
    for lo, hi in blocks:
        f = lambda s: s[(s.index.year >= lo) & (s.index.year <= hi)]
        rows.append({'block': f'{lo}-{hi}', 'cnn_ann': f(cnn).mean() * PPY, 'cnn_sharpe': sharpe(f(cnn)),
                     'reversal_ann': f(rev).mean() * PPY, 'reversal_sharpe': sharpe(f(rev)),
                     'corr_cnn_reversal': f(cnn).corr(f(rev))})
    rows.append({'block': 'Full', 'cnn_ann': cnn.mean() * PPY, 'cnn_sharpe': sharpe(cnn), 'reversal_ann': rev.mean() * PPY,
                 'reversal_sharpe': sharpe(rev), 'corr_cnn_reversal': cnn.corr(rev)})
    by_block = pd.DataFrame(rows)
    by_year.to_parquet(out / 'by_year.parquet', index=False)
    by_block.to_parquet(out / 'by_block.parquet', index=False)
    with pd.ExcelWriter(out / 'by_block.xlsx') as xw:
        by_block.to_excel(xw, sheet_name='by_block', index=False)
        by_year.to_excel(xw, sheet_name='by_year', index=False)
    res['E_F'] = {'by_block': by_block.round(4).to_dict('records'),
                  'rank_corr_score_past20_mean': float(by_year.rank_corr_score_past20.mean())}
    print(by_year.round(3).to_string())
    print(by_block.round(3).to_string())
    (out / 'verification.json').write_text(json.dumps(res, indent=2, default=str))
    print('OUTPUT', out)


if __name__ == '__main__':
    main()
