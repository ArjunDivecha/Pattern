"""
Script: verify_expanding.py
Version: 1.0 (2026-10-13)

Description:
    Independent checks of the expanding-window (annual retraining) result, the companion of
    verify_fixed.py for the train-once run. Works on whatever years are complete, so it can
    spot-test a run in progress. Does not reuse report.py's portfolio code.
      A. Recompute the decile long-short from the raw ensemble scores and index returns and
         match report.py's cohort returns (all names, bankrupt at last price).
      B. Alignment: re-render 600 random scored samples (spread over the complete years) with
         the reference Python renderer, score them on CPU with that year's own seed models and
         normalization, and match the stored per-seed scores. For a cloud run this also checks
         that scores trained and scored on CUDA came home attached to the right rows.
      C. Timing: every cohort lies inside its own fold's test year, at least 58 sessions after
         that fold's validation end (folds.csv), and cohort forward windows do not overlap.
      D. Placebo: shuffle scores within each date 200 times; the real long-short t-stat is
         compared with the placebo distribution.
      E. Rank IC (Spearman, score vs realised 20-session return) per cohort date, by year,
         plus score dispersion and rank correlation with the past 20-session return.
      F. A plain 20-session reversal book on the same cohorts and names, by year, and its
         correlation with the CNN book.
      G. Optional (--compare-report): year-by-year long-short against another run's
         cohort_returns.parquet (e.g. the train-once run) over the same years.

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/index.parquet
    <run dir>/folds.csv, <run dir>/Y<year>/{COMPLETE.json, ensemble_scores.parquet}
    <run dir>/Y<year>/seed_<s>/{best.pt, normalization.json}
    <report dir>/cohort_returns.parquet (from report.py on the same run)
    [--compare-report]/cohort_returns.parquet (optional)
Output files (in <run dir>/verify_<UTC timestamp>/):
    verification.json, by_year.parquet, by_year.xlsx, placebo_tstats.npy, verify.log (tee'd by caller)

Usage:
    .venv/bin/python research/r3000_headline/verify_expanding.py --run-dir DIR --report-dir DIR [--compare-report DIR]

Dependencies: numpy, pandas, torch (CPU), openpyxl.
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
from verify_fixed import nw_t, sharpe, ls_by_date, PPY  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
VARIANT = 'all, bankrupt at last price'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--report-dir', required=True)
    ap.add_argument('--compare-report')
    ap.add_argument('--index-dir', default=str(ROOT / 'cache/r3000_I20_M20_v2_index'))
    a = ap.parse_args()
    run, rep = Path(a.run_dir).resolve(), Path(a.report_dir).resolve()
    out = run / ('verify_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    root, ref = resolve_dataset()
    years = sorted(int(p.name[1:]) for p in run.glob('Y*') if (p / 'COMPLETE.json').exists())
    res = {'run_dir': str(run), 'report_dir': str(rep), 'years': years}
    print('years', years, flush=True)

    sc = pd.concat([pd.read_parquet(run / f'Y{y}' / 'ensemble_scores.parquet') for y in years])
    idx = pd.read_parquet(Path(a.index_dir) / 'index.parquet',
                          columns=['row_id', 'security', 'date', 'session_position', 'forward_return_20', 'terminal_within_20', 'return_to_last'])
    assert sc.row_id.is_unique
    m = sc.merge(idx, on='row_id', validate='one_to_one', suffixes=('', '_i'))
    assert len(m) == len(sc)
    assert (m.security == m.security_i).all() and (m.date == m.date_i).all() and (m.session_position == m.session_position_i).all()
    assert np.allclose(m.score, m[[f'score_seed{s}' for s in SEEDS]].mean(axis=1), atol=1e-6), 'ensemble is not the seed mean'
    known = m.forward_return_20.notna()
    term = m.terminal_within_20 & m.return_to_last.notna() & ~known
    m['ret'] = np.where(known, m.forward_return_20, np.where(term, m.return_to_last, np.nan))
    first = m.session_position.min()
    coh = m[(m.session_position - first) % H == 0].copy()

    # A. independent recompute
    mine = ls_by_date(coh, 'score')
    theirs = pd.read_parquet(rep / 'cohort_returns.parquet')
    theirs = theirs[theirs.variant == VARIANT].set_index('date').long_short.dropna()
    j = pd.concat([mine.rename('mine'), theirs.rename('report')], axis=1).dropna()
    res['A_recompute'] = {'cohorts': len(j), 'max_abs_diff': float((j.mine - j.report).abs().max()),
                          'sharpe_mine': float(sharpe(mine)), 'nw_t_mine': float(nw_t(mine)),
                          'passed': bool(len(j) == len(theirs) and (j.mine - j.report).abs().max() < 1e-10)}
    print('A', res['A_recompute'], flush=True)

    # B. alignment with each year's own models
    pick = m.sample(600, random_state=7)
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
    pyear = pick.date.dt.year.to_numpy()
    diffs = {}
    for s in SEEDS:
        pr = np.full(len(pick), np.nan)
        for y in np.unique(pyear):
            sel = pyear == y
            sd = run / f'Y{y}' / f'seed_{s}'
            model = ChartCNN(ModelConfig(), ImageConfig())
            model.load_state_dict(torch.load(sd / 'best.pt', map_location='cpu', weights_only=True))
            model.eval()
            n = json.loads((sd / 'normalization.json').read_text())
            with torch.no_grad():
                x = torch.from_numpy((imgs[sel].astype('float32') - n['mean']) / n['std'])
                pr[sel] = torch.softmax(model(x), 1)[:, 1].numpy()
        st = pick[f'score_seed{s}'].to_numpy()
        diffs[s] = {'max_abs_diff': float(np.abs(pr - st).max()), 'share_within_1e-5': float((np.abs(pr - st) < 1e-5).mean()),
                    'share_within_1e-4': float((np.abs(pr - st) < 1e-4).mean()),
                    'corr': float(np.corrcoef(pr, st)[0, 1]), 'corr_if_shifted_one_row': float(np.corrcoef(pr, np.roll(st, 1))[0, 1])}
    # Scores were computed on CUDA; CPU re-scoring differs in float rounding, and ~0.5% of images differ by a
    # few pixels (compiled vs Python renderer tie-breaking). Alignment is judged on match share and correlation.
    res['B_alignment'] = {'samples': 600, 'years': sorted(set(int(y) for y in pyear)), 'by_seed': diffs,
                          'passed': bool(all(d['share_within_1e-4'] > 0.97 and d['corr'] > 0.999 for d in diffs.values()))}
    print('B', res['B_alignment'], flush=True)

    # C. timing against each fold
    folds = pd.read_csv(run / 'folds.csv', parse_dates=['validation_end', 'test_start', 'test_end']).set_index('year')
    sessions = pd.read_parquet(root / 'sessions.parquet').date.to_numpy()
    cyear = coh.date.dt.year
    vpos = cyear.map(lambda y: int(np.searchsorted(sessions, folds.loc[y, 'validation_end'].to_datetime64()))).to_numpy()
    inside = ((coh.date >= cyear.map(folds.test_start)) & (coh.date <= cyear.map(folds.test_end))).all()
    gap = (coh.session_position.to_numpy() - vpos).min()
    cpos = np.sort(coh.session_position.unique())
    res['C_timing'] = {'min_sessions_after_validation_end': int(gap), 'cohorts_inside_own_test_year': bool(inside),
                       'min_cohort_spacing': int(np.diff(cpos).min()),
                       'passed': bool(gap >= 58 and inside and np.diff(cpos).min() >= H)}
    print('C', res['C_timing'], flush=True)

    # D. placebo
    rng = np.random.default_rng(7)
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

    # E/F. IC, dispersion, reversal
    past = idx[['security', 'session_position', 'forward_return_20']].rename(columns={'forward_return_20': 'past20'})
    past['session_position'] = past.session_position + H
    coh = coh.merge(past, on=['security', 'session_position'], how='left')
    coh['rev'] = -coh.past20
    coh['year'] = coh.date.dt.year
    ic = coh.dropna(subset=['ret']).groupby('date').apply(lambda d: d.score.rank().corr(d.ret.rank()), include_groups=False)
    rev = ls_by_date(coh.dropna(subset=['rev']), 'rev')
    rows = []
    for y, g in coh.groupby('year'):
        f = lambda s: s[s.index.year == y]
        rows.append({'year': y, 'cohorts': len(f(mine)), 'cnn_ls_ann': f(mine).mean() * PPY, 'rank_ic_mean': f(ic).mean(),
                     'rank_ic_hit': (f(ic) > 0).mean(), 'reversal_ls_ann': f(rev).mean() * PPY,
                     'score_sd': g.groupby('date').score.std().mean(),
                     'rank_corr_score_past20': g.groupby('date').apply(lambda d: d.score.rank().corr(d.past20.rank()), include_groups=False).mean()})
    by_year = pd.DataFrame(rows)
    res['E_ic'] = {'rank_ic_mean': float(ic.mean()), 'rank_ic_nw_t': float(nw_t(ic)), 'rank_ic_hit': float((ic > 0).mean()), 'cohorts': len(ic)}
    res['F_reversal'] = {'reversal_ann': float(rev.mean() * PPY), 'reversal_sharpe': float(sharpe(rev)),
                         'corr_cnn_reversal': float(mine.corr(rev)), 'rank_corr_score_past20_mean': float(by_year.rank_corr_score_past20.mean())}
    print('E', res['E_ic'], flush=True)
    print('F', res['F_reversal'], flush=True)

    # G. comparison run
    if a.compare_report:
        o = pd.read_parquet(Path(a.compare_report) / 'cohort_returns.parquet')
        o = o[o.variant == VARIANT].set_index('date').long_short.dropna()
        o = o[(o.index.year >= min(years)) & (o.index.year <= max(years))]
        by_year['compare_ls_ann'] = by_year.year.map(o.groupby(o.index.year).mean() * PPY)
        res['G_compare'] = {'compare_report': str(Path(a.compare_report).resolve()), 'this_sharpe': float(sharpe(mine)),
                            'compare_sharpe': float(sharpe(o)), 'this_ann': float(mine.mean() * PPY), 'compare_ann': float(o.mean() * PPY),
                            'years_this_better': int((by_year.cnn_ls_ann > by_year.compare_ls_ann).sum()), 'years': len(by_year)}
        print('G', res['G_compare'], flush=True)

    by_year.to_parquet(out / 'by_year.parquet', index=False)
    with pd.ExcelWriter(out / 'by_year.xlsx') as xw:
        by_year.to_excel(xw, sheet_name='by_year', index=False)
    print(by_year.round(4).to_string())
    (out / 'verification.json').write_text(json.dumps(res, indent=2, default=str))
    print('OUTPUT', out)


if __name__ == '__main__':
    main()
