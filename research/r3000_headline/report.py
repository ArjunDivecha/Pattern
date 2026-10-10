"""
Script: report.py
Version: 1.0 (2026-10-10)

Description:
    Portfolio report for the R3000 headline rerun. Reads the ensemble scores written by
    train.py and builds gross, equal-weight decile portfolios on non-overlapping
    20-session cohorts (a cohort starts every 20th session from the first test session;
    each holds for 20 sessions). Long-short = top decile minus bottom decile.
    No transaction costs or turnover penalties.

    Corrections relative to the original R1000 headline:
      - simple (arithmetic) TRI returns, never mean log returns;
      - point-in-time Russell 3000 membership, survivorship-certified from 2000;
      - delisted holdings keep a return: delisting inside the horizon earns the return to
        the last listed price; bankruptcies are shown two ways, at the last price and
        at -100% (total loss), so the terminal assumption is visible;
      - price floor variants: none, $5 on the traded (nominal) close at formation, and
        $5 on the split-adjusted close (shown only to measure the look-ahead it causes);
      - size slices: large = top 1000 eligible names by market cap at formation (R1000
        proxy), small = the rest (R2000 proxy). Cap = latest shares outstanding filed on
        or before the formation date x traded close.
    Samples whose outcome is unknown for other reasons (a missing bar in the horizon)
    are dropped from that cohort and counted in the output.

Input files:
    <run dir>/Y<year>/ensemble_scores.parquet, <run dir>/config.json  (from train.py)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/index.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/pattern_daily_v2_20261010/
        nominal_prices.parquet, delisting_outcomes.parquet, sessions.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/fundamentals.parquet
        (BS_SH_OUT only; its SHA256 is recorded in the summary)
Output files (in <run dir>/report_<UTC timestamp>/):
    cohort_returns.parquet   per cohort x variant: decile returns, LS, EW benchmark, counts
    metrics.parquet/.xlsx    per variant x period (Full/5y/3y/1y): annual return, vol,
                             Sharpe, CAGR, t-stat, hit rate for LS, top, bottom, benchmark
    deciles.parquet/.xlsx    mean cohort return by decile per variant
    charts.pdf               cumulative LS and decile curves (light mode)
    report.html              memo-style summary with embedded charts
    summary.json, report.log (tee'd by caller)

Usage:
    .venv/bin/python research/r3000_headline/report.py --run-dir DIR [--allow-partial] [--min-names 100]
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import io
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import resolve_dataset, H  # noqa: E402

FUND = ROOT.parent / 'Global Data Mart/sources/equities/data/vintages/2026-06-30/fundamentals.parquet'
PERIODS_PER_YEAR = 252 / H


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stats(x):
    x = pd.Series(x).dropna()
    if len(x) < 3:
        return dict(cohorts=len(x), ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan, cagr=np.nan, t_stat=np.nan, hit_rate=np.nan)
    m, s = x.mean(), x.std(ddof=1)
    # Non-overlapping cohorts; Newey-West with 2 lags absorbs residual autocorrelation.
    r = (x - m).to_numpy()
    v = (r ** 2).sum() + sum(2 * (1 - L / 3) * (r[L:] * r[:-L]).sum() for L in (1, 2))
    t = m / (np.sqrt(v) / len(x)) if v > 0 else np.nan
    growth = np.prod(1 + x.to_numpy())
    return dict(cohorts=len(x), ann_return=m * PERIODS_PER_YEAR, ann_vol=s * np.sqrt(PERIODS_PER_YEAR),
                sharpe=m / s * np.sqrt(PERIODS_PER_YEAR) if s > 0 else np.nan,
                cagr=growth ** (PERIODS_PER_YEAR / len(x)) - 1 if growth > 0 else -1.0, t_stat=t, hit_rate=float((x > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--index-dir', default=str(ROOT / 'cache/r3000_I20_M20_v2_index'))
    ap.add_argument('--allow-partial', action='store_true')
    ap.add_argument('--min-names', type=int, default=100, help='skip cohort dates with fewer eligible names')
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    root, ref = resolve_dataset()
    years = sorted(int(p.name[1:]) for p in run.glob('Y*') if (p / 'COMPLETE.json').exists())
    expected = list(range(2000, 2027))
    if not years:
        raise SystemExit('No completed test years yet')
    if years != expected and not a.allow_partial:
        raise SystemExit(f'Only {len(years)} of {len(expected)} years complete; pass --allow-partial for an interim report')
    out = run / ('report_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    sc = pd.concat([pd.read_parquet(run / f'Y{y}' / 'ensemble_scores.parquet', columns=['row_id', 'security', 'date', 'session_position', 'score']) for y in years])
    idx = pd.read_parquet(Path(a.index_dir) / 'index.parquet', columns=['row_id', 'forward_return_20', 'terminal_within_20', 'return_to_last'])
    sc = sc.merge(idx, on='row_id', validate='one_to_one')
    filt = [('security', 'in', sorted(sc.security.unique())), ('date', '>=', sc.date.min()), ('date', '<=', sc.date.max())]
    nom = pd.read_parquet(root / 'nominal_prices.parquet', columns=['security', 'date', 'close_nominal', 'close_split_adj'], filters=filt)
    nom['security'] = nom.security.astype(str)
    sc = sc.merge(nom, on=['security', 'date'], how='left', validate='one_to_one')
    dl = pd.read_parquet(root / 'delisting_outcomes.parquet', columns=['security', 'outcome'])
    dl['security'] = dl.security.astype(str)
    sc = sc.merge(dl, on='security', how='left')

    # Cohort dates: every H-th session from the first test session.
    sessions = pd.read_parquet(root / 'sessions.parquet')
    first = sc.session_position.min()
    sc = sc[(sc.session_position - first) % H == 0].copy()

    # Market cap at formation (point-in-time shares x traded close).
    sh = pd.read_parquet(FUND, filters=[('field', '==', 'BS_SH_OUT'), ('period_type', '==', 'quarterly')],
                         columns=['security', 'availability_date', 'value'])
    sh = sh.dropna().sort_values('availability_date').drop_duplicates(['security', 'availability_date'], keep='last')
    sh['security'] = sh.security.astype(str)
    sc = sc.sort_values('date')
    sc = pd.merge_asof(sc, sh.rename(columns={'availability_date': 'date', 'value': 'shares'}).sort_values('date'),
                       on='date', by='security', direction='backward')
    sc['mcap'] = sc.shares * sc.close_nominal
    sc['cap_rank'] = sc.groupby('date').mcap.rank(ascending=False, method='first')

    # Realised cohort return under each terminal assumption.
    known = sc.forward_return_20.notna()
    term = sc.terminal_within_20 & sc.return_to_last.notna() & ~known
    bankrupt = sc.outcome.eq('bankrupt')
    sc['ret_last'] = np.where(known, sc.forward_return_20, np.where(term, sc.return_to_last, np.nan))
    sc['ret_bk_zero'] = np.where(term & bankrupt, -1.0, sc.ret_last)

    variants = {
        'all, bankrupt at last price': (lambda d: d.index == d.index, 'ret_last'),
        'all, bankrupt at zero': (lambda d: d.index == d.index, 'ret_bk_zero'),
        '$5 floor on traded price': (lambda d: d.close_nominal >= 5, 'ret_bk_zero'),
        '$5 floor on split-adjusted price (look-ahead)': (lambda d: d.close_split_adj >= 5, 'ret_bk_zero'),
        'large caps (top 1000, R1000 proxy)': (lambda d: d.cap_rank <= 1000, 'ret_bk_zero'),
        'small caps (rest, R2000 proxy)': (lambda d: d.cap_rank > 1000, 'ret_bk_zero'),
        'small caps, $5 floor on traded price': (lambda d: (d.cap_rank > 1000) & (d.close_nominal >= 5), 'ret_bk_zero'),
    }
    rows, dec_rows = [], []
    for name, (mask, col) in variants.items():
        d = sc[mask(sc).astype(bool)].copy()
        d['ret'] = d[col]
        for date, g in d.groupby('date', sort=True):
            if len(g) < a.min_names:
                continue
            g = g.sort_values(['score', 'security'])
            g['decile'] = np.arange(len(g)) * 10 // len(g) + 1
            r = g.dropna(subset=['ret'])
            means = r.groupby('decile').ret.mean()
            rows.append({'variant': name, 'date': date, 'names': len(g), 'unknown_dropped': int(g.ret.isna().sum()),
                         'terminal_in_book': int(g.terminal_within_20.sum()),
                         'benchmark_ew': r.ret.mean(), **{f'D{k}': means.get(k, np.nan) for k in range(1, 11)}})
    coh = pd.DataFrame(rows)
    coh['long_short'] = coh.D10 - coh.D1
    coh.to_parquet(out / 'cohort_returns.parquet', index=False)

    last = coh.date.max()
    periods = {'Full': coh.date.min(), '5y': last - pd.DateOffset(years=5), '3y': last - pd.DateOffset(years=3), '1y': last - pd.DateOffset(years=1)}
    met = []
    for name, g in coh.groupby('variant', sort=False):
        for pname, start in periods.items():
            w = g[g.date > start] if pname != 'Full' else g
            for leg in ['long_short', 'D10', 'D1', 'benchmark_ew']:
                met.append({'variant': name, 'period': pname, 'leg': leg, 'first_cohort': w.date.min(), 'last_cohort': w.date.max(), **stats(w[leg])})
            met[-1]['avg_names'] = w.names.mean()
    met = pd.DataFrame(met)
    met.to_parquet(out / 'metrics.parquet', index=False)
    dec = coh.groupby('variant', sort=False)[[f'D{k}' for k in range(1, 11)] + ['benchmark_ew']].mean() * PERIODS_PER_YEAR
    dec.to_parquet(out / 'deciles.parquet')
    with pd.ExcelWriter(out / 'metrics.xlsx') as xw:
        met.to_excel(xw, sheet_name='metrics', index=False)
        dec.to_excel(xw, sheet_name='deciles_annualised')
        coh.to_excel(xw, sheet_name='cohorts', index=False)

    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9})
    figs = []
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, g in coh.groupby('variant', sort=False):
        ax.plot(g.date, np.log((1 + g.long_short.fillna(0)).cumprod()), label=name, lw=1.2)
    ax.set_title('Top-minus-bottom decile, cumulative log growth (gross, 20-session cohorts)')
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    figs.append(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    g = coh[coh.variant == 'all, bankrupt at zero']
    for k in range(1, 11):
        ax.plot(g.date, np.log((1 + g[f'D{k}'].fillna(0)).cumprod()), label=f'D{k}', lw=1, color=plt.cm.RdYlGn((k - 1) / 9))
    ax.plot(g.date, np.log((1 + g.benchmark_ew.fillna(0)).cumprod()), label='equal-weight all', color='black', lw=1.5, ls='--')
    ax.set_title('Deciles vs equal-weight benchmark, all names, bankrupt at zero (cumulative log growth)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.3)
    figs.append(fig)
    with PdfPages(out / 'charts.pdf') as pdf:
        for f in figs:
            pdf.savefig(f)
    imgs = []
    for f in figs:
        b = io.BytesIO()
        f.savefig(b, format='png', dpi=110)
        imgs.append(base64.b64encode(b.getvalue()).decode())
        plt.close(f)

    ls = met[(met.leg == 'long_short') & (met.period == 'Full')].set_index('variant')
    fmt = lambda v, p=False: '—' if pd.isna(v) else (f'{v:+.1%}' if p else f'{v:.2f}')
    table = ''.join(f'<tr><td>{v}</td><td>{fmt(r.ann_return, True)}</td><td>{fmt(r.sharpe)}</td><td>{fmt(r.t_stat)}</td></tr>' for v, r in ls.iterrows())
    partial = '' if years == expected else f'<p><b>Interim:</b> {len(years)} of {len(expected)} test years complete ({years[0]}–{years[-1]}).</p>'
    html = f"""<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><title>R3000 Headline Rerun</title>
<style>body{{font-family:-apple-system,Helvetica,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;background:#fff;color:#111}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}td:first-child{{text-align:left}}img{{max-width:100%}}</style></head><body>
<h1>R3000 headline rerun: I20/R20, expanding, gross</h1>{partial}
<p>Dataset {ref['dataset_id']}; test cohorts {coh.date.min().date()} to {coh.date.max().date()}; {coh[coh.variant=='all, bankrupt at zero'].names.mean():.0f} names per cohort on average.
Returns are simple, equal-weighted within decile, with no costs. Long-short is the top decile minus the bottom decile on non-overlapping 20-session cohorts.</p>
<table><tr><th>Variant</th><th>Long-short annual return</th><th>Sharpe</th><th>t-stat</th></tr>{table}</table>
<p>All periods, legs and the equal-weight benchmark are in metrics.xlsx. A dash marks a cell with too few cohorts to compute.</p>
{''.join(f'<img src="data:image/png;base64,{i}">' for i in imgs)}
</body></html>"""
    (out / 'report.html').write_text(html)
    summary = {'run_dir': str(run), 'dataset': ref, 'years_complete': years, 'cohorts': int(coh.date.nunique()),
               'fundamentals_sha256': sha(FUND), 'size_rank_coverage': float(sc.mcap.notna().mean()),
               'terminal_samples_in_cohorts': int(term.sum()), 'bankrupt_terminal_samples': int((term & bankrupt).sum()),
               'unknown_outcome_samples_in_cohorts': int((~known & ~term).sum()),
               'long_short_full': ls[['ann_return', 'sharpe', 't_stat', 'cagr', 'cohorts']].to_dict('index')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print('OUTPUT', out)


if __name__ == '__main__':
    main()
