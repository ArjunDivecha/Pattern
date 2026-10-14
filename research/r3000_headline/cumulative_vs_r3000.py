"""
Script: cumulative_vs_r3000.py
Version: 1.0 (2026-10-14)

Description:
    Cumulative growth of the CNN portfolios against the Russell 3000 itself, for the R3000
    headline run (expanding retraining, 2000-2026). Uses the same non-overlapping 20-session
    cohorts, point-in-time R3000 members and delisting convention (bankrupt at last price) as
    report.py, gross of all costs. Benchmarks are built from the same names on the same dates:
      R3000 cap-weighted   weights = point-in-time shares x traded close at formation (the index's
                           own weighting; names without a market cap, ~2% of rows, are left out of it)
      R3000 equal-weight   every member equally
    CNN books (equal weight within each): top decile; top three deciles; the market without the
    bottom decile (deciles 2-10); and the market-neutral D10-D1 long-short, raw and
    factor-neutralised (residual after all controls + sector, as in neutralize.py).
    Reports CAGR, volatility, Sharpe, max drawdown and growth of $1 over Full/5y/3y/1y.

Input files:
    <run dir>/neutralize_<ts>/characteristics.parquet (from research/r3000_headline/neutralize.py)
    <run dir>/report_<ts>/cohort_returns.parquet (from report.py; used only to check the deciles match)
Output files (in <run dir>/cumulative_<UTC timestamp>/):
    cohort_series.parquet   per cohort date: return of every line
    stats.parquet, stats.xlsx  per line x period: CAGR, vol, Sharpe, max drawdown, growth of $1
    cumulative_vs_r3000.pdf charts (light mode): long-only books vs R3000; long-short books; drawdowns

Usage:
    .venv/bin/python research/r3000_headline/cumulative_vs_r3000.py --char FILE --report-dir DIR

Dependencies: numpy, pandas, matplotlib, openpyxl.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neutralize import rank_norm, ALL  # noqa: E402
from verify_fixed import PPY  # noqa: E402


def neutral_residual(d):
    rk = rank_norm(d, ['score'] + ALL)
    rk.columns = ['cnn'] + ALL
    sec = pd.get_dummies(d.sector, drop_first=True, dtype=float).to_numpy()
    X = np.column_stack([np.ones(len(d)), rk[ALL].to_numpy(), sec])
    y = rk.cnn.to_numpy()
    dates = d.date.to_numpy()
    b = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    res = np.empty(len(d))
    for i in range(len(b) - 1):
        s = slice(b[i], b[i + 1])
        res[s] = y[s] - X[s] @ np.linalg.lstsq(X[s], y[s], rcond=None)[0]
    return res


def decile(g, col):
    g = g.sort_values([col, 'security'])
    return pd.Series(np.arange(len(g)) * 10 // len(g) + 1, index=g.index)


def stats(x):
    x = x.dropna()
    w = (1 + x).cumprod()
    dd = w / w.cummax() - 1
    return {'cohorts': len(x), 'cagr': w.iloc[-1] ** (PPY / len(x)) - 1, 'ann_vol': x.std() * np.sqrt(PPY),
            'sharpe': x.mean() / x.std() * np.sqrt(PPY), 'max_drawdown': dd.min(), 'growth_of_1': w.iloc[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--char', required=True)
    ap.add_argument('--report-dir', required=True)
    a = ap.parse_args()
    char, rep = Path(a.char).resolve(), Path(a.report_dir).resolve()
    run = char.parent.parent
    out = run / ('cumulative_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    d = pd.read_parquet(char)
    d = d[d.date.isin(d.dropna(subset=['ret']).date.unique())].sort_values(['date', 'security']).reset_index(drop=True)
    d['neu'] = neutral_residual(d)
    d['mcap'] = np.exp(d.log_mcap)
    d['dec'] = d.groupby('date', group_keys=False).apply(lambda g: decile(g, 'score'), include_groups=False)
    d['dec_neu'] = d.groupby('date', group_keys=False).apply(lambda g: decile(g, 'neu'), include_groups=False)
    k = d.dropna(subset=['ret'])
    rows = []
    for date, g in k.groupby('date', sort=True):
        c = g.dropna(subset=['mcap'])
        rows.append({'date': date,
                     'R3000 cap-weighted': np.average(c.ret, weights=c.mcap),
                     'R3000 equal-weight': g.ret.mean(),
                     'CNN top decile': g.ret[g.dec == 10].mean(),
                     'CNN top 3 deciles': g.ret[g.dec >= 8].mean(),
                     'CNN market ex bottom decile': g.ret[g.dec >= 2].mean(),
                     'D1': g.ret[g.dec == 1].mean(),
                     'neu D10': g.ret[g.dec_neu == 10].mean(), 'neu D1': g.ret[g.dec_neu == 1].mean()})
    s = pd.DataFrame(rows).set_index('date')
    s['CNN long-short D10-D1 (raw)'] = s['CNN top decile'] - s.D1
    s['CNN long-short D10-D1 (factor-neutralised)'] = s['neu D10'] - s['neu D1']
    s = s.drop(columns=['D1', 'neu D10', 'neu D1'])

    # check the deciles reproduce report.py
    r = pd.read_parquet(rep / 'cohort_returns.parquet')
    r = r[r.variant == 'all, bankrupt at last price'].set_index('date')
    diff = (s['CNN top decile'] - r.D10.reindex(s.index)).abs().max()
    print(f'max |D10 - report D10| = {diff:.2e}; max |EW - report EW| = {(s["R3000 equal-weight"] - r.benchmark_ew.reindex(s.index)).abs().max():.2e}')
    assert diff < 1e-12

    s.to_parquet(out / 'cohort_series.parquet')
    last = s.index.max()
    periods = {'Full': s.index.min() - pd.Timedelta(days=1), '5y': last - pd.DateOffset(years=5),
               '3y': last - pd.DateOffset(years=3), '1y': last - pd.DateOffset(years=1)}
    st = pd.DataFrame([{'line': c, 'period': p, **stats(s.loc[s.index > start, c])} for p, start in periods.items() for c in s.columns])
    st.to_parquet(out / 'stats.parquet', index=False)
    with pd.ExcelWriter(out / 'stats.xlsx') as xw:
        st.to_excel(xw, sheet_name='stats', index=False)
        s.to_excel(xw, sheet_name='cohort_returns')

    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9})
    long_lines = {'R3000 cap-weighted': ('black', 2.0, '-'), 'R3000 equal-weight': ('#777777', 1.4, '--'),
                  'CNN top decile': ('#1a7f37', 1.8, '-'), 'CNN top 3 deciles': ('#2f6fb3', 1.4, '-'),
                  'CNN market ex bottom decile': ('#c46a00', 1.4, '-')}
    with PdfPages(out / 'cumulative_vs_r3000.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(11, 6))
        for c, (col, lw, ls) in long_lines.items():
            w = (1 + s[c]).cumprod()
            f = st[(st.line == c) & (st.period == 'Full')].iloc[0]
            ax.plot(w, color=col, lw=lw, ls=ls, label=f'{c}: ${w.iloc[-1]:.1f}, CAGR {f.cagr:.1%}, Sharpe {f.sharpe:.2f}')
        ax.set_yscale('log')
        ax.set_title('Growth of $1, 2000 to mid-2026: CNN long-only books vs the Russell 3000 (gross, log scale)')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5))
        for c, col in [('CNN long-short D10-D1 (raw)', '#1a7f37'), ('CNN long-short D10-D1 (factor-neutralised)', '#8a3ab9'),
                       ('R3000 cap-weighted', 'black')]:
            w = (1 + s[c]).cumprod()
            f = st[(st.line == c) & (st.period == 'Full')].iloc[0]
            ax.plot(w, color=col, lw=1.6, label=f'{c}: ${w.iloc[-1]:.1f}, CAGR {f.cagr:.1%}, Sharpe {f.sharpe:.2f}, max DD {f.max_drawdown:.0%}')
        ax.set_yscale('log')
        ax.set_title('Growth of $1: market-neutral long-short books vs the Russell 3000 (gross, log scale)')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        for c in ['R3000 cap-weighted', 'CNN top decile', 'CNN long-short D10-D1 (raw)']:
            w = (1 + s[c]).cumprod()
            ax.plot(w / w.cummax() - 1, lw=1.3, color={'R3000 cap-weighted': 'black', 'CNN top decile': '#1a7f37',
                                                         'CNN long-short D10-D1 (raw)': '#c46a00'}[c], label=c)
        ax.set_title('Drawdown from previous peak')
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pd.set_option('display.width', 250)
    print(st.round(3).to_string(index=False))
    print('OUTPUT', out)


if __name__ == '__main__':
    main()
