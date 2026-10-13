"""
Script: decile_profile.py
Version: 1.0 (2026-10-13)

Description:
    Full ten-decile profile of a report.py run, not just the top-minus-bottom spread.
    Reads report.py's per-cohort decile returns (gross, equal-weight, non-overlapping
    20-session cohorts) and, for each portfolio variant and period, reports every decile's
    annual return, volatility, Sharpe, CAGR, and its excess return over the equal-weight
    benchmark with a Newey-West t-stat and hit rate. Also measures monotonicity (rank
    correlation of decile number with return, overall and cohort by cohort), the decile
    profile by five-year block, and a few wider spreads (top 3 minus bottom 3 deciles,
    top half minus bottom half). Periods follow report.py: Full, and the last 5/3/1 years
    counted back from the last cohort date. No costs or turnover penalties.

Input files:
    <report dir>/cohort_returns.parquet (from research/r3000_headline/report.py)
Output files (in <report dir>/../decile_profile_<UTC timestamp>/):
    deciles_by_period.parquet   variant x period x decile statistics
    deciles_by_block.parquet    variant x five-year block x decile excess return
    spreads.parquet             variant x period x spread statistics
    decile_profile.xlsx         the three tables above, one sheet each
    decile_profile.pdf          charts (light mode): excess return by decile per variant,
                                cumulative growth of all ten deciles, block heatmap
    summary.json

Usage:
    .venv/bin/python research/r3000_headline/decile_profile.py --report-dir DIR

Dependencies: numpy, pandas, matplotlib, openpyxl.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_fixed import nw_t, PPY  # noqa: E402

DEC = [f'D{k}' for k in range(1, 11)]
VARIANTS = ['all, bankrupt at last price', '$5 floor on traded price', 'large caps (top 1000, R1000 proxy)',
            'small caps (rest, R2000 proxy)', 'small caps, $5 floor on traded price']
BLOCKS = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2026)]


def leg_stats(x, bench):
    x, bench = x.astype(float), bench.astype(float)
    ok = x.notna() & bench.notna()
    x, bench = x[ok], bench[ok]
    ex = x - bench
    m, s = x.mean(), x.std()
    growth = float(np.prod(1 + x))
    return {'cohorts': int(len(x)), 'ann_return': m * PPY, 'ann_vol': s * np.sqrt(PPY),
            'sharpe': m / s * np.sqrt(PPY) if s > 0 else np.nan,
            'cagr': growth ** (PPY / len(x)) - 1 if growth > 0 else -1.0,
            'excess_ann': ex.mean() * PPY, 'excess_nw_t': nw_t(ex), 'excess_hit': float((ex > 0).mean()),
            'tracking_error': ex.std() * np.sqrt(PPY)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-dir', required=True)
    a = ap.parse_args()
    rep = Path(a.report_dir).resolve()
    out = rep.parent / ('decile_profile_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    coh = pd.read_parquet(rep / 'cohort_returns.parquet')
    last = coh.date.max()
    periods = {'Full': None, '5y': last - pd.DateOffset(years=5), '3y': last - pd.DateOffset(years=3), '1y': last - pd.DateOffset(years=1)}

    rows, spreads, blocks, mono = [], [], [], {}
    for v in VARIANTS:
        g = coh[coh.variant == v].dropna(subset=DEC + ['benchmark_ew']).sort_values('date')
        for pname, start in periods.items():
            w = g if start is None else g[g.date > start]
            for k, d in enumerate(DEC, 1):
                rows.append({'variant': v, 'period': pname, 'decile': k, **leg_stats(w[d], w.benchmark_ew)})
            rows.append({'variant': v, 'period': pname, 'decile': 'EW', **leg_stats(w.benchmark_ew, w.benchmark_ew)})
            sp = {'D10-D1': w.D10 - w.D1, 'top3-bottom3': w[DEC[7:]].mean(1) - w[DEC[:3]].mean(1),
                  'top5-bottom5': w[DEC[5:]].mean(1) - w[DEC[:5]].mean(1), 'D10-EW': w.D10 - w.benchmark_ew,
                  'EW-D1': w.benchmark_ew - w.D1}
            for name, x in sp.items():
                spreads.append({'variant': v, 'period': pname, 'spread': name, 'cohorts': len(x), 'ann_return': x.mean() * PPY,
                                'ann_vol': x.std() * np.sqrt(PPY), 'sharpe': x.mean() / x.std() * np.sqrt(PPY), 'nw_t': nw_t(x),
                                'hit': float((x > 0).mean())})
            per_cohort = w[DEC].apply(lambda r: pd.Series(r.to_numpy()).rank().corr(pd.Series(np.arange(1, 11))), axis=1)
            avg = w[DEC].mean().to_numpy()
            mono[(v, pname)] = {'rank_corr_of_mean_returns': float(pd.Series(avg).rank().corr(pd.Series(np.arange(1, 11)))),
                                'mean_per_cohort_rank_corr': float(per_cohort.mean()), 'share_cohorts_positive': float((per_cohort > 0).mean())}
        for lo, hi in BLOCKS:
            w = g[(g.date.dt.year >= lo) & (g.date.dt.year <= hi)]
            ex = w[DEC].sub(w.benchmark_ew, axis=0).mean() * PPY
            blocks.append({'variant': v, 'block': f'{lo}-{hi}', 'cohorts': len(w), **ex.to_dict(),
                           'benchmark_ann': w.benchmark_ew.mean() * PPY})

    dp = pd.DataFrame(rows)
    sp = pd.DataFrame(spreads)
    bl = pd.DataFrame(blocks)
    mo = pd.DataFrame([{'variant': k[0], 'period': k[1], **x} for k, x in mono.items()])
    dp['decile'] = dp.decile.astype(str)
    dp.to_parquet(out / 'deciles_by_period.parquet', index=False)
    sp.to_parquet(out / 'spreads.parquet', index=False)
    bl.to_parquet(out / 'deciles_by_block.parquet', index=False)
    with pd.ExcelWriter(out / 'decile_profile.xlsx') as xw:
        dp.to_excel(xw, sheet_name='deciles_by_period', index=False)
        sp.to_excel(xw, sheet_name='spreads', index=False)
        bl.to_excel(xw, sheet_name='deciles_by_block', index=False)
        mo.to_excel(xw, sheet_name='monotonicity', index=False)

    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9})
    cmap = plt.get_cmap('RdYlGn')
    with PdfPages(out / 'decile_profile.pdf') as pdf:
        fig, axes = plt.subplots(len(VARIANTS), 1, figsize=(8.5, 2.3 * len(VARIANTS)), sharex=True)
        for ax, v in zip(axes, VARIANTS):
            f = dp[(dp.variant == v) & (dp.period == 'Full') & (dp.decile != 'EW')]
            colors = [cmap(k / 9) for k in range(10)]
            ax.bar(range(1, 11), f.excess_ann * 100, color=colors, edgecolor='#444', linewidth=0.5)
            for k, (e, t) in enumerate(zip(f.excess_ann, f.excess_nw_t), 1):
                ax.text(k, e * 100 + (0.3 if e >= 0 else -0.3), f't {t:.1f}', ha='center', va='bottom' if e >= 0 else 'top', fontsize=7)
            ax.axhline(0, color='#333', lw=0.8)
            ax.margins(y=0.18)
            ax.set_ylabel('% / yr vs EW')
            ax.set_title(f'{v}: decile return minus equal-weight benchmark, 2000-2026 (gross)', fontsize=9)
        axes[-1].set_xticks(range(1, 11))
        axes[-1].set_xlabel('decile (1 = lowest CNN score, 10 = highest)')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for v in VARIANTS[:1] + VARIANTS[2:4]:
            g = coh[coh.variant == v].dropna(subset=DEC + ['benchmark_ew']).sort_values('date').set_index('date')
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for k, d in enumerate(DEC):
                ax.plot((1 + g[d]).cumprod(), color=cmap(k / 9), lw=1.6 if d in ('D1', 'D10') else 0.9, label=d)
            ax.plot((1 + g.benchmark_ew).cumprod(), color='black', lw=1.2, ls='--', label='EW')
            ax.set_yscale('log')
            ax.set_title(f'{v}: growth of $1 in each decile (gross, log scale)')
            ax.legend(ncol=6, fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, v in zip(axes, VARIANTS[:1] + VARIANTS[2:4]):
            b = bl[bl.variant == v].set_index('block')[DEC] * 100
            lim = np.nanmax(np.abs(b.to_numpy()))
            im = ax.imshow(b.to_numpy(), cmap='RdYlGn', vmin=-lim, vmax=lim, aspect='auto')
            ax.set_xticks(range(10), [str(k) for k in range(1, 11)])
            ax.set_yticks(range(len(b)), b.index)
            for i in range(b.shape[0]):
                for j in range(10):
                    ax.text(j, i, f'{b.iat[i, j]:.0f}', ha='center', va='center', fontsize=7)
            ax.set_title(v.split(' (')[0] + ': % / yr vs EW', fontsize=9)
            ax.set_xlabel('decile')
        fig.colorbar(im, ax=axes, shrink=0.8)
        pdf.savefig(fig)
        plt.close(fig)

    summary = {'report_dir': str(rep), 'last_cohort': str(last.date()),
               'monotonicity': mo.round(4).to_dict('records')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option('display.width', 250)
    for v in VARIANTS:
        f = dp[(dp.variant == v) & (dp.period == 'Full')][['decile', 'ann_return', 'ann_vol', 'sharpe', 'excess_ann', 'excess_nw_t', 'excess_hit']]
        print(v)
        print(f.round(3).to_string(index=False))
    print(sp[sp.period == 'Full'].round(3).to_string(index=False))
    print(mo.round(3).to_string(index=False))
    for v in VARIANTS[:1] + VARIANTS[2:4]:
        print(v)
        print((bl[bl.variant == v].set_index('block')[DEC + ['benchmark_ann']] * 100).round(1).to_string())
    for p in ['5y', '3y', '1y']:
        f = dp[(dp.variant == VARIANTS[0]) & (dp.period == p)][['decile', 'ann_return', 'excess_ann', 'excess_nw_t']]
        print(p, (f.set_index('decile').excess_ann * 100).round(1).to_dict())
    print('OUTPUT', out)


if __name__ == '__main__':
    main()
