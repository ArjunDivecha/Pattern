"""
Script: offsets.py
Version: 1.1 (2026-10-14)

Description:
    Start-date robustness for the R3000 headline run (expanding retraining, 2000-2026). The
    report forms a portfolio every 20th session counted from the first test session (offset 0).
    The CNN scores every stock every day, so the same backtest can start on any of the 20
    sessions; this runs all 20 and builds the staggered book that does not depend on the start.

    Part 1, the 20 offsets. For each offset k = 0..19, cohorts form on sessions first+k, first+k+20,
    ... and hold 20 sessions, exactly as report.py (deciles over all names, returns over known
    outcomes, delisted names earn the return to the last price; the size slices use bankrupt =
    -100% as report.py does). Offset 0 must reproduce report.py's cohort returns; this is asserted.
    Books: raw score and sector-demeaned score (rank-normalised score minus its Bloomberg-sector
    mean, as in sector_slice.py), for all names, large caps (top 1000 by point-in-time market cap)
    and small caps (rest).

    Part 2, the staggered book. Every session a new tranche forms (deciles on that day's scores)
    and holds 20 sessions, buy and hold, equal weight at formation; a name that delists inside the
    horizon is held at its last price (cash) for the rest of the tranche, and a name with an unknown
    outcome is dropped, as in Part 1. The book holds the 20 live tranches in equal shares, so its
    daily return is the mean of the 20 tranches' daily returns. D10-D1 is long the top-decile
    tranches and short the bottom. Daily returns from the total return index. The book's daily
    returns are positively autocorrelated (lag-1 ~0.1; 20-day variance ratio 1.3-1.8), so a
    sqrt(252) daily Sharpe overstates it; v1.1 reports statistics on calendar-month compounded
    returns (Sharpe x sqrt(12), NW t with 3 lags) as the primary measure and keeps the daily
    Sharpe alongside for reference. Each tranche's 20-day buy-and-hold return is checked against
    the cohort return of Part 1.

    Periods: Full, 5y, 3y, 1y (counted back from the last date) and 2000-2011 / 2012-2026. Gross
    of all costs. Sectors are today's Bloomberg INDUSTRY_SECTOR applied backwards (not PIT).

Input files:
    <run dir>/Y<year>/ensemble_scores.parquet  (from train.py)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/index.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010):
        prices.parquet (total_return_index_gross_dividends), quality.parquet (off_exchange_calendar),
        nominal_prices.parquet (close_split_adj, via mcap.py), delisting_outcomes.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/fundamentals.parquet (BS_SH_OUT, via mcap.py)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/ref_fields/security_reference.parquet (via sector_slice.py)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/edgar/data/{security_cik_map,entities}.parquet (via sector_slice.py)
    <run dir>/report_<ts>/cohort_returns.parquet (check only)
Output files (in <run dir>/offsets_<UTC timestamp>/):
    offset_cohorts.parquet    book x offset x cohort date: D1, D10, EW, D10-D1
    offset_stats.parquet      book x offset x period: D10-D1 stats
    offset_summary.parquet    book x period: min / median / max across the 20 offsets, offset 0's rank
    staggered_daily.parquet   book x date: daily D10, D1, EW and D10-D1 returns of the staggered book
    staggered_stats.parquet   book x period x leg: monthly-return annual return, vol, Sharpe, NW t; CAGR, max drawdown; daily Sharpe
    offsets.xlsx              the stats tables
    offsets.pdf               charts (light mode)
    summary.json; progress.json (updated per stage)

Usage:
    .venv/bin/python research/r3000_headline/offsets.py --run-dir DIR --report-dir DIR

Dependencies: numpy, pandas, matplotlib, openpyxl.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import resolve_dataset, H  # noqa: E402
from verify_fixed import nw_t, PPY  # noqa: E402
from mcap import market_caps  # noqa: E402
from sector_slice import sector_map  # noqa: E402

T0 = time.time()
BOOKS = {  # name: (universe, signal, return column)
    'all names, raw': ('all', 'raw', 'ret_last'),
    'all names, sector-neutral': ('all', 'sec', 'ret_last'),
    'large caps, raw': ('large', 'raw', 'ret_bk_zero'),
    'large caps, sector-neutral': ('large', 'sec', 'ret_bk_zero'),
    'small caps, raw': ('small', 'raw', 'ret_bk_zero'),
    'small caps, sector-neutral': ('small', 'sec', 'ret_bk_zero'),
}


def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def periods(idx):
    last = idx.max()
    return {'Full': idx == idx, '5y': idx > last - pd.DateOffset(years=5), '3y': idx > last - pd.DateOffset(years=3),
            '1y': idx > last - pd.DateOffset(years=1), '2000-2011': idx.year <= 2011, '2012-2026': idx.year >= 2012}


def cohort_stats(x):
    x = x.dropna()
    if len(x) < 3:
        return {'cohorts': len(x), 'ann_return': np.nan, 'sharpe': np.nan, 'nw_t': np.nan}
    return {'cohorts': len(x), 'ann_return': x.mean() * PPY, 'sharpe': x.mean() / x.std() * np.sqrt(PPY), 'nw_t': nw_t(x)}


def daily_stats(x):
    x = x.dropna()
    w = (1 + x).cumprod()
    m = (1 + x).resample('ME').prod() - 1     # calendar-month returns: daily returns are autocorrelated
    return {'months': len(m), 'ann_return': m.mean() * 12, 'ann_vol': m.std() * np.sqrt(12),
            'sharpe': m.mean() / m.std() * np.sqrt(12), 'cagr': w.iloc[-1] ** (252 / len(x)) - 1,
            'max_drawdown': float((w / w.cummax() - 1).min()), 'nw_t': nw_t(m, lags=3),
            'sharpe_daily_sqrt252': x.mean() / x.std() * np.sqrt(252), 'lag1_autocorr_daily': x.autocorr(1)}


def assign_deciles(d, col):
    """Decile within (date) over all names in d, ties by security, as report.py."""
    d = d.sort_values(['date', col, 'security'])
    pos = d.groupby('date').cumcount().to_numpy()
    n = d.groupby('date')[col].transform('size').to_numpy()
    return pd.Series(pos * 10 // n + 1, index=d.index).reindex(d.index)


def main():
    global OBS
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--report-dir', required=True)
    ap.add_argument('--index-dir', default=str(ROOT / 'cache/r3000_I20_M20_v2_index'))
    ap.add_argument('--min-names', type=int, default=100)
    a = ap.parse_args()
    run, rep = Path(a.run_dir).resolve(), Path(a.report_dir).resolve()
    out = run / ('offsets_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    prog = {'stage': 'start', 'out': str(out)}
    write_json(out / 'progress.json', prog)
    root, ref = resolve_dataset()

    # ---- daily scored panel, as report.py but every session ----
    years = sorted(int(p.name[1:]) for p in run.glob('Y*') if (p / 'COMPLETE.json').exists())
    assert years == list(range(2000, 2027)), years
    sc = pd.concat([pd.read_parquet(run / f'Y{y}' / 'ensemble_scores.parquet', columns=['row_id', 'security', 'date', 'session_position', 'score']) for y in years])
    idx = pd.read_parquet(Path(a.index_dir) / 'index.parquet', columns=['row_id', 'forward_return_20', 'terminal_within_20', 'return_to_last'])
    sc = sc.merge(idx, on='row_id', validate='one_to_one')
    sc['security'] = sc.security.astype(str)
    dl = pd.read_parquet(root / 'delisting_outcomes.parquet', columns=['security', 'outcome'])
    dl['security'] = dl.security.astype(str)
    sc = sc.merge(dl, on='security', how='left')
    known = sc.forward_return_20.notna()
    term = sc.terminal_within_20 & sc.return_to_last.notna() & ~known
    sc['ret_last'] = np.where(known, sc.forward_return_20, np.where(term, sc.return_to_last, np.nan))
    sc['ret_bk_zero'] = np.where(term & sc.outcome.eq('bankrupt'), -1.0, sc.ret_last)
    sc['terminal'] = term
    sc = sc.sort_values(['date', 'security']).reset_index(drop=True)
    log(f'scored rows {len(sc):,}, sessions {sc.date.nunique():,}')
    sc['mcap'] = market_caps(root, sc)
    sc['cap_rank'] = sc.groupby('date').mcap.rank(ascending=False, method='first')
    sm = sector_map(sorted(sc.security.unique()))
    sc = sc.merge(sm[['security', 'sector']], on='security', how='left')
    first = sc.session_position.min()
    sc['offset'] = (sc.session_position - first) % H
    log('market caps and sectors attached')

    # per-universe signals and deciles (ranks re-taken inside the universe, as sector_slice.py)
    uni_mask = {'all': np.ones(len(sc), bool), 'large': (sc.cap_rank <= 1000).to_numpy(), 'small': (sc.cap_rank > 1000).to_numpy()}
    dec = {}
    for uni, m in uni_mask.items():
        g = sc.loc[m, ['date', 'security', 'score', 'sector']]
        r = g.groupby('date').score.rank(pct=True) - 0.5
        sec = r - r.groupby([g.date, g.sector]).transform('mean')
        dec[(uni, 'raw')] = assign_deciles(g.assign(s=g.score), 's')
        dec[(uni, 'sec')] = assign_deciles(g.assign(s=sec), 's')
        log(f'deciles {uni}')
    names = sc.groupby('date').size()

    # ---- Part 1: 20 offsets ----
    prog['stage'] = 'offsets'
    write_json(out / 'progress.json', prog)
    coh_rows = []
    for book, (uni, sig, col) in BOOKS.items():
        m = uni_mask[uni]
        g = sc.loc[m, ['date', 'offset', col]].assign(dec=dec[(uni, sig)])
        cnt = g.groupby('date').size()
        g = g[g.date.map(cnt) >= a.min_names].dropna(subset=[col])
        p = g.pivot_table(index=['date', 'offset'], columns='dec', values=col, aggfunc='mean')
        ew = g.groupby(['date', 'offset'])[col].mean()
        df = pd.DataFrame({'D1': p[1], 'D10': p[10], 'EW': ew}).reset_index()
        df['LS'] = df.D10 - df.D1
        df['book'] = book
        coh_rows.append(df)
    coh = pd.concat(coh_rows, ignore_index=True)
    coh.to_parquet(out / 'offset_cohorts.parquet', index=False)

    r = pd.read_parquet(rep / 'cohort_returns.parquet')
    for book, variant in [('all names, raw', 'all, bankrupt at last price'), ('large caps, raw', 'large caps (top 1000, R1000 proxy)'),
                          ('small caps, raw', 'small caps (rest, R2000 proxy)')]:
        mine = coh[(coh.book == book) & (coh.offset == 0)].set_index('date').LS.dropna()
        theirs = r[r.variant == variant].set_index('date').long_short.dropna()   # report keeps one final all-unknown cohort as blank
        diff = (mine - theirs.reindex(mine.index)).abs().max()
        log(f'offset 0 vs report, {book}: max |diff| {diff:.2e}, cohorts {len(mine)} vs {len(theirs)}')
        assert diff < 1e-12 and len(mine) == len(theirs), book

    st_rows = []
    for (book, k), g in coh.groupby(['book', 'offset']):
        s = g.set_index('date').LS.sort_index()
        for p, msk in periods(s.index).items():
            st_rows.append({'book': book, 'offset': k, 'period': p, **cohort_stats(s[msk])})
    ost = pd.DataFrame(st_rows)
    ost.to_parquet(out / 'offset_stats.parquet', index=False)
    summ = []
    for (book, p), g in ost.groupby(['book', 'period'], sort=False):
        o0 = g[g.offset == 0].iloc[0]
        summ.append({'book': book, 'period': p, 'sharpe_offset0': o0.sharpe, 'sharpe_min': g.sharpe.min(),
                     'sharpe_median': g.sharpe.median(), 'sharpe_mean': g.sharpe.mean(), 'sharpe_max': g.sharpe.max(),
                     'offset0_rank_of_20': int((g.sharpe > o0.sharpe).sum() + 1),
                     'ann_return_offset0': o0.ann_return, 'ann_return_min': g.ann_return.min(), 'ann_return_max': g.ann_return.max(),
                     'nw_t_min': g.nw_t.min(), 'nw_t_max': g.nw_t.max(), 'offsets_t_above_2': int((g.nw_t > 2).sum())})
    osum = pd.DataFrame(summ)
    osum.to_parquet(out / 'offset_summary.parquet', index=False)
    log('offsets done')

    # ---- Part 2: staggered book from daily TRI paths ----
    prog['stage'] = 'staggered'
    write_json(out / 'progress.json', prog)
    secs = sorted(sc.security.unique())
    sess = pd.read_parquet(root / 'sessions.parquet')
    lo = sc.session_position.min()
    hi = sess.session_index.max()
    filt = [('security', 'in', secs), ('date', '>=', sc.date.min())]
    px = pd.read_parquet(root / 'prices.parquet', columns=['security', 'date', 'total_return_index_gross_dividends'], filters=filt)
    q = pd.read_parquet(root / 'quality.parquet', columns=['security', 'date', 'off_exchange_calendar'], filters=filt)
    px = px.merge(q, on=['security', 'date'], validate='one_to_one')
    px = px[~px.off_exchange_calendar]
    px['security'] = px.security.astype(str)
    px = px.merge(sess, on='date', how='inner')
    sec_id = {s: i for i, s in enumerate(secs)}
    T = np.full((hi - lo + 1, len(secs)), np.nan)
    T[px.session_index.to_numpy() - lo, px.security.map(sec_id).to_numpy()] = px.total_return_index_gross_dividends.to_numpy(float)
    OBS = ~np.isnan(T)                        # bars actually observed (before carrying forward)
    T = pd.DataFrame(T).ffill().to_numpy()   # after the last bar a name is held at its last price
    del px, q
    cal = sess.set_index('session_index').date
    log(f'TRI matrix {T.shape}')

    sc['col'] = sc.security.map(sec_id).to_numpy()
    sc['row'] = sc.session_position.to_numpy() - lo
    daily = {}
    max_check = {}
    for book, (uni, sig, col) in BOOKS.items():
        m = uni_mask[uni]
        g = sc.loc[m, ['date', 'row', 'col', col]].assign(dec=dec[(uni, sig)])
        cnt = g.groupby('date').size()
        g = g[g.date.map(cnt) >= a.min_names].dropna(subset=[col])
        legs = {'D10': g[g.dec == 10], 'D1': g[g.dec == 1], 'EW': g}
        acc = {}
        worst = 0.0
        for leg, gl in legs.items():
            ret_acc = np.zeros(T.shape[0])
            n_acc = np.zeros(T.shape[0])
            for (row0,), gg in gl.groupby(['row']):
                cols = gg['col'].to_numpy()
                if row0 + H >= T.shape[0]:
                    continue
                path = T[row0:row0 + H + 1, cols] / T[row0, cols]
                v = path.mean(axis=1)                           # buy-and-hold tranche value, equal weight at formation
                if col == 'ret_bk_zero':
                    # bankrupt names at -100%: written off the session after their last bar
                    bk = np.flatnonzero(gg[col].to_numpy() == -1.0)
                    if len(bk):
                        path = path.copy()
                        for j in bk:
                            seen = np.flatnonzero(OBS[row0:row0 + H + 1, cols[j]])
                            path[seen[-1] + 1:, j] = 0.0
                        v = path.mean(axis=1)
                worst = max(worst, abs(v[-1] - 1 - gg[col].mean()))
                ret_acc[row0 + 1:row0 + H + 1] += v[1:] / v[:-1] - 1
                n_acc[row0 + 1:row0 + H + 1] += 1
            acc[leg] = (ret_acc, n_acc)
        full = acc['EW'][1] == H                                 # days with all 20 tranches live
        dts = cal.reindex(np.arange(T.shape[0]) + lo).to_numpy()
        df = pd.DataFrame({leg: np.where(full, acc[leg][0] / np.maximum(acc[leg][1], 1), np.nan) for leg in acc}, index=pd.DatetimeIndex(dts))
        df = df[full]
        df['LS'] = df.D10 - df.D1
        daily[book] = df
        max_check[book] = worst
        log(f'staggered {book}: days {len(df):,}, max |tranche 20d return - cohort mean| {worst:.2e}, '
            f'LS daily-sqrt252 Sharpe {df.LS.mean() / df.LS.std() * np.sqrt(252):.2f} (overstated; see monthly stats)')
        prog[f'staggered_{book}'] = 'done'
        write_json(out / 'progress.json', prog)
    dd = pd.concat({b: v for b, v in daily.items()}, names=['book', 'date']).reset_index()
    dd.to_parquet(out / 'staggered_daily.parquet', index=False)
    sst = []
    for book, df in daily.items():
        for p, msk in periods(df.index).items():
            for leg in ['LS', 'D10', 'D1', 'EW']:
                sst.append({'book': book, 'period': p, 'leg': leg, **daily_stats(df.loc[msk, leg])})
    sst = pd.DataFrame(sst)
    sst.to_parquet(out / 'staggered_stats.parquet', index=False)

    with pd.ExcelWriter(out / 'offsets.xlsx') as xw:
        pd.DataFrame({'note': ['Gross of all costs. Offset k = cohorts formed every 20th session starting k sessions after the first test session; offset 0 is report.py.',
                               'Staggered book: 20 overlapping buy-and-hold tranches, one formed each session, equal shares; daily returns.',
                               'Sectors are today\'s Bloomberg INDUSTRY_SECTOR applied backwards (not point-in-time).']}).to_excel(xw, sheet_name='notes', index=False)
        osum.to_excel(xw, sheet_name='offset_summary', index=False)
        sst.to_excel(xw, sheet_name='staggered_stats', index=False)
        ost.to_excel(xw, sheet_name='offset_stats', index=False)

    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9})
    with PdfPages(out / 'offsets.pdf') as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
        for ax, book in zip(axes.T.ravel(), BOOKS):
            g = ost[(ost.book == book) & (ost.period == 'Full')].sort_values('offset')
            ax.bar(g.offset, g.sharpe, color=['#c46a00' if k == 0 else '#2f6fb3' for k in g.offset])
            stg = sst[(sst.book == book) & (sst.period == 'Full') & (sst.leg == 'LS')].iloc[0]
            ax.axhline(stg.sharpe, color='black', lw=1.2, ls='--', label=f'staggered book (monthly returns): {stg.sharpe:.2f}')
            ax.set_title(f'{book}: D10-D1 Sharpe by start offset, 2000-2026', fontsize=8.5)
            ax.legend(fontsize=7, loc='lower right')
            ax.set_xticks(range(0, 20, 2))
        for ax in axes[-1]:
            ax.set_xlabel('start offset (sessions after the first test session; orange = report)')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        for book, colr in zip(BOOKS, ['#1a7f37', '#2f6fb3', '#c46a00', '#8a3ab9', '#777777', '#b3302f']):
            x = daily[book].LS
            f = sst[(sst.book == book) & (sst.period == 'Full') & (sst.leg == 'LS')].iloc[0]
            ax.plot((1 + x).cumprod(), color=colr, lw=1.3, label=f'{book}: CAGR {f.cagr:.1%}, Sharpe {f.sharpe:.2f}, max DD {f.max_drawdown:.0%}')
        ax.set_yscale('log')
        ax.set_title('Staggered D10-D1 books (20 overlapping tranches), growth of $1, daily, gross, log scale')
        ax.legend(fontsize=7.5, loc='upper left')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5))
        for k in range(H):
            s = coh[(coh.book == 'all names, raw') & (coh.offset == k)].set_index('date').LS.sort_index()
            ax.plot((1 + s).cumprod(), color='#c46a00' if k == 0 else '#9bb7d9', lw=1.6 if k == 0 else 0.7, zorder=3 if k == 0 else 2)
        x = daily['all names, raw'].LS
        ax.plot((1 + x).cumprod(), color='black', lw=1.4, label='staggered book (daily)')
        ax.set_yscale('log')
        ax.set_title('All names, raw D10-D1: each of the 20 start offsets (blue; orange = report) and the staggered book (black)')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    summary = {'run_dir': str(run), 'report_dir': str(rep), 'scored_rows': int(len(sc)), 'first_session_position': int(first),
               'tranche_return_check_max_abs': max_check,
               'offset_summary_full': osum[osum.period == 'Full'].round(4).to_dict('records'),
               'staggered_ls': sst[sst.leg == 'LS'].round(4).to_dict('records')}
    write_json(out / 'summary.json', summary)
    prog['stage'] = 'done'
    write_json(out / 'progress.json', prog)
    pd.set_option('display.width', 250)
    print(osum.round(2).to_string(index=False))
    print(sst[sst.leg == 'LS'].pivot(index='book', columns='period', values='sharpe').round(2).to_string())
    print(sst[(sst.leg == 'LS') & (sst.period == 'Full')].round(3).to_string(index=False))
    print(sst[(sst.leg != 'LS') & (sst.period == 'Full') & (sst.book == 'all names, raw')].round(3).to_string(index=False))
    print('OUTPUT', out)


if __name__ == '__main__':
    main()
