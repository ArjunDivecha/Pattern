"""
Script: smallliq_ls.py
Version: 1.1 (2026-11-01)

Description:
    v1.1 adds --universe r3000 (every scored point-in-time Russell 3000 member) and --per-side-usd, a
    capacity screen at a fixed dollar book: a name is eligible for an N-per-side book only if its 60-session
    mean dollar volume (SmallLiq's adv60_usd, all members) is at least (per-side capital / N) / (1% x 5
    sessions), i.e. 20x the position, SmallLiq's own participation contract. Each leg is then measured
    against the equal weight of that N's eligible pool. The 10%-of-pool book is skipped under a capacity
    screen (its N depends on the pool). Holdings profile (median ADV, spread, share under $5) is reported.
    Default arguments reproduce v1.0 exactly.

    Long-short backtest of the R3000 CNN score inside the Global Data Mart's SmallLiq indexed universe
    (equities.small_liq_indexed_monthly, eligible_indexed): the smaller two-thirds of the Russell 3000
    that pass price, spread, continuity and market-liquidity-indexed ADV screens at each calendar
    month-end, 2000-01 to 2026-06. The universe is a market-liquidity filter only; short borrow is not
    verified (short_borrow_verified is always False).

    At each month-end formation date the eligible names are ranked by their CNN score on the last
    session on or before the month-end (the expanding-window out-of-sample score of the headline run;
    names without a valid chart that day cannot be ranked and are dropped, counted). The book goes long
    the N highest and short the N lowest, equal weight at entry, buy and hold for the month, rebalanced
    monthly, for N = 10, 20, 30, 40, 50, 75, 100 per side, plus the top and bottom 10% of the pool.

    Execution, as SmallLiq requires ("trade only on a later session"): enter at the OPEN of the first
    session after the formation session, exit at the open of the first session after the next
    formation session, when the replacement book enters. The signal-close convention (enter at the
    formation close) is reported alongside for comparison. Opens are priced on the total-return scale
    (TRI x open / close of the bar). A name with no bar on the entry session cannot be bought and is
    dropped (counted); a name that stops trading during the month is held at its last price. The last
    holding month (formed 2026-05-31) exits at the 2026-06-30 close, the end of the frozen data.

    Reported per N: long leg, short leg, long-short (long minus short), long minus the pool's equal
    weight, pool minus short, and the pool's equal-weight return (benchmark). Gross of all costs and
    borrow. Monthly statistics: annual mean, volatility, Sharpe, Newey-West t (3 lags), CAGR, worst
    drawdown, share of positive months, over Full, 2006-2026 (SmallLiq marks 2000-2005 as early-coverage
    diagnostic), 2000-2011, 2012-2026, 5y, 3y, 1y. Monthly one-way turnover per side is reported for
    information only (no cost is charged).

Input files:
    gdm dataset "equities.small_liq_monthly" (adv60_usd, median_full_spread_bps60, nominal_close, r3000_member; all members)
    gdm dataset "equities.small_liq_indexed_monthly"
        (/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/small_liq_indexed/monthly.parquet)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z/Y<year>/ensemble_scores.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010: prices, quality, sessions)
Output files (in <run dir>/smallliq_ls_<UTC timestamp>/, or monthly_ls_<universe>[_<capital>]_<UTC timestamp>/ for v1.1 options):
    monthly.parquet     N x entry x month: long, short, LS, long_vs_pool, pool_vs_short, pool, names, turnover
    stats.parquet       N x entry x book x period statistics; smallliq_ls.xlsx; smallliq_ls.pdf (light mode)
    coverage.parquet    per month: eligible names, scored names, dropped at entry
    summary.json; log tee'd by the caller to verify_logs/smallliq_ls_v1.0.log

Usage:
    .venv/bin/python research/r3000_headline/smallliq_ls.py [--universe smallliq|r3000] [--per-side-usd 1000000]

Dependencies: numpy, pandas, matplotlib, openpyxl; gdm (Global Data Mart), verify_fixed.nw_t, prepare.resolve_dataset.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart')
from gdm import dataset  # noqa: E402
from prepare import resolve_dataset  # noqa: E402
from verify_fixed import nw_t  # noqa: E402

RUN = ROOT / 'research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z'
NS = [10, 20, 30, 40, 50, 75, 100, 'decile']
ENTRIES = {'open': 'next-session open (realistic)', 'close': 'formation-session close (research convention)'}
T0 = time.time()


def log(m):
    print(f'[{time.time() - T0:7.1f}s] {m}', flush=True)


def mstats(x):
    x = x.dropna()
    if len(x) < 6:
        return {'months': len(x)}
    w = (1 + x).cumprod()
    return {'months': len(x), 'ann_return': x.mean() * 12, 'ann_vol': x.std() * np.sqrt(12),
            'sharpe': x.mean() / x.std() * np.sqrt(12), 'nw_t': nw_t(x, lags=3), 'cagr': w.iloc[-1] ** (12 / len(x)) - 1,
            'max_drawdown': float((w / w.cummax() - 1).min()), 'share_positive': (x > 0).mean()}


def periods(idx):
    last = idx.max()
    return {'Full': idx == idx, '2006-2026': idx.year >= 2006, '2000-2011': idx.year <= 2011, '2012-2026': idx.year >= 2012,
            '5y': idx > last - pd.DateOffset(years=5), '3y': idx > last - pd.DateOffset(years=3), '1y': idx > last - pd.DateOffset(years=1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--universe', choices=['smallliq', 'r3000'], default='smallliq')
    ap.add_argument('--per-side-usd', type=float, default=None)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if a.universe == 'smallliq' and a.per_side_usd is None:
        out = RUN / ('smallliq_ls_' + stamp)
    else:
        out = RUN / (f'monthly_ls_{a.universe}' + (f'_{a.per_side_usd / 1e6:g}m' if a.per_side_usd else '') + '_' + stamp)
    out.mkdir()
    root, _ = resolve_dataset()
    liq = pd.read_parquet(dataset('equities.small_liq_monthly', must_exist=True),
                          columns=['asof_date', 'security', 'r3000_member', 'adv60_usd', 'median_full_spread_bps60', 'nominal_close'])
    liq['security'] = liq.security.astype(str)
    if a.universe == 'smallliq':
        uni = pd.read_parquet(dataset('equities.small_liq_indexed_monthly', must_exist=True),
                              columns=['asof_date', 'security', 'eligible_indexed'])
        uni = uni[uni.eligible_indexed].copy()
        uni['security'] = uni.security.astype(str)
        uni = uni.merge(liq.drop(columns='r3000_member'), on=['asof_date', 'security'], how='left')
    else:
        uni = liq[liq.r3000_member].drop(columns='r3000_member').copy()
    log(f'{a.universe}: {len(uni):,} security-months, {uni.asof_date.nunique()} months, '
        f'pool {uni.groupby("asof_date").size().min()}-{uni.groupby("asof_date").size().max()}; per side {a.per_side_usd}')

    sess = pd.read_parquet(root / 'sessions.parquet')
    cal = pd.DatetimeIndex(sess.date)
    months = sorted(uni.asof_date.unique())
    form_pos = {m: int(cal.searchsorted(pd.Timestamp(m), side='right') - 1) for m in months}   # last session <= month-end
    form_date = {m: cal[p] for m, p in form_pos.items()}

    sc = pd.concat([pd.read_parquet(RUN / f'Y{y}' / 'ensemble_scores.parquet', columns=['security', 'date', 'score'],
                                    filters=[('date', 'in', [d for d in form_date.values() if d.year == y])])
                    for y in range(2000, 2027)])
    sc['security'] = sc.security.astype(str)
    uni['date'] = uni.asof_date.map(form_date)
    pool = uni.merge(sc, on=['security', 'date'], how='left')
    cov = pool.groupby('asof_date').agg(eligible=('security', 'size'), scored=('score', 'count')).reset_index()
    pool = pool.dropna(subset=['score'])
    log(f'scored share of eligible names: {cov.scored.sum() / cov.eligible.sum():.3%}')

    secs = sorted(pool.security.unique())
    lo = min(form_pos.values())
    filt = [('security', 'in', secs), ('date', '>=', cal[lo])]
    px = pd.read_parquet(root / 'prices.parquet', columns=['security', 'date', 'open', 'close', 'total_return_index_gross_dividends'], filters=filt)
    q = pd.read_parquet(root / 'quality.parquet', columns=['security', 'date', 'off_exchange_calendar'], filters=filt)
    px = px.merge(q, on=['security', 'date'], validate='one_to_one')
    px = px[~px.off_exchange_calendar]
    px['security'] = px.security.astype(str)
    px = px.merge(sess, on='date', how='inner')
    sid = {k: i for i, k in enumerate(secs)}
    nrow = sess.session_index.max() - lo + 1
    rr, cc = px.session_index.to_numpy() - lo, px.security.map(sid).to_numpy()
    T = np.full((nrow, len(secs)), np.nan)
    T[rr, cc] = px.total_return_index_gross_dividends.to_numpy(float)
    ratio = (px.open / px.close).to_numpy(float)
    okr = np.isfinite(ratio) & (ratio > 0.1) & (ratio < 10)
    O = np.full_like(T, np.nan)
    O[rr[okr], cc[okr]] = T[rr[okr], cc[okr]] * ratio[okr]
    OBS = ~np.isnan(T)
    T = pd.DataFrame(T).ffill().to_numpy()
    del px, q
    log(f'price matrix {T.shape}')

    def leg_return(cols, f_row, nf_row, entry):
        """Equal-weight buy-and-hold return of `cols` from formation row f_row to next formation row nf_row."""
        if entry == 'close':
            e_val, x_val = T[f_row, cols], T[nf_row, cols]
            return float(np.nanmean(x_val / e_val - 1)), 0
        e_row, x_row = f_row + 1, nf_row + 1
        live = OBS[e_row, cols]
        c = cols[live]
        e = O[e_row, c].copy()
        e[~np.isfinite(e)] = T[e_row, c[~np.isfinite(e)]]
        if x_row >= nrow:                                   # end of the frozen data: exit at the last close
            x = T[nrow - 1, c]
        else:
            x = O[x_row, c].copy()
            bad = ~(np.isfinite(x) & OBS[x_row, c])
            x[bad] = T[x_row - 1, c[bad]]
        return float(np.mean(x / e - 1)), int((~live).sum())

    rows, prev = [], {}
    by_month = {m: g for m, g in pool.groupby('asof_date')}
    for i, m in enumerate(months[:-1]):
        g_all = by_month[m].sort_values(['score', 'security'])
        f_row, nf_row = form_pos[m] - lo, form_pos[months[i + 1]] - lo
        for N in NS:
            if a.per_side_usd and N == 'decile':
                continue
            g = g_all if not a.per_side_usd else g_all[g_all.adv60_usd >= 20 * a.per_side_usd / N]
            cols = g.security.map(sid).to_numpy()
            n_pool = len(g)
            k = max(1, n_pool // 10) if N == 'decile' else N
            if 2 * k > n_pool:
                continue
            short_c, long_c = cols[:k], cols[-k:]
            hl, hs = g.iloc[-k:], g.iloc[:k]
            to_l = 1 - len(set(long_c) & prev.get((N, 'L'), set())) / k if (N, 'L') in prev else np.nan
            to_s = 1 - len(set(short_c) & prev.get((N, 'S'), set())) / k if (N, 'S') in prev else np.nan
            prev[(N, 'L')], prev[(N, 'S')] = set(long_c), set(short_c)
            for en in ENTRIES:
                rl, dl_ = leg_return(long_c, f_row, nf_row, en)
                rs, ds_ = leg_return(short_c, f_row, nf_row, en)
                rp, dp_ = leg_return(cols, f_row, nf_row, en)
                rows.append({'N': str(N), 'entry': en, 'month': pd.Timestamp(months[i + 1]), 'formed': pd.Timestamp(m),
                             'long': rl, 'short': rs, 'LS': rl - rs, 'long_vs_pool': rl - rp, 'pool_vs_short': rp - rs, 'pool': rp,
                             'names_per_side': k, 'pool_names': n_pool, 'turnover_long': to_l, 'turnover_short': to_s,
                             'dropped_at_entry': dl_ + ds_,
                             'long_med_adv': hl.adv60_usd.median(), 'short_med_adv': hs.adv60_usd.median(),
                             'long_med_spread_bps': hl.median_full_spread_bps60.median(), 'short_med_spread_bps': hs.median_full_spread_bps60.median(),
                             'long_under5': (hl.nominal_close < 5).mean(), 'short_under5': (hs.nominal_close < 5).mean()})
    mon = pd.DataFrame(rows)
    mon.to_parquet(out / 'monthly.parquet', index=False)
    cov.to_parquet(out / 'coverage.parquet', index=False)
    log(f'{mon.month.nunique()} holding months, {mon.formed.min().date()} to {mon.month.max().date()}')

    st = []
    for (N, en), d in mon.groupby(['N', 'entry']):
        d = d.set_index('month')
        for p, msk in periods(d.index).items():
            for book in ['LS', 'long', 'short', 'long_vs_pool', 'pool_vs_short', 'pool']:
                st.append({'N': N, 'entry': en, 'book': book, 'period': p, **mstats(d.loc[msk, book])})
    stats = pd.DataFrame(st)
    stats.to_parquet(out / 'stats.parquet', index=False)
    order = [str(n) for n in NS if str(n) in set(mon.N)]
    tov = mon[mon.entry == 'open'].groupby('N')[['turnover_long', 'turnover_short', 'pool_names', 'long_med_adv', 'short_med_adv',
                                                 'long_med_spread_bps', 'short_med_spread_bps', 'long_under5', 'short_under5']].median().reindex(order)
    with pd.ExcelWriter(out / 'smallliq_ls.xlsx') as xw:
        pd.DataFrame({'note': [f'Universe: {a.universe} ' + ('(GDM equities.small_liq_indexed_monthly eligible_indexed; market liquidity only; borrow NOT verified)' if a.universe == 'smallliq' else '(every scored point-in-time R3000 member)') + (f'; capacity screen ${a.per_side_usd:,.0f} per side: ADV60 >= 20 x position (1% participation, 5 sessions), fixed dollars' if a.per_side_usd else '') + '.',
                               'Signal: R3000 headline CNN score on the last session on or before each month-end. Long top N, short bottom N, equal weight, monthly.',
                               'open = enter next-session open, exit next month\'s next-session open; close = formation close to next formation close. Gross of all costs and borrow.',
                               'LS = long minus short; long_vs_pool / pool_vs_short = each leg against the eligible pool\'s equal weight; pool = equal-weight eligible pool.']}).to_excel(xw, sheet_name='notes', index=False)
        stats[stats.book == 'LS'].to_excel(xw, sheet_name='LS', index=False)
        stats.to_excel(xw, sheet_name='all_stats', index=False)
        tov.reset_index().to_excel(xw, sheet_name='turnover_and_holdings', index=False)
        cov.to_excel(xw, sheet_name='coverage', index=False)

    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9})
    ls_open = stats[(stats.book == 'LS') & (stats.entry == 'open')].set_index(['N', 'period'])
    with PdfPages(out / 'smallliq_ls.pdf') as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        xs = np.arange(len(order))
        for p, colr in [('Full', '#15211d'), ('2000-2011', '#8a97a8'), ('2012-2026', '#0f7a58'), ('5y', '#b8432f')]:
            axes[0].plot(xs, [ls_open.loc[(n, p), 'sharpe'] for n in order], marker='o', color=colr, label=p)
            axes[1].plot(xs, [ls_open.loc[(n, p), 'ann_return'] * 100 for n in order], marker='o', color=colr, label=p)
        for ax, t in zip(axes, ['Sharpe ratio of long-short', 'Annual return of long-short, % (gross)']):
            ax.set_xticks(xs); ax.set_xticklabels([f'{n}\nper side' if n != 'decile' else '10% of\npool' for n in order], fontsize=8)
            ax.axhline(0, color='black', lw=0.7); ax.grid(alpha=0.25); ax.legend(fontsize=8); ax.set_title(t, fontsize=9)
        fig.suptitle(f'CNN long-short, universe {a.universe}' + (f', ${a.per_side_usd / 1e6:g}m per side capacity screen' if a.per_side_usd else '') + ', by number of stocks per side (next-open entry, monthly)', fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5))
        for n, colr in zip([x for x in ['10', '30', '50', '100', 'decile'] if x in order], ['#b8432f', '#d9a08f', '#0f7a58', '#5fae8d', '#15211d']):
            d = mon[(mon.N == n) & (mon.entry == 'open')].set_index('month').LS
            f = ls_open.loc[(n, 'Full')]
            ax.plot((1 + d).cumprod(), color=colr, lw=1.3, label=f'{n if n != "decile" else "10% of pool"} per side: {f.ann_return:.1%}/yr, Sharpe {f.sharpe:.2f}, worst drawdown {f.max_drawdown:.0%}')
        ax.set_yscale('log'); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc='upper left')
        ax.set_title('Growth of $1, long-short, next-open entry, gross', fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        d = mon[(mon.N == '40') & (mon.entry == 'open')].set_index('month')
        yr = (1 + d[['long_vs_pool', 'pool_vs_short']]).groupby(d.index.year).prod() - 1
        x = np.arange(len(yr))
        ax.bar(x - 0.2, yr.long_vs_pool * 100, 0.4, color='#0f7a58', label='long 40 vs pool')
        ax.bar(x + 0.2, yr.pool_vs_short * 100, 0.4, color='#b8432f', label='pool vs short 40')
        ax.set_xticks(x[::2]); ax.set_xticklabels(yr.index[::2], rotation=45, fontsize=8)
        ax.axhline(0, color='black', lw=0.8); ax.grid(alpha=0.25, axis='y'); ax.legend(fontsize=8)
        ax.set_title('40 per side: each leg against the equal-weight pool, by calendar year, % (2026 to June)', fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    key = stats[(stats.book == 'LS') & stats.period.isin(['Full', '2006-2026', '2000-2011', '2012-2026', '5y', '3y'])]
    json.dump({'out': str(out), 'universe': a.universe, 'per_side_usd': a.per_side_usd, 'scored_share': float(cov.scored.sum() / cov.eligible.sum()),
               'pool_size_min_max': [int(cov.scored.min()), int(cov.scored.max())],
               'LS': key.round(4).to_dict('records'), 'turnover_open': tov.round(3).reset_index().to_dict('records'),
               'elapsed_s': round(time.time() - T0, 1)}, open(out / 'summary.json', 'w'), indent=1, default=str)
    for en in ENTRIES:
        k = key[key.entry == en]
        print(f'--- {en}: Sharpe')
        print(k.pivot_table(index='N', columns='period', values='sharpe').reindex(order).round(2).to_string())
        print(f'--- {en}: annual return')
        print(k.pivot_table(index='N', columns='period', values='ann_return').reindex(order).round(3).to_string())
    f = stats[(stats.entry == 'open') & (stats.period == 'Full')].pivot_table(index='N', columns='book', values=['ann_return', 'sharpe']).reindex(order).round(3)
    print(f.to_string())
    print(stats[(stats.entry == 'open') & (stats.period == 'Full') & (stats.book == 'LS')].set_index('N').reindex(order)[['ann_vol', 'max_drawdown', 'nw_t', 'share_positive']].round(3).to_string())
    print(tov.round(2).to_string())
    log(f'DONE {out}')


if __name__ == '__main__':
    main()
