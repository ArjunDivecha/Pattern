"""
Script: short_bottom1.py
Version: 1.0 (2026-11-01)

Description:
    Backtest of a short-only book on the CNN's bottom 1%. Every session, the Russell 3000 members
    whose CNN score ranks in the bottom 1% of that day's scores (about 30 names) form a tranche, equal
    weight at entry, held 20 sessions buy-and-hold; the book holds the 20 live tranches in equal shares
    (the staggered construction of offsets.py). The short book's daily return is minus the basket's
    (constant short exposure per tranche, rebalanced daily, as offsets.py's long-short).

    Two entries: next-day open (realistic: signal computed overnight; enter at the open of t+1, exit at
    the open of t+21, exactly as entry_lag.py's open1) and the signal-day close (the research
    convention). Two treatments of names that delist inside the 20 days: held at the last price
    (cautious for a short) and, for bankrupt names, written off to zero the session after the last bar
    (the short gains 100%).

    Books reported, all gross of costs and borrow:
      short_only     = - bottom-1% basket
      short_hedged   = equal-weight universe - bottom-1% basket (short the losers, long the average stock)
      bottom1_long   = the bottom-1% basket itself, and EW = the equal-weight universe (benchmark)
    Statistics on calendar-month returns over Full / 5y / 3y / 1y / 2000-2011 / 2012-2026, as offsets.py.
    The equal-weight leg at the signal-day close must reproduce offsets.py's staggered EW (asserted).

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z/Y<year>/ensemble_scores.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/index.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010: prices, quality,
        sessions, delisting_outcomes; nominal_prices via mcap.py)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/fundamentals.parquet (via mcap.py)
    <run dir>/offsets_20261015T011336Z/staggered_daily.parquet (reproduction check)
Output files (in <run dir>/short_bottom1_<UTC timestamp>/):
    daily.parquet     variant x entry x date: bottom-1% basket, EW, short_only, short_hedged daily returns
    stats.parquet     variant x entry x book x period stats; short_bottom1.xlsx; short_bottom1.pdf (light mode)
    profile.parquet   what the shorts are: names per tranche, small-cap share, under-$5 share, median market cap, by year
    summary.json; log tee'd by the caller

Usage:
    .venv/bin/python research/r3000_headline/short_bottom1.py

Dependencies: numpy, pandas, matplotlib, openpyxl; offsets.py helpers, mcap.py.
"""
from pathlib import Path
from datetime import datetime, timezone
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offsets import ROOT, H, log, write_json, periods, daily_stats, resolve_dataset, market_caps  # noqa: E402

RUN = ROOT / 'research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z'
INDEX = ROOT / 'cache/r3000_I20_M20_v2_index/index.parquet'
CUT = 0.01
ENTRIES = {'open1': 'next-day open', 'close0': 'signal-day close'}
VARIANTS = {'last_price': 'delisted held at last price', 'bankrupt_zero': 'bankrupt names written off to zero'}


def main():
    out = RUN / ('short_bottom1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    prog = {'stage': 'start'}
    write_json(out / 'progress.json', prog)
    root, _ = resolve_dataset()
    sc = pd.concat([pd.read_parquet(RUN / f'Y{y}' / 'ensemble_scores.parquet', columns=['row_id', 'security', 'date', 'session_position', 'score']) for y in range(2000, 2027)])
    idx = pd.read_parquet(INDEX, columns=['row_id', 'forward_return_20', 'terminal_within_20', 'return_to_last'])
    sc = sc.merge(idx, on='row_id', validate='one_to_one')
    sc['security'] = sc.security.astype(str)
    dl = pd.read_parquet(root / 'delisting_outcomes.parquet', columns=['security', 'outcome'])
    dl['security'] = dl.security.astype(str)
    sc = sc.merge(dl, on='security', how='left')
    known = sc.forward_return_20.notna()
    term = sc.terminal_within_20 & sc.return_to_last.notna() & ~known
    sc['ret_last'] = np.where(known, sc.forward_return_20, np.where(term, sc.return_to_last, np.nan))
    sc['bk'] = term & sc.outcome.eq('bankrupt')
    sc = sc.sort_values(['date', 'security']).reset_index(drop=True)
    sc['mcap'] = market_caps(root, sc)
    sc['cap_rank'] = sc.groupby('date').mcap.rank(ascending=False, method='first')
    # eligibility exactly as offsets.py's 'all names, raw' book: known (or terminal) outcome, >= 100 names
    sc = sc.dropna(subset=['ret_last'])
    cnt = sc.groupby('date').size()
    sc = sc[sc.date.map(cnt) >= 100].reset_index(drop=True)
    sc['pct'] = sc.groupby('date').score.rank(pct=True, method='first')
    sc['short'] = sc.pct <= CUT
    log(f'eligible rows {len(sc):,}; bottom-1% rows {int(sc.short.sum()):,}')

    # what the shorts are
    nom = pd.read_parquet(root / 'nominal_prices.parquet', columns=['security', 'date', 'close_nominal'],
                          filters=[('date', '>=', sc.date.min())])
    nom['security'] = nom.security.astype(str)
    s = sc[sc.short].merge(nom, on=['security', 'date'], how='left')
    prof = s.groupby(s.date.dt.year).agg(names_per_day=('security', lambda x: len(x)), small_cap_share=('cap_rank', lambda x: (x > 1000).mean()),
                                         under_5_dollars=('close_nominal', lambda x: (x < 5).mean()), median_mcap_musd=('mcap', 'median'),
                                         bankrupt_share=('bk', 'mean'))
    prof['names_per_day'] = prof.names_per_day / s.groupby(s.date.dt.year).date.nunique()
    prof.to_parquet(out / 'profile.parquet')
    log('profile:\n' + prof.round(3).to_string())

    # prices: close and open on the total-return scale
    secs = sorted(sc.security.unique())
    sess = pd.read_parquet(root / 'sessions.parquet')
    lo, hi = sc.session_position.min(), sess.session_index.max()
    filt = [('security', 'in', secs), ('date', '>=', sc.date.min())]
    px = pd.read_parquet(root / 'prices.parquet', columns=['security', 'date', 'open', 'close', 'total_return_index_gross_dividends'], filters=filt)
    q = pd.read_parquet(root / 'quality.parquet', columns=['security', 'date', 'off_exchange_calendar'], filters=filt)
    px = px.merge(q, on=['security', 'date'], validate='one_to_one')
    px = px[~px.off_exchange_calendar]
    px['security'] = px.security.astype(str)
    px = px.merge(sess, on='date', how='inner')
    sec_id = {k: i for i, k in enumerate(secs)}
    rr, cc = px.session_index.to_numpy() - lo, px.security.map(sec_id).to_numpy()
    T = np.full((hi - lo + 1, len(secs)), np.nan)
    T[rr, cc] = px.total_return_index_gross_dividends.to_numpy(float)
    ratio = (px.open / px.close).to_numpy(float)
    okr = np.isfinite(ratio) & (ratio > 0.1) & (ratio < 10)
    O = np.full_like(T, np.nan)
    O[rr[okr], cc[okr]] = T[rr[okr], cc[okr]] * ratio[okr]
    OBS = ~np.isnan(T)
    T = pd.DataFrame(T).ffill().to_numpy()
    del px, q, nom
    nrow = T.shape[0]
    dts = pd.DatetimeIndex(sess.set_index('session_index').date.reindex(np.arange(nrow) + lo).to_numpy())
    sc['col'] = sc.security.map(sec_id).to_numpy()
    sc['row'] = sc.session_position.to_numpy() - lo
    log(f'TRI matrix {T.shape}')

    daily = {}
    for var in VARIANTS:
        legs = {'B1': sc[sc.short], 'EW': sc}
        res = {en: {} for en in ENTRIES}
        for leg, gl in legs.items():
            fs = {en: np.zeros(nrow) for en in ENTRIES}
            fn = {en: np.zeros(nrow) for en in ENTRIES}
            intra, over = np.full(nrow, np.nan), np.full(nrow, np.nan)
            for (row0,), gg in gl.groupby(['row']):
                if row0 + H >= nrow:
                    continue
                cols = gg['col'].to_numpy()
                bk = np.flatnonzero(gg.bk.to_numpy()) if var == 'bankrupt_zero' else np.array([], int)
                p0 = T[row0:row0 + H + 1, cols] / T[row0, cols]
                for j in bk:
                    seen = np.flatnonzero(OBS[row0:row0 + H + 1, cols[j]])
                    p0[seen[-1] + 1:, j] = 0.0
                v0 = p0.mean(axis=1)
                fs['close0'][row0 + 1:row0 + H + 1] += v0[1:] / v0[:-1] - 1
                fn['close0'][row0 + 1:row0 + H + 1] += 1
                if row0 + H + 1 >= nrow:
                    continue
                live = OBS[row0 + 1, cols]
                if not live.any():
                    continue
                c1 = cols[live]
                bk1 = np.flatnonzero(np.isin(np.flatnonzero(live), bk))
                e = O[row0 + 1, c1].copy()
                fb = ~np.isfinite(e)
                e[fb] = T[row0 + 1, c1[fb]]
                xo = O[row0 + H + 1, c1].copy()
                xok = np.isfinite(xo) & OBS[row0 + H + 1, c1]
                xo[~xok] = T[row0 + H, c1[~xok]]
                pth = np.vstack([np.ones(len(c1)), T[row0 + 1:row0 + H + 1, c1] / e, xo / e])
                for j in bk1:
                    seen = np.flatnonzero(OBS[row0 + 1:row0 + H + 2, c1[j]])
                    pth[1 + seen[-1] + 1:, j] = 0.0
                vo = pth.mean(axis=1)
                intra[row0 + 1] = vo[1] - 1
                fs['open1'][row0 + 2:row0 + H + 1] += vo[2:H + 1] / vo[1:H] - 1
                fn['open1'][row0 + 2:row0 + H + 1] += 1
                over[row0 + H + 1] = vo[H + 1] / vo[H] - 1
            res['close0'][leg] = np.where(fn['close0'] == H, fs['close0'] / np.maximum(fn['close0'], 1), np.nan)
            slot = (1 + over) * (1 + intra) - 1
            res['open1'][leg] = np.where((fn['open1'] == H - 1) & np.isfinite(slot), (fs['open1'] + slot) / H, np.nan)
        for en in ENTRIES:
            df = pd.DataFrame({'bottom1_long': res[en]['B1'], 'EW': res[en]['EW']}, index=dts).dropna()
            df['short_only'] = -df.bottom1_long
            df['short_hedged'] = df.EW - df.bottom1_long
            daily[(var, en)] = df
        log(f'{var} done')
        prog[var] = 'done'
        write_json(out / 'progress.json', prog)

    ref = pd.read_parquet(RUN / 'offsets_20261015T011336Z' / 'staggered_daily.parquet')
    ref = ref[ref.book == 'all names, raw'].set_index('date').EW
    mine = daily[('last_price', 'close0')].EW.reindex(ref.index)
    diff = float(np.nanmax(np.abs(mine.to_numpy() - ref.to_numpy())))
    log(f'EW leg reproduces offsets.py staggered EW: max abs diff {diff:.2e}, days {mine.notna().sum()} vs {len(ref)}')
    assert diff < 1e-10 and mine.notna().all(), diff

    rows, dd = [], []
    for (var, en), df in daily.items():
        common = daily[(var, 'open1')].index.intersection(daily[(var, 'close0')].index)
        dfc = df.loc[common]
        dd.append(df.assign(variant=var, entry=en).rename_axis('date').reset_index())
        for p, msk in periods(dfc.index).items():
            for book in ['short_only', 'short_hedged', 'bottom1_long', 'EW']:
                rows.append({'variant': var, 'entry': en, 'book': book, 'period': p, **daily_stats(dfc.loc[msk, book])})
    stats = pd.DataFrame(rows)
    pd.concat(dd, ignore_index=True).to_parquet(out / 'daily.parquet', index=False)
    stats.to_parquet(out / 'stats.parquet', index=False)
    with pd.ExcelWriter(out / 'short_bottom1.xlsx') as xw:
        pd.DataFrame({'note': ['Gross of all costs, including stock borrow. Staggered book: a tranche of the bottom 1% each session, held 20 sessions, 20 tranches in equal shares.',
                               'short_only = minus the bottom-1% basket; short_hedged = equal-weight universe minus the basket. Calendar-month statistics.',
                               'open1 = enter next-day open, exit 20 sessions later at the open; close0 = signal-day close (research convention).',
                               'last_price = delisted names held at the last price; bankrupt_zero = bankrupt names written off to zero (short gains 100%).']}).to_excel(xw, sheet_name='notes', index=False)
        stats.to_excel(xw, sheet_name='stats', index=False)
        prof.reset_index().to_excel(xw, sheet_name='what_is_shorted', index=False)

    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9})
    main_df = daily[('last_price', 'open1')]
    with PdfPages(out / 'short_bottom1.pdf') as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for book, colr, lab in [('short_hedged', '#0f7a58', 'Short bottom 1%, long the average stock'), ('short_only', '#b8432f', 'Short-only bottom 1%'),
                                ('EW', '#8a97a8', 'Average stock (equal weight), long')]:
            st = stats[(stats.variant == 'last_price') & (stats.entry == 'open1') & (stats.book == book) & (stats.period == 'Full')].iloc[0]
            axes[0].plot((1 + main_df[book]).cumprod(), color=colr, lw=1.4, label=f'{lab}: {st.ann_return:.1%}/yr, Sharpe {st.sharpe:.2f}, worst drawdown {st.max_drawdown:.0%}')
        axes[0].set_yscale('log'); axes[0].grid(alpha=0.25); axes[0].legend(fontsize=7.5, loc='upper left')
        axes[0].set_title('Growth of $1, next-day-open entry, delisted names at last price, gross', fontsize=9)
        yr = (1 + main_df[['short_only', 'short_hedged']]).groupby(main_df.index.year).prod() - 1
        xx = np.arange(len(yr))
        axes[1].bar(xx - 0.2, yr.short_only * 100, 0.4, color='#b8432f', label='short-only')
        axes[1].bar(xx + 0.2, yr.short_hedged * 100, 0.4, color='#0f7a58', label='short + long average stock')
        axes[1].set_xticks(xx[::2]); axes[1].set_xticklabels(yr.index[::2], rotation=45, fontsize=7.5)
        axes[1].axhline(0, color='black', lw=0.8); axes[1].grid(alpha=0.25, axis='y'); axes[1].legend(fontsize=8)
        axes[1].set_title('Calendar-year return, % (2026 to June)', fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    write_json(out / 'summary.json', {'out': str(out), 'ew_reproduction_max_abs_diff': diff,
                                      'stats_full_and_halves': stats[stats.period.isin(['Full', '2000-2011', '2012-2026', '5y', '3y', '1y'])].round(4).to_dict('records'),
                                      'profile': prof.round(4).reset_index().to_dict('records')})
    prog['stage'] = 'done'
    write_json(out / 'progress.json', prog)
    key = stats[stats.book.isin(['short_only', 'short_hedged', 'EW'])]
    print(key.pivot_table(index=['variant', 'entry', 'period'], columns='book', values='ann_return').round(4).to_string())
    print(key.pivot_table(index=['variant', 'entry', 'period'], columns='book', values='sharpe').round(2).to_string())
    print(key[key.period == 'Full'][['variant', 'entry', 'book', 'ann_vol', 'max_drawdown', 'nw_t', 'cagr']].round(3).to_string(index=False))
    log(f'DONE {out}')


if __name__ == '__main__':
    main()
