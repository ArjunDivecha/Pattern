"""
Script: entry_lag.py
Version: 1.0 (2026-10-23)

Description:
    Does the R3000 CNN edge survive a realistic entry? Every backtest so far (report.py, offsets.py)
    enters at the close of the day the chart ends on, the very close the network just read, which a
    trader cannot do exactly. This rebuilds offsets.py's staggered book (a tranche formed every
    session from that day's deciles, held 20 sessions buy-and-hold, equal weight at entry, 20 live
    tranches in equal shares) under three entry rules, everything else unchanged:

      close0  enter at the signal day's close t, exit at close t+20 (the existing convention; must
              reproduce offsets.py's staggered_daily.parquet, asserted)
      open1   compute the signal overnight, enter at the NEXT day's open t+1, exit at the open of
              t+21, when the replacement tranche enters (exactly 20 sessions held). On the rotation
              day the slot earns the old tranche's overnight gap, then the new tranche's intraday.
      close1  skip a full day: enter at close t+1, exit at close t+21.

    The open is priced on the total-return scale as TRI x open / close of that bar (split-adjusted
    open and close from the same row). A name with no bar on t+1 cannot be bought and is dropped
    from that tranche (counted); a bar with an unusable open enters at that day's close (counted).
    After a name's last bar it is held at its last price, as in offsets.py; bankrupt names in the
    size books are written off the session after their last bar, as in offsets.py. The deciles and
    the eligible names (known 20-day outcome at formation) are exactly offsets.py's, so only the
    entry timing differs.

    It also decomposes each tranche's 20-day return into the overnight gap after the signal
    (close t -> open t+1), the first day's intraday (open t+1 -> close t+1) and days 2-20, to show
    where the edge is earned.

    Gross of all costs. Statistics on calendar-month compounded returns (Sharpe x sqrt(12), NW t with
    3 lags), periods Full / 5y / 3y / 1y / 2000-2011 / 2012-2026, as offsets.py.

Input files:
    <run dir>/Y<year>/ensemble_scores.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/index.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010):
        prices.parquet (open, close, total_return_index_gross_dividends), quality.parquet, sessions.parquet,
        delisting_outcomes.parquet; nominal_prices.parquet via mcap.py
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/fundamentals.parquet (via mcap.py)
    GDM security_reference / EDGAR maps (via sector_slice.py)
    <run dir>/offsets_20261015T011336Z/staggered_daily.parquet (reproduction check)
    with <run dir> = /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z
Output files (in <run dir>/entry_lag_<UTC timestamp>/):
    lag_daily.parquet        book x mode x date: daily D10, D1, EW, LS returns
    lag_stats.parquet        book x mode x period x leg: annual return, vol, Sharpe, NW t, CAGR, max drawdown
    decomposition.parquet    book x leg: mean tranche return split into overnight gap, day-1 intraday, days 2-20
    entry_lag.xlsx           the tables
    entry_lag.pdf            charts (light mode)
    summary.json, progress.json; log tee'd by the caller to entry_lag.log

Usage:
    .venv/bin/python research/r3000_headline/entry_lag.py --run-dir DIR

Dependencies: numpy, pandas, matplotlib, openpyxl; offsets.py, prepare.py, verify_fixed.py, mcap.py,
sector_slice.py helpers (unmodified).
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offsets import (ROOT, BOOKS, H, log, write_json, periods, daily_stats, assign_deciles,  # noqa: E402
                     resolve_dataset, market_caps, sector_map)

MODES = ['close0', 'open1', 'close1']
MODE_LABEL = {'close0': 'signal-day close (current backtests)', 'open1': 'next-day open', 'close1': 'next-day close'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--index-dir', default=str(ROOT / 'cache/r3000_I20_M20_v2_index'))
    ap.add_argument('--min-names', type=int, default=100)
    ap.add_argument('--check', default='offsets_20261015T011336Z/staggered_daily.parquet')
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    out = run / ('entry_lag_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    prog = {'stage': 'start', 'out': str(out)}
    write_json(out / 'progress.json', prog)
    root, ref = resolve_dataset()

    # ---- scored panel, deciles and eligibility: identical to offsets.py ----
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
    sc = sc.sort_values(['date', 'security']).reset_index(drop=True)
    sc['mcap'] = market_caps(root, sc)
    sc['cap_rank'] = sc.groupby('date').mcap.rank(ascending=False, method='first')
    sm = sector_map(sorted(sc.security.unique()))
    sc = sc.merge(sm[['security', 'sector']], on='security', how='left')
    uni_mask = {'all': np.ones(len(sc), bool), 'large': (sc.cap_rank <= 1000).to_numpy(), 'small': (sc.cap_rank > 1000).to_numpy()}
    dec = {}
    for uni, m in uni_mask.items():
        g = sc.loc[m, ['date', 'security', 'score', 'sector']]
        r = g.groupby('date').score.rank(pct=True) - 0.5
        sec = r - r.groupby([g.date, g.sector]).transform('mean')
        dec[(uni, 'raw')] = assign_deciles(g.assign(s=g.score), 's')
        dec[(uni, 'sec')] = assign_deciles(g.assign(s=sec), 's')
    log(f'scored rows {len(sc):,}; deciles built')

    # ---- close and open on the total-return scale ----
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
    sec_id = {s: i for i, s in enumerate(secs)}
    rr, cc = px.session_index.to_numpy() - lo, px.security.map(sec_id).to_numpy()
    T = np.full((hi - lo + 1, len(secs)), np.nan)
    T[rr, cc] = px.total_return_index_gross_dividends.to_numpy(float)
    ratio = (px.open / px.close).to_numpy(float)
    ok_open = np.isfinite(ratio) & (ratio > 0) & (ratio < 10) & (ratio > 0.1)
    O = np.full_like(T, np.nan)
    O[rr[ok_open], cc[ok_open]] = T[rr[ok_open], cc[ok_open]] * ratio[ok_open]
    OBS = ~np.isnan(T)
    T = pd.DataFrame(T).ffill().to_numpy()
    bad_open_bars = int((~ok_open).sum())
    del px, q
    cal = sess.set_index('session_index').date
    dts_all = pd.DatetimeIndex(cal.reindex(np.arange(T.shape[0]) + lo).to_numpy())
    log(f'TRI matrix {T.shape}; bars with unusable open: {bad_open_bars:,}')

    sc['col'] = sc.security.map(sec_id).to_numpy()
    sc['row'] = sc.session_position.to_numpy() - lo
    nrow = T.shape[0]
    daily, decomp, counts = {}, [], {}
    for book, (uni, sig, col) in BOOKS.items():
        m = uni_mask[uni]
        g = sc.loc[m, ['date', 'row', 'col', col]].assign(dec=dec[(uni, sig)])
        cnt = g.groupby('date').size()
        g = g[g.date.map(cnt) >= a.min_names].dropna(subset=[col])
        legs = {'D10': g[g.dec == 10], 'D1': g[g.dec == 1], 'EW': g}
        acc = {md: {} for md in MODES}
        nc = {'dropped_no_bar_t1': 0, 'open_fallback_to_close': 0, 'names': 0}
        for leg, gl in legs.items():
            full_sum = {md: np.zeros(nrow) for md in MODES}
            full_n = {md: np.zeros(nrow) for md in MODES}
            intra, over = np.full(nrow, np.nan), np.full(nrow, np.nan)
            parts = []
            for (row0,), gg in gl.groupby(['row']):
                if row0 + H + 1 >= nrow:
                    continue
                cols = gg['col'].to_numpy()
                bk = np.flatnonzero(gg[col].to_numpy() == -1.0) if col == 'ret_bk_zero' else np.array([], int)

                def writeoff(path, r_start):
                    for j in bk:
                        seen = np.flatnonzero(OBS[r_start:r_start + path.shape[0], cols[j]])
                        if len(seen):
                            path[seen[-1] + 1:, j] = 0.0
                    return path

                # close0: rows row0..row0+H
                p0 = writeoff(T[row0:row0 + H + 1, cols] / T[row0, cols], row0)
                v0 = p0.mean(axis=1)
                full_sum['close0'][row0 + 1:row0 + H + 1] += v0[1:] / v0[:-1] - 1
                full_n['close0'][row0 + 1:row0 + H + 1] += 1
                # names that can be bought on t+1
                live = OBS[row0 + 1, cols]
                if leg == 'EW':
                    nc['dropped_no_bar_t1'] += int((~live).sum())
                    nc['names'] += len(cols)
                if not live.any():
                    continue
                c1, bk1 = cols[live], np.flatnonzero(np.isin(np.arange(len(cols))[live], bk))
                # close1: rows row0+1..row0+1+H
                p1 = T[row0 + 1:row0 + H + 2, c1] / T[row0 + 1, c1]
                for j in bk1:
                    seen = np.flatnonzero(OBS[row0 + 1:row0 + H + 2, c1[j]])
                    p1[seen[-1] + 1:, j] = 0.0
                v1 = p1.mean(axis=1)
                full_sum['close1'][row0 + 2:row0 + H + 2] += v1[1:] / v1[:-1] - 1
                full_n['close1'][row0 + 2:row0 + H + 2] += 1
                # open1: entry at open t+1, closes t+1..t+H, exit at open t+H+1
                e = O[row0 + 1, c1].copy()
                fb = ~np.isfinite(e)
                if leg == 'EW':
                    nc['open_fallback_to_close'] += int(fb.sum())
                e[fb] = T[row0 + 1, c1[fb]]
                xo = O[row0 + H + 1, c1].copy()
                xo_ok = np.isfinite(xo) & OBS[row0 + H + 1, c1]
                xo[~xo_ok] = T[row0 + H, c1[~xo_ok]]          # no bar at t+H+1: still at last price
                closes = T[row0 + 1:row0 + H + 1, c1]
                pth = np.vstack([np.ones(len(c1)), closes / e, xo / e])   # entry, H closes, exit open
                for j in bk1:
                    seen = np.flatnonzero(OBS[row0 + 1:row0 + H + 2, c1[j]])
                    pth[1 + seen[-1] + 1:, j] = 0.0
                vo = pth.mean(axis=1)
                intra[row0 + 1] = vo[1] - 1                                  # open t+1 -> close t+1
                full_sum['open1'][row0 + 2:row0 + H + 1] += vo[2:H + 1] / vo[1:H] - 1
                full_n['open1'][row0 + 2:row0 + H + 1] += 1
                over[row0 + H + 1] = vo[H + 1] / vo[H] - 1                   # close t+H -> open t+H+1
                # decomposition of the close0 tranche return (names live on t+1)
                gap = np.where(np.isfinite(O[row0 + 1, c1]), O[row0 + 1, c1], T[row0 + 1, c1]) / T[row0, c1]
                parts.append((row0, (gap - 1).mean(), (T[row0 + 1, c1] / np.where(np.isfinite(O[row0 + 1, c1]), O[row0 + 1, c1], T[row0 + 1, c1]) - 1).mean(),
                              (p0[-1][live] / p0[1][live] - 1).mean(), v0[-1] - 1))
            for md in ['close0', 'close1']:
                acc[md][leg] = np.where(full_n[md] == H, full_sum[md] / np.maximum(full_n[md], 1), np.nan)
            slot = (1 + over) * (1 + intra) - 1
            acc['open1'][leg] = np.where((full_n['open1'] == H - 1) & np.isfinite(slot), (full_sum['open1'] + slot) / H, np.nan)
            dp = pd.DataFrame(parts, columns=['row', 'overnight_gap', 'day1_intraday', 'days2_20', 'total_close0'])
            decomp.append({'book': book, 'leg': leg, **{k: float(dp[k].mean()) for k in ['overnight_gap', 'day1_intraday', 'days2_20', 'total_close0']}})
        counts[book] = nc
        for md in MODES:
            df = pd.DataFrame({leg: acc[md][leg] for leg in ['D10', 'D1', 'EW']}, index=dts_all).dropna()
            df['LS'] = df.D10 - df.D1
            daily[(book, md)] = df
        log(f'{book}: close0 {len(daily[(book, "close0")]):,} days, open1 {len(daily[(book, "open1")]):,}, close1 {len(daily[(book, "close1")]):,}; {nc}')
        prog[book] = 'done'
        write_json(out / 'progress.json', prog)

    # ---- reproduction check: close0 == offsets.py staggered book ----
    ref_d = pd.read_parquet(run / a.check)
    check = {}
    for book in BOOKS:
        r = ref_d[ref_d.book == book].set_index('date')[['D10', 'D1', 'EW', 'LS']]
        mine = daily[(book, 'close0')].reindex(r.index)
        check[book] = float(np.nanmax(np.abs(mine.to_numpy() - r.to_numpy())))
        assert len(daily[(book, 'close0')]) == len(r), (book, len(daily[(book, 'close0')]), len(r))
    log(f'close0 reproduces offsets.py staggered book: max abs diff {max(check.values()):.2e}')
    assert max(check.values()) < 1e-10, check

    # ---- common sample: dates present in all three modes ----
    rows, lag_rows = [], []
    for book in BOOKS:
        common = daily[(book, 'close0')].index
        for md in MODES[1:]:
            common = common.intersection(daily[(book, md)].index)
        for md in MODES:
            df = daily[(book, md)]
            lag_rows.append(df.assign(book=book, mode=md).rename_axis('date').reset_index())
            dfc = df.loc[common]
            for p, msk in periods(dfc.index).items():
                for leg in ['LS', 'D10', 'D1', 'EW']:
                    rows.append({'book': book, 'mode': md, 'period': p, 'leg': leg, 'first': str(dfc.index.min().date()),
                                 'last': str(dfc.index.max().date()), **daily_stats(dfc.loc[msk, leg])})
    stats = pd.DataFrame(rows)
    pd.concat(lag_rows, ignore_index=True).to_parquet(out / 'lag_daily.parquet', index=False)
    stats.to_parquet(out / 'lag_stats.parquet', index=False)
    dcp = pd.DataFrame(decomp)
    for k in ['overnight_gap', 'day1_intraday', 'days2_20', 'total_close0']:
        dcp[k + '_ann'] = dcp[k] * 252 / H
    dcp.to_parquet(out / 'decomposition.parquet', index=False)
    with pd.ExcelWriter(out / 'entry_lag.xlsx') as xw:
        pd.DataFrame({'note': ['Gross of all costs. Staggered book as offsets.py; only the entry timing differs.',
                               'close0 = enter at the signal-day close (existing backtests); open1 = enter next-day open, exit open 20 sessions later; close1 = enter next-day close.',
                               'Statistics on a common set of dates across the three modes, calendar-month returns.',
                               'decomposition: mean tranche return (per 20 sessions, and x252/20 annualised) split into the overnight gap after the signal, the first day intraday, and days 2-20.']}).to_excel(xw, sheet_name='notes', index=False)
        stats[stats.leg == 'LS'].to_excel(xw, sheet_name='LS_stats', index=False)
        stats.to_excel(xw, sheet_name='all_stats', index=False)
        dcp.to_excel(xw, sheet_name='decomposition', index=False)

    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9})
    colors = {'close0': '#15211d', 'open1': '#0f7a58', 'close1': '#b8432f'}
    with PdfPages(out / 'entry_lag.pdf') as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, book in zip(axes.T.ravel(), BOOKS):
            for md in MODES:
                st = stats[(stats.book == book) & (stats['mode'] == md) & (stats.period == 'Full') & (stats.leg == 'LS')].iloc[0]
                x = daily[(book, md)].LS
                ax.plot((1 + x).cumprod(), color=colors[md], lw=1.2, label=f'{MODE_LABEL[md]}: {st.ann_return:.1%}/yr, Sharpe {st.sharpe:.2f}')
            ax.set_yscale('log')
            ax.set_title(f'{book}: top-minus-bottom decile, growth of $1', fontsize=8.5)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(alpha=0.25)
        fig.suptitle('Entry timing: staggered book, gross, 2000-2026', fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sub = dcp[dcp.leg != 'EW']
        piv = {k: sub.pivot(index='book', columns='leg', values=k + '_ann') for k in ['overnight_gap', 'day1_intraday', 'days2_20']}
        ls = pd.DataFrame({k: v['D10'] - v['D1'] for k, v in piv.items()}).reindex(list(BOOKS))
        ls.plot.bar(ax=ax, color=['#b8432f', '#d9a08f', '#0f7a58'], width=0.75)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_ylabel('top-minus-bottom decile, annualised')
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title('Where the 20-day edge is earned: overnight gap after the signal, first-day intraday, days 2-20 (signal-close entry)', fontsize=9)
        ax.set_xticklabels(ls.index, rotation=20, ha='right', fontsize=8)
        ax.legend(['overnight gap (close t to open t+1)', 'first day intraday', 'days 2 to 20'], fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    key = stats[(stats.leg == 'LS') & (stats.period.isin(['Full', '2000-2011', '2012-2026', '5y', '3y']))]
    write_json(out / 'summary.json', {
        'out': str(out), 'reproduction_max_abs_diff': check, 'bars_with_unusable_open': bad_open_bars, 'counts': counts,
        'LS': key[['book', 'mode', 'period', 'ann_return', 'ann_vol', 'sharpe', 'nw_t', 'max_drawdown']].round(4).to_dict('records'),
        'decomposition': dcp.round(5).to_dict('records')})
    prog['stage'] = 'done'
    write_json(out / 'progress.json', prog)
    print(key[key.book.isin(['all names, raw', 'all names, sector-neutral', 'large caps, raw', 'small caps, raw'])]
          .pivot_table(index=['book', 'period'], columns='mode', values='sharpe').round(2).to_string())
    print(key[(key.period == 'Full')].pivot_table(index='book', columns='mode', values='ann_return').round(4).to_string())
    print(dcp[['book', 'leg', 'overnight_gap_ann', 'day1_intraday_ann', 'days2_20_ann', 'total_close0_ann']].round(4).to_string())
    log('DONE')


if __name__ == '__main__':
    main()
