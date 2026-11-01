"""
Script: stress_100.py
Version: 1.0 (2026-11-01)

Description:
    Stress test of the CNN long-short book found in smallliq_ls.py v1.1: full point-in-time Russell 3000,
    long the 100 highest and short the 100 lowest CNN scores, equal weight, rebalanced monthly, entered at
    the next session's open, with a $1m-per-side capacity screen (60-session mean dollar volume at least
    20x the $10,000 position, i.e. 1% participation over 5 sessions). Gross of all costs; no cost or
    borrow penalty is charged (house rule) - borrow is reported only as a break-even fee.

    Tests:
      1. Rebalance day. The book is rebuilt on each of 20 schedules: formation on the k-th session after
         each month-end (k = 0..19), using the universe and liquidity of the latest month-end on or before
         formation and the CNN score of the formation session; enter at the next open, exit at the open
         after the next formation. Reports the spread of results across the 20 schedules and the
         schedule-averaged book (mean of the 20 schedules' returns for the same formation month; holding
         windows differ by up to 19 sessions, so this is an approximation of a 20-tranche book).
      2. Junk filters: shorts only among names with a nominal close >= $5 at the month-end; both legs >= $5.
      3. Scale: the same book at $5m, $10m and $25m per side (ADV60 >= $1m, $2m, $5m).
      4. Factor alpha of the schedule-averaged book: Fama-French 5 factors + momentum (Ken French), and
         that model plus JKP short-term reversal (ret_1_0), max daily return (rmax1_21d) and idiosyncratic
         volatility (ivol_capm_21d), value-weighted US factors. Newey-West t (3 lags).
      5. Behaviour: up vs down market months, rolling 36-month Sharpe, worst drawdowns, calendar years.
      6. Borrow break-even: the annual fee on all shorts, or on the sub-$5 shorts only, that would erase the
         book's average return.

Input files:
    gdm "equities.small_liq_monthly" (r3000_member, adv60_usd, nominal_close; all members, month-ends)
    gdm "factors.french_monthly", gdm "jkp.usa_factors_vw_cap"
        (/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/...; read via gdm.dataset)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z/Y<year>/ensemble_scores.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010: prices, quality, sessions)
Output files (in <run dir>/stress_100_<UTC timestamp>/):
    monthly.parquet (variant x schedule x month), schedule_stats.parquet, averaged_stats.parquet,
    factor_regressions.parquet, drawdowns.parquet, years.parquet, stress_100.xlsx, stress_100.pdf (light mode),
    summary.json; log tee'd by the caller to verify_logs/stress_100_v1.0.log

Usage:
    .venv/bin/python research/r3000_headline/stress_100.py

Dependencies: numpy, pandas, matplotlib, openpyxl; gdm; prepare.resolve_dataset, verify_fixed.nw_t.
"""
from pathlib import Path
from datetime import datetime, timezone
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
N = 100
OFFSETS = range(20)
VARIANTS = {  # name: (per-side capital, short $5 floor, both-leg $5 floor)
    'base_1m': (1e6, False, False),
    'short_floor5': (1e6, True, False),
    'both_floor5': (1e6, False, True),
    'cap_5m': (5e6, False, False),
    'cap_10m': (10e6, False, False),
    'cap_25m': (25e6, False, False),
}
LABEL = {'base_1m': '$1m/side (base)', 'short_floor5': 'shorts >= $5', 'both_floor5': 'both legs >= $5',
         'cap_5m': '$5m/side', 'cap_10m': '$10m/side', 'cap_25m': '$25m/side'}
T0 = time.time()


def log(m):
    print(f'[{time.time() - T0:7.1f}s] {m}', flush=True)


def mstats(x):
    x = x.dropna()
    if len(x) < 12:
        return {'months': len(x)}
    w = (1 + x).cumprod()
    return {'months': len(x), 'ann_return': x.mean() * 12, 'ann_vol': x.std() * np.sqrt(12), 'sharpe': x.mean() / x.std() * np.sqrt(12),
            'nw_t': nw_t(x, lags=3), 'cagr': w.iloc[-1] ** (12 / len(x)) - 1, 'max_drawdown': float((w / w.cummax() - 1).min()),
            'share_positive': (x > 0).mean()}


def periods(idx):
    last = idx.max()
    return {'Full': idx == idx, '2000-2011': idx.year <= 2011, '2012-2026': idx.year >= 2012, '5y': idx > last - pd.DateOffset(years=5),
            '3y': idx > last - pd.DateOffset(years=3), '1y': idx > last - pd.DateOffset(years=1)}


def ols_nw(y, X, lags=3):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n = len(y)
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    return b, b / np.sqrt(np.diag(V)), 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))


def main():
    out = RUN / ('stress_100_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    root, _ = resolve_dataset()
    liq = pd.read_parquet(dataset('equities.small_liq_monthly', must_exist=True), columns=['asof_date', 'security', 'r3000_member', 'adv60_usd', 'nominal_close'])
    liq = liq[liq.r3000_member].drop(columns='r3000_member')
    liq['security'] = liq.security.astype(str)
    sess = pd.read_parquet(root / 'sessions.parquet')
    cal = pd.DatetimeIndex(sess.date)
    months = sorted(liq.asof_date.unique())
    me_pos = np.array([int(cal.searchsorted(pd.Timestamp(m), side='right') - 1) for m in months])
    liq_by = {m: g.set_index('security') for m, g in liq.groupby('asof_date')}
    sc = pd.concat([pd.read_parquet(RUN / f'Y{y}' / 'ensemble_scores.parquet', columns=['security', 'date', 'score']) for y in range(2000, 2027)])
    sc['security'] = sc.security.astype(str)
    sc_by = {d: g.set_index('security').score for d, g in sc.groupby('date')}
    log(f'{len(months)} month-ends, {len(sc):,} scores')

    secs = sorted(sc.security.unique())
    lo = int(me_pos.min())
    filt = [('security', 'in', secs), ('date', '>=', cal[lo])]
    px = pd.read_parquet(root / 'prices.parquet', columns=['security', 'date', 'open', 'close', 'total_return_index_gross_dividends'], filters=filt)
    q = pd.read_parquet(root / 'quality.parquet', columns=['security', 'date', 'off_exchange_calendar'], filters=filt)
    px = px.merge(q, on=['security', 'date'], validate='one_to_one')
    px = px[~px.off_exchange_calendar]
    px['security'] = px.security.astype(str)
    px = px.merge(sess, on='date', how='inner')
    sid = {k: i for i, k in enumerate(secs)}
    nrow = int(sess.session_index.max()) - lo + 1
    rr, cc = px.session_index.to_numpy() - lo, px.security.map(sid).to_numpy()
    T = np.full((nrow, len(secs)), np.nan)
    T[rr, cc] = px.total_return_index_gross_dividends.to_numpy(float)
    ratio = (px.open / px.close).to_numpy(float)
    okr = np.isfinite(ratio) & (ratio > 0.1) & (ratio < 10)
    O = np.full_like(T, np.nan)
    O[rr[okr], cc[okr]] = T[rr[okr], cc[okr]] * ratio[okr]
    OBS = ~np.isnan(T)
    T = pd.DataFrame(T).ffill().to_numpy()
    del px, q, sc
    log(f'price matrix {T.shape}')

    def leg(cols, e_row, x_row):
        live = OBS[e_row, cols]
        c = cols[live]
        e = O[e_row, c].copy()
        e[~np.isfinite(e)] = T[e_row, c[~np.isfinite(e)]]
        if x_row >= nrow:
            x = T[nrow - 1, c]
        else:
            x = O[x_row, c].copy()
            bad = ~(np.isfinite(x) & OBS[x_row, c])
            x[bad] = T[x_row - 1, c[bad]]
        return float(np.mean(x / e - 1))

    rows = []
    for k in OFFSETS:
        f = me_pos + k - lo                                   # formation rows
        for j in range(len(months) - 1):
            if f[j] >= nrow or f[j] + lo >= len(cal):
                continue
            fdate = cal[f[j] + lo]
            s = sc_by.get(fdate)
            if s is None:
                continue
            u = liq_by[months[j]].join(s, how='inner').dropna(subset=['score'])
            e_row, x_row = f[j] + 1, f[j + 1] + 1
            if e_row >= nrow:
                continue
            for var, (cap, sfloor, bfloor) in VARIANTS.items():
                g = u[u.adv60_usd >= 20 * cap / N]
                if bfloor:
                    g = g[g.nominal_close >= 5]
                g = g.sort_values('score')
                gs = g[g.nominal_close >= 5] if sfloor else g
                if len(g) < 2 * N or len(gs) < N:
                    continue
                lc = g.index[-N:].map(sid).to_numpy()
                scl = gs.index[:N].map(sid).to_numpy()
                rl, rs, rp = leg(lc, e_row, x_row), leg(scl, e_row, x_row), leg(g.index.map(sid).to_numpy(), e_row, x_row)
                rows.append({'variant': var, 'offset': k, 'month': pd.Timestamp(months[j + 1]), 'long': rl, 'short': rs, 'pool': rp,
                             'LS': rl - rs, 'pool_names': len(g), 'short_under5': float((gs.nominal_close.iloc[:N] < 5).mean())})
        log(f'schedule {k} done')
    mon = pd.DataFrame(rows)
    mon.to_parquet(out / 'monthly.parquet', index=False)

    # 1-3. by schedule and schedule-averaged
    sst = []
    for (var, k), d in mon.groupby(['variant', 'offset']):
        d = d.set_index('month').sort_index()
        for p, m in periods(d.index).items():
            sst.append({'variant': var, 'offset': k, 'period': p, **mstats(d.loc[m, 'LS'])})
    sst = pd.DataFrame(sst)
    sst.to_parquet(out / 'schedule_stats.parquet', index=False)
    avg = mon.groupby(['variant', 'month'])[['long', 'short', 'pool', 'LS', 'short_under5']].mean().reset_index()
    ast = []
    for var, d in avg.groupby('variant'):
        d = d.set_index('month').sort_index()
        for p, m in periods(d.index).items():
            for book in ['LS', 'long', 'short', 'pool']:
                ast.append({'variant': var, 'book': book, 'period': p, **mstats(d.loc[m, book])})
    ast = pd.DataFrame(ast)
    ast.to_parquet(out / 'averaged_stats.parquet', index=False)
    spread = sst[sst.period.isin(['Full', '2012-2026', '5y'])].groupby(['variant', 'period']).sharpe.describe()[['min', '25%', '50%', '75%', 'max']]
    log('Sharpe across the 20 schedules:\n' + spread.round(2).to_string())
    log('schedule-averaged long-short:\n' + ast[(ast.book == 'LS')].pivot_table(index='variant', columns='period', values='sharpe').round(2).to_string())
    log(ast[(ast.book == 'LS')].pivot_table(index='variant', columns='period', values='ann_return').round(3).to_string())

    # 4. factor alpha (schedule-averaged base book)
    base = avg[avg.variant == 'base_1m'].set_index('month').sort_index()
    ff = pd.read_parquet(dataset('factors.french_monthly', must_exist=True))
    ff = ff[ff.source_file.isin(['F-F_Research_Data_5_Factors_2x3_CSV.zip', 'F-F_Momentum_Factor_CSV.zip'])]
    ff = ff.pivot_table(index='date', columns='factor', values='value')
    ff.index = pd.to_datetime(ff.index).to_period('M')
    jk = pd.read_parquet(dataset('jkp.usa_factors_vw_cap', must_exist=True))
    jk = jk[jk.name.isin(['ret_1_0', 'rmax1_21d', 'ivol_capm_21d'])].pivot_table(index='date', columns='name', values='ret')
    jk.index = pd.to_datetime(jk.index).to_period('M')
    y = base.LS.copy()
    y.index = y.index.to_period('M')
    regs = []
    models = {'FF5 + momentum': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom'],
              'FF5 + momentum + reversal, lottery, idio vol': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom', 'ret_1_0', 'rmax1_21d', 'ivol_capm_21d'],
              'market only': ['Mkt-RF']}
    fx = ff.join(jk, how='outer')
    for name, cols in models.items():
        for per, m in {'Full': y.index.year >= 2000, '2012-2026': y.index.year >= 2012}.items():
            d = pd.concat([y[m].rename('LS'), fx[cols]], axis=1, join='inner').dropna()
            b, t, r2 = ols_nw(d.LS.to_numpy(), d[cols].to_numpy())
            regs.append({'model': name, 'period': per, 'months': len(d), 'alpha_ann': b[0] * 12, 'alpha_t': t[0], 'r2': r2,
                         **{f'beta_{c}': v for c, v in zip(cols, b[1:])}, **{f't_{c}': v for c, v in zip(cols, t[1:])}})
    regs = pd.DataFrame(regs)
    regs.to_parquet(out / 'factor_regressions.parquet', index=False)
    log('factor regressions:\n' + regs[['model', 'period', 'months', 'alpha_ann', 'alpha_t', 'r2']].round(3).to_string(index=False))
    log(regs[[c for c in regs.columns if c.startswith('beta_') or c.startswith('t_')]].round(2).to_string())
    jk_last = str(jk.index.max())

    # 5. behaviour
    mk = fx['Mkt-RF']
    d = pd.concat([y.rename('LS'), mk], axis=1, join='inner').dropna()
    updown = {'up_market_months': int((d['Mkt-RF'] > 0).sum()), 'LS_mean_up': float(d.LS[d['Mkt-RF'] > 0].mean()),
              'LS_mean_down': float(d.LS[d['Mkt-RF'] <= 0].mean()),
              'LS_mean_market_worst_decile': float(d.LS[d['Mkt-RF'] <= d['Mkt-RF'].quantile(0.1)].mean()),
              'LS_mean_market_best_decile': float(d.LS[d['Mkt-RF'] >= d['Mkt-RF'].quantile(0.9)].mean())}
    log(f'up/down: {updown}')
    w = (1 + base.LS).cumprod()
    dd = w / w.cummax() - 1
    dds, seen = [], pd.Series(False, index=dd.index)
    for _ in range(4):
        cand = dd[~seen]
        if cand.min() >= -0.05:
            break
        trough = cand.idxmin()
        peak = w[:trough].idxmax()
        rec = w[trough:][w[trough:] >= w[peak]]
        end = rec.index[0] if len(rec) else dd.index[-1]
        dds.append({'peak': peak.date(), 'trough': trough.date(), 'recovered': end.date() if len(rec) else None, 'depth': float(cand.min())})
        seen |= (dd.index >= peak) & (dd.index <= end)
    dds = pd.DataFrame(dds)
    dds.to_parquet(out / 'drawdowns.parquet', index=False)
    yrs = (1 + avg.pivot_table(index='month', columns='variant', values='LS')).groupby(lambda x: x.year).prod() - 1
    yrs.to_parquet(out / 'years.parquet')
    roll = base.LS.rolling(36).apply(lambda x: x.mean() / x.std() * np.sqrt(12))
    log('drawdowns:\n' + dds.to_string(index=False))

    # 6. borrow break-even
    full = ast[(ast.variant == 'base_1m') & (ast.book == 'LS') & (ast.period == 'Full')].iloc[0]
    rec12 = ast[(ast.variant == 'base_1m') & (ast.book == 'LS') & (ast.period == '2012-2026')].iloc[0]
    u5 = float(base.short_under5.mean())
    be = {'all_shorts_full': float(full.ann_return), 'all_shorts_2012_2026': float(rec12.ann_return),
          'sub5_share_of_shorts': u5, 'sub5_only_full': float(full.ann_return / u5), 'sub5_only_2012_2026': float(rec12.ann_return / u5)}
    log(f'borrow break-even: {be}')

    with pd.ExcelWriter(out / 'stress_100.xlsx') as xw:
        pd.DataFrame({'note': ['Full R3000, long top 100 / short bottom 100 CNN scores, equal weight, monthly, next-open entry; $1m-per-side capacity screen (ADV60 >= 20x position) unless stated. Gross; no costs or borrow charged.',
                               'Schedules: formation on the k-th session after each month-end, k = 0..19. Averaged = mean of the 20 schedules per formation month (approximation).',
                               f'Factors: Ken French FF5 + momentum; JKP value-weighted US ret_1_0, rmax1_21d, ivol_capm_21d (JKP data ends {jk_last}). Newey-West 3 lags.',
                               'Borrow break-even: annual fee on short notional that would erase the average long-short return.']}).to_excel(xw, sheet_name='notes', index=False)
        ast.to_excel(xw, sheet_name='averaged_stats', index=False)
        spread.reset_index().to_excel(xw, sheet_name='schedule_spread', index=False)
        sst.to_excel(xw, sheet_name='schedule_stats', index=False)
        regs.to_excel(xw, sheet_name='factor_alpha', index=False)
        dds.to_excel(xw, sheet_name='drawdowns', index=False)
        yrs.reset_index().to_excel(xw, sheet_name='years', index=False)
        pd.DataFrame([updown | be]).T.reset_index().to_excel(xw, sheet_name='behaviour_borrow', index=False)

    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9})
    with PdfPages(out / 'stress_100.pdf') as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        for ax, per in zip(axes, ['Full', '2012-2026', '5y']):
            for i, var in enumerate(VARIANTS):
                v = sst[(sst.variant == var) & (sst.period == per)].sharpe
                ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=12, color='#8a97a8')
                a = ast[(ast.variant == var) & (ast.book == 'LS') & (ast.period == per)].sharpe.iloc[0]
                ax.scatter([i], [a], s=60, marker='D', color='#0f7a58', zorder=3)
            ax.set_xticks(range(len(VARIANTS))); ax.set_xticklabels([LABEL[v] for v in VARIANTS], rotation=30, ha='right', fontsize=8)
            ax.axhline(0, color='black', lw=0.7); ax.grid(alpha=0.25); ax.set_title(f'{per}: Sharpe by rebalance day (grey) and averaged (green)', fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        for var, colr in zip(VARIANTS, ['#15211d', '#b8432f', '#d9a08f', '#0f7a58', '#5fae8d', '#8a97a8']):
            s_ = avg[avg.variant == var].set_index('month').LS.sort_index()
            st_ = ast[(ast.variant == var) & (ast.book == 'LS') & (ast.period == 'Full')].iloc[0]
            axes[0].plot((1 + s_).cumprod(), color=colr, lw=1.3 if var == 'base_1m' else 1, label=f'{LABEL[var]}: {st_.ann_return:.1%}/yr, Sharpe {st_.sharpe:.2f}')
        axes[0].set_yscale('log'); axes[0].grid(alpha=0.25); axes[0].legend(fontsize=7.5, loc='upper left')
        axes[0].set_title('Growth of $1, schedule-averaged long-short, gross', fontsize=9)
        axes[1].plot(roll, color='#0f7a58'); axes[1].axhline(0, color='black', lw=0.7); axes[1].grid(alpha=0.25)
        axes[1].set_title('Base book: rolling 36-month Sharpe', fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4.2))
        yb = yrs['base_1m']
        ax.bar(yb.index.astype(str), yb * 100, color=np.where(yb >= 0, '#0f7a58', '#b8432f'))
        ax.axhline(0, color='black', lw=0.7); ax.grid(alpha=0.25, axis='y'); ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.set_title('Base book: calendar-year return, %, schedule-averaged (2026 to June)', fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    json.dump({'out': str(out), 'schedule_sharpe_spread': spread.round(3).reset_index().to_dict('records'),
               'averaged_LS': ast[ast.book == 'LS'].round(4).to_dict('records'), 'factor_regressions': regs.round(4).to_dict('records'),
               'jkp_last_month': jk_last, 'updown': updown, 'drawdowns': dds.to_dict('records'), 'borrow_break_even': be,
               'elapsed_s': round(time.time() - T0, 1)}, open(out / 'summary.json', 'w'), indent=1, default=str)
    log(f'DONE {out}')


if __name__ == '__main__':
    main()
