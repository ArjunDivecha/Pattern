"""
Script: regime_100.py
Version: 1.1 (2026-11-01)
    v1.1: --stress-dir and --n choose which stress_100.py run to analyse (default: the 100-name run, output
          identical to v1.0); output folder, xlsx and pdf are named regime_<n>.

Description:
    When does the 100-name $1m-per-side CNN long-short book (stress_100.py, base variant) work and when does
    it fail? Splits the book's monthly returns by market conditions, two kinds:
      A. Same-month conditions (explain, not tradable): equal-weight Russell 3000 return, cross-sectional
         return dispersion, a "junk rally" spread (top minus bottom quintile of 60-day realized volatility),
         a low-price spread (bottom minus top price quintile), a size spread (bottom minus top market-cap
         quintile), Fama-French market, momentum and profitability factors, change in VIX.
      B. Conditions known at the formation month-end (could be used to time the book): VIX, corporate credit
         spread, 10y-3m term spread, 12-month change in the 2y yield, equal-weight market return over the
         prior 1 and 12 months, prior-3-month dispersion, prior-3-month junk rally, the book's own prior
         12-month return, ISM PMI (lagged one extra month for publication).
    Each condition is cut into terciles; reports mean monthly return, annualised Sharpe and hit rate per
    tercile, the top-minus-bottom difference with a t-stat, and the same split inside 2000-2011 and
    2012-2026 (breakpoints set within each half) to check stability. Also a multivariate regression of the
    book on the same-month conditions, what those conditions did during each of the four worst drawdowns,
    and where each predictive condition stands at the latest month.

    Book series: schedule offset 0 (formation at the month-end session, entry next open, exit next month-end
    + 1 open), which lines up with calendar months; the 20-schedule average is reported alongside as a check.
    Gross of all costs (house rule).

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z/stress_100_20261101T041600Z/monthly.parquet
    gdm "equities.small_liq_monthly" (r3000_member, mcap_usd, nominal_close, realized_vol60)
    gdm "macro.series_daily" (VIX, BICLB10Y, USGG10YR, USGG3M, USGG2YR, NAPMPMI)
    gdm "factors.french_monthly" (FF5 + momentum)
        (all under /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/...; read via gdm.dataset)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010: prices, sessions)
Output files (in <run dir>/regime_100_<UTC timestamp>/):
    conditions.parquet (month x condition), buckets.parquet, regression.parquet, drawdown_conditions.parquet,
    current.parquet, regime_<n>.xlsx, regime_<n>.pdf (light mode), summary.json;
    log tee'd by the caller to verify_logs/regime_<n>_v1.x.log

Usage:
    .venv/bin/python research/r3000_headline/regime_100.py [--n 40 --stress-dir <run dir>/stress_40_<ts>]

Dependencies: numpy, pandas, matplotlib, openpyxl; gdm; prepare.resolve_dataset, verify_fixed.nw_t.
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

RUN = ROOT / 'research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z'
STRESS = RUN / 'stress_100_20261101T041600Z'
T0 = time.time()

SAME = {  # name: label
    'ew_mkt': 'Equal-weight R3000 return',
    'dispersion': 'Cross-sectional dispersion',
    'junk_vol': 'Junk rally (high-vol minus low-vol)',
    'junk_price': 'Low-price minus high-price',
    'size': 'Small minus big',
    'Mkt-RF': 'FF market',
    'Mom': 'FF momentum',
    'RMW': 'FF profitability',
    'dvix': 'Change in VIX',
}
PRED = {
    'vix': 'VIX at formation',
    'credit': 'Credit spread',
    'term': 'Term spread 10y-3m',
    'd2y_12m': '12m change in 2y yield',
    'mkt_1m': 'Prior-month EW market',
    'mkt_12m': 'Prior-12m EW market',
    'disp_3m': 'Prior-3m dispersion',
    'junk_3m': 'Prior-3m junk rally',
    'book_12m': "Book's own prior-12m return",
    'pmi': 'ISM PMI (lagged)',
}


def log(m):
    print(f'[{time.time() - T0:7.1f}s] {m}', flush=True)


def tstat_diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se


def tercile_table(df, var, y, label, kind):
    rows = []
    for per, m in {'Full': df.index.year >= 2000, '2000-2011': df.index.year <= 2011, '2012-2026': df.index.year >= 2012}.items():
        d = df.loc[m, [var, y]].dropna()
        if len(d) < 30:
            continue
        q = pd.qcut(d[var], 3, labels=['low', 'mid', 'high'])
        r = {'kind': kind, 'condition': var, 'label': label, 'period': per, 'months': len(d)}
        for b in ['low', 'mid', 'high']:
            x = d.loc[q == b, y]
            r[f'{b}_range'] = f'{d.loc[q == b, var].min():.3g} to {d.loc[q == b, var].max():.3g}'
            r[f'{b}_mean_m'] = x.mean()
            r[f'{b}_sharpe'] = x.mean() / x.std() * np.sqrt(12)
            r[f'{b}_hit'] = (x > 0).mean()
        r['high_minus_low_m'] = r['high_mean_m'] - r['low_mean_m']
        r['t_high_minus_low'] = tstat_diff(d.loc[q == 'high', y], d.loc[q == 'low', y])
        r['corr'] = d[var].corr(d[y], method='spearman')
        rows.append(r)
    return rows


def main():
    out = RUN / (f'regime_{N}_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()

    # --- book
    mon = pd.read_parquet(STRESS / 'monthly.parquet')
    b = mon[mon.variant == 'base_1m']
    book0 = b[b.offset == 0].set_index('month').LS.sort_index()
    bookavg = b.groupby('month').LS.mean().sort_index()
    book0.index = book0.index.to_period('M')
    bookavg.index = bookavg.index.to_period('M')
    log(f'book: {len(book0)} months {book0.index.min()}..{book0.index.max()}; offset-0 mean {book0.mean():.4f}, avg mean {bookavg.mean():.4f}')

    # --- R3000 month-end cross-section
    root, _ = resolve_dataset()
    liq = pd.read_parquet(dataset('equities.small_liq_monthly', must_exist=True),
                          columns=['asof_date', 'security', 'r3000_member', 'mcap_usd', 'nominal_close', 'realized_vol60'])
    liq = liq[liq.r3000_member].drop(columns='r3000_member')
    liq['security'] = liq.security.astype(str)
    sess = pd.read_parquet(root / 'sessions.parquet')
    cal = pd.DatetimeIndex(sess.date)
    months = sorted(liq.asof_date.unique())
    me_sess = {pd.Timestamp(m): cal[cal.searchsorted(pd.Timestamp(m), side='right') - 1] for m in months}
    px = pd.read_parquet(root / 'prices.parquet', columns=['security', 'date', 'total_return_index_gross_dividends'],
                         filters=[('date', 'in', list(set(me_sess.values())))])
    px['security'] = px.security.astype(str)
    tri = px.pivot_table(index='date', columns='security', values='total_return_index_gross_dividends')
    log(f'month-end TRI {tri.shape}')
    rows = []
    for j in range(len(months) - 1):
        m0, m1 = pd.Timestamp(months[j]), pd.Timestamp(months[j + 1])
        u = liq[liq.asof_date == months[j]].set_index('security')
        s0, s1 = me_sess[m0], me_sess[m1]
        if s0 not in tri.index or s1 not in tri.index:
            continue
        r = (tri.loc[s1] / tri.loc[s0] - 1).reindex(u.index)
        u = u.assign(r=r).dropna(subset=['r'])
        u['r'] = u.r.clip(*u.r.quantile([0.01, 0.99]))

        def spread(col, top_minus_bottom=True):
            v = u.dropna(subset=[col])
            q = pd.qcut(v[col].rank(method='first'), 5, labels=False)
            hi, lo = v.r[q == 4].mean(), v.r[q == 0].mean()
            return hi - lo if top_minus_bottom else lo - hi
        rows.append({'month': m1.to_period('M'), 'ew_mkt': u.r.mean(), 'dispersion': u.r.std(), 'n_names': len(u),
                     'junk_vol': spread('realized_vol60'), 'junk_price': spread('nominal_close', False),
                     'size': spread('mcap_usd', False)})
    cs = pd.DataFrame(rows).set_index('month')
    log(f'cross-section conditions: {len(cs)} months, median names {cs.n_names.median():.0f}')

    # --- macro (month-end values)
    mac = pd.read_parquet(dataset('macro.series_daily', must_exist=True))
    mac = mac[mac.ticker.isin(['VIX Index', 'BICLB10Y Index', 'USGG10YR Index', 'USGG3M Index', 'USGG2YR Index', 'NAPMPMI Index'])]
    mac = mac.assign(value=pd.to_numeric(mac.value, errors='coerce')).pivot_table(index='date', columns='ticker', values='value')
    mac.index = pd.to_datetime(mac.index)
    mme = mac.ffill().resample('ME').last()
    mme.index = mme.index.to_period('M')
    ff = pd.read_parquet(dataset('factors.french_monthly', must_exist=True))
    ff = ff[ff.source_file.isin(['F-F_Research_Data_5_Factors_2x3_CSV.zip', 'F-F_Momentum_Factor_CSV.zip'])]
    ff = ff.pivot_table(index='date', columns='factor', values='value')
    ff.index = pd.to_datetime(ff.index).to_period('M')
    if ff['Mkt-RF'].abs().max() > 1:
        ff = ff / 100

    C = cs.join(ff[['Mkt-RF', 'Mom', 'RMW', 'SMB']], how='left')
    C['dvix'] = mme['VIX Index'].diff()
    # predictive: values at the end of the month BEFORE the holding month
    lag = lambda s: s.shift(1)  # noqa: E731  index is the holding month
    full_idx = pd.period_range('1999-01', C.index.max(), freq='M')
    mm = mme.reindex(full_idx)
    C['vix'] = lag(mm['VIX Index']).reindex(C.index)
    C['credit'] = lag(mm['BICLB10Y Index']).reindex(C.index)
    C['term'] = lag(mm['USGG10YR Index'] - mm['USGG3M Index']).reindex(C.index)
    C['d2y_12m'] = lag(mm['USGG2YR Index'].diff(12)).reindex(C.index)
    C['pmi'] = mm['NAPMPMI Index'].shift(2).reindex(C.index)
    C['mkt_1m'] = C.ew_mkt.shift(1)
    C['mkt_12m'] = (1 + C.ew_mkt).rolling(12).apply(np.prod, raw=True).shift(1) - 1
    C['disp_3m'] = C.dispersion.rolling(3).mean().shift(1)
    C['junk_3m'] = C.junk_vol.rolling(3).sum().shift(1)
    C['LS'] = book0.reindex(C.index)
    C['LS_avg'] = bookavg.reindex(C.index)
    C['book_12m'] = (1 + C.LS).rolling(12).apply(np.prod, raw=True).shift(1) - 1
    C = C[C.LS.notna()]
    C.to_parquet(out / 'conditions.parquet')
    log(f'conditions: {C.shape}, {C.index.min()}..{C.index.max()}; missing per column:\n{C.isna().sum()[C.isna().sum() > 0].to_string()}')

    # --- tercile buckets
    bk = []
    for v, lab in SAME.items():
        bk += tercile_table(C, v, 'LS', lab, 'same_month')
    for v, lab in PRED.items():
        bk += tercile_table(C, v, 'LS', lab, 'predictive')
        bk += [dict(r, kind='predictive_avgbook') for r in tercile_table(C, v, 'LS_avg', lab, 'predictive')]
    bk = pd.DataFrame(bk)
    bk.to_parquet(out / 'buckets.parquet', index=False)
    show = ['condition', 'period', 'low_mean_m', 'mid_mean_m', 'high_mean_m', 'low_sharpe', 'high_sharpe', 't_high_minus_low', 'corr']
    for k in ['same_month', 'predictive', 'predictive_avgbook']:
        log(f'{k} terciles (monthly mean LS):\n' + bk[bk.kind == k][show].round(3).to_string(index=False))

    # --- multivariate same-month regression
    from verify_fixed import nw_t  # noqa: F401  (kept for parity with stress_100)
    import numpy.linalg as la
    reg = []
    for cols in [['ew_mkt', 'junk_vol', 'dispersion'], ['ew_mkt', 'junk_vol', 'dispersion', 'Mom', 'RMW', 'size']]:
        for per, m in {'Full': C.index.year >= 2000, '2012-2026': C.index.year >= 2012}.items():
            d = C.loc[m, ['LS'] + cols].dropna()
            X = np.column_stack([np.ones(len(d)), d[cols].to_numpy()])
            y = d.LS.to_numpy()
            beta = la.lstsq(X, y, rcond=None)[0]
            e = y - X @ beta
            # Newey-West 3 lags
            Xe = X * e[:, None]
            S = Xe.T @ Xe
            for L in range(1, 4):
                w = 1 - L / 4
                G = Xe[L:].T @ Xe[:-L]
                S += w * (G + G.T)
            XtXi = la.inv(X.T @ X)
            se = np.sqrt(np.diag(XtXi @ S @ XtXi))
            r2 = 1 - e.var() / y.var()
            reg.append({'model': '+'.join(cols), 'period': per, 'months': len(d), 'alpha_ann': beta[0] * 12, 't_alpha': beta[0] / se[0], 'r2': r2,
                        **{f'b_{c}': v for c, v in zip(cols, beta[1:])}, **{f't_{c}': v for c, v in zip(cols, beta[1:] / se[1:])}})
    reg = pd.DataFrame(reg)
    reg.to_parquet(out / 'regression.parquet', index=False)
    log('same-month regression:\n' + reg.round(3).T.to_string())

    # --- drawdowns: what conditions did
    w = (1 + C.LS).cumprod()
    dd = w / w.cummax() - 1
    eps = []
    seen = pd.Series(False, index=dd.index)
    for _ in range(6):
        cand = dd[~seen]
        if cand.empty or cand.min() > -0.08:
            break
        tr = cand.idxmin()
        pk = w[:tr].idxmax()
        after = w[tr:]
        rec = after[after >= w[pk]].index
        end = rec[0] if len(rec) else dd.index[-1]
        seen[pk:end] = True
        span = C.loc[pk:tr].iloc[1:]
        eps.append({'peak': str(pk), 'trough': str(tr), 'recovered': str(rec[0]) if len(rec) else 'not yet', 'depth': float(dd[tr]),
                    'months': len(span),
                    **{f'cum_{v}': float((1 + span[v]).prod() - 1) for v in ['ew_mkt', 'junk_vol', 'junk_price', 'size', 'Mkt-RF', 'Mom', 'RMW']},
                    'avg_dispersion': float(span.dispersion.mean()), 'avg_vix_at_formation': float(span.vix.mean())})
    epi = pd.DataFrame(eps)
    epi.to_parquet(out / 'drawdown_conditions.parquet', index=False)
    log('drawdown conditions:\n' + epi.round(3).T.to_string())
    base_rates = {f'typical_{k}_per_month': float(C[v].mean()) for k, v in [('junk_vol', 'junk_vol'), ('ew_mkt', 'ew_mkt'), ('dispersion', 'dispersion')]}
    log(f'base rates: {base_rates}')

    # --- current readings (latest month-end available for each predictive condition)
    cur = []
    nxt = {}
    last_m = C.index.max()
    # recompute predictive values for the month AFTER the last book month, i.e. as of the last month-end
    nxt['vix'] = mm['VIX Index'].dropna().iloc[-1]
    nxt['credit'] = mm['BICLB10Y Index'].dropna().iloc[-1]
    nxt['term'] = (mm['USGG10YR Index'] - mm['USGG3M Index']).dropna().iloc[-1]
    nxt['d2y_12m'] = mm['USGG2YR Index'].diff(12).dropna().iloc[-1]
    nxt['pmi'] = mm['NAPMPMI Index'].dropna().iloc[-1]
    nxt['mkt_1m'] = C.ew_mkt.iloc[-1]
    nxt['mkt_12m'] = (1 + C.ew_mkt.iloc[-12:]).prod() - 1
    nxt['disp_3m'] = C.dispersion.iloc[-3:].mean()
    nxt['junk_3m'] = C.junk_vol.iloc[-3:].sum()
    nxt['book_12m'] = (1 + C.LS.iloc[-12:]).prod() - 1
    asof = {k: str(last_m) for k in ['mkt_1m', 'mkt_12m', 'disp_3m', 'junk_3m', 'book_12m']}
    for k in ['vix', 'credit', 'term', 'd2y_12m', 'pmi']:
        col = {'vix': 'VIX Index', 'credit': 'BICLB10Y Index', 'term': 'USGG10YR Index', 'd2y_12m': 'USGG2YR Index', 'pmi': 'NAPMPMI Index'}[k]
        asof[k] = str(mm[col].dropna().index[-1])
    for v, lab in PRED.items():
        s = C[v].dropna()
        lo, hi = s.quantile([1 / 3, 2 / 3])
        val = nxt[v]
        terc = 'low' if val <= lo else ('high' if val > hi else 'mid')
        pct = float((s < val).mean())
        cur.append({'condition': v, 'label': lab, 'value': float(val), 'as_of': asof[v], 'tercile': terc, 'percentile': pct})
    cur = pd.DataFrame(cur)
    cur.to_parquet(out / 'current.parquet', index=False)
    log('current readings:\n' + cur.round(3).to_string(index=False))

    # --- xlsx
    with pd.ExcelWriter(out / f'regime_{N}.xlsx') as xw:
        bk.to_excel(xw, sheet_name='terciles', index=False)
        reg.to_excel(xw, sheet_name='regression', index=False)
        epi.to_excel(xw, sheet_name='drawdowns', index=False)
        cur.to_excel(xw, sheet_name='current', index=False)
        C.reset_index().assign(month=lambda d: d.month.astype(str)).to_excel(xw, sheet_name='conditions', index=False)

    # --- pdf
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    with PdfPages(out / f'regime_{N}.pdf') as pdf:
        for kind, title in [('same_month', 'Book return by same-month condition (tercile means, % per month)'),
                            ('predictive', 'Book return by condition known at formation (tercile means, % per month)')]:
            sub = bk[(bk.kind == kind)]
            vars_ = sub.condition.unique()
            fig, axs = plt.subplots(int(np.ceil(len(vars_) / 4)), 4, figsize=(11, 8.5))
            for ax, v in zip(axs.flat, vars_):
                for i, (per, col) in enumerate([('Full', '#1f4e79'), ('2000-2011', '#9dc3e6'), ('2012-2026', '#c55a11')]):
                    r = sub[(sub.condition == v) & (sub.period == per)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    ax.bar(np.arange(3) + (i - 1) * 0.27, [r.low_mean_m * 100, r.mid_mean_m * 100, r.high_mean_m * 100], 0.27, color=col, label=per)
                ax.axhline(0, color='k', lw=0.6)
                ax.set_xticks(range(3), ['low', 'mid', 'high'])
                ax.set_title((SAME | PRED)[v], fontsize=8)
            for ax in list(axs.flat)[len(vars_):]:
                ax.axis('off')
            axs.flat[0].legend(fontsize=7, frameon=False)
            fig.suptitle(title)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        fig, ax = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        t = C.index.to_timestamp()
        ax[0].plot(t, ((1 + C.LS).rolling(12).apply(np.prod, raw=True) - 1) * 100, color='#1f4e79', label='Book, trailing 12m %')
        ax[0].plot(t, ((1 + C.junk_vol).rolling(12).apply(np.prod, raw=True) - 1) * 100, color='#c55a11', label='Junk rally (high-vol minus low-vol), trailing 12m %')
        ax[0].axhline(0, color='k', lw=0.6)
        ax[0].legend(frameon=False)
        ax[0].set_title('The book against the junk rally, trailing 12 months')
        ax[1].fill_between(t, dd * 100, 0, color='#c55a11', alpha=0.5)
        ax[1].set_title('Book drawdown, %')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.scatter(C.junk_vol * 100, C.LS * 100, s=8, color='#1f4e79', alpha=0.6)
        late = C.index >= pd.Period('2025-04', 'M')
        ax.scatter(C.junk_vol[late] * 100, C.LS[late] * 100, s=14, color='#c55a11', label='Apr 2025 onward')
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', lw=0.5)
        ax.set_xlabel('Junk rally that month, % (high-vol quintile minus low-vol quintile)')
        ax.set_ylabel('Book return that month, %')
        ax.legend(frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    summ = {'book_months': int(len(C)), 'first': str(C.index.min()), 'last': str(C.index.max()),
            'book_offset0_ann_mean': float(C.LS.mean() * 12), 'book_avg_ann_mean': float(C.LS_avg.mean() * 12),
            'corr_offset0_vs_avg': float(C.LS.corr(C.LS_avg)), **base_rates,
            'files': {p.name: str(p) for p in sorted(out.iterdir())}}
    (out / 'summary.json').write_text(json.dumps(summ, indent=2))
    log(f'done -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--stress-dir', default=str(STRESS))
    a = ap.parse_args()
    N, STRESS = a.n, Path(a.stress_dir)
    main()
