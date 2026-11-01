"""
Script: holdings_deepdive.py
Version: 1.0 (2026-11-01)

Description:
    Position-level deep dive into a concentrated CNN long-short book produced by stress_100.py v1.1 (--n N),
    base variant ($1m per side capacity screen, full Russell 3000, monthly, next-open entry, gross).
    Answers: what kind of stocks does each side hold; how much does it trade; how concentrated is the P&L
    (do a handful of big winners or short collapses carry it); which side earns the money; which sectors and
    which kinds of positions (cheap vs not, recent losers vs winners, volatile vs calm) make or lose money;
    which names recur; and how the N-name book compares with the 100-name book.

    Characteristics are the month-end snapshot at formation (schedule offset 0 forms on that month-end
    session, so it is exact there; other schedules form up to 19 sessions later). Profiles, turnover and
    bucket splits use schedule 0; concentration and repeat names use all 20 schedules.
    Sectors are today's Bloomberg INDUSTRY_SECTOR applied backwards (not point-in-time; flagged).
    Prior returns come from month-end total-return index levels in the frozen price vintage.

Input files:
    <run dir>/stress_<n>_<ts>/holdings.parquet, monthly.parquet   (stress_100.py v1.1 --n N)
    <run dir>/stress_100_20261101T041600Z/monthly.parquet         (100-name comparison)
      where <run dir> = /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z
    gdm "equities.small_liq_monthly" (r3000_member, nominal_close, mcap_usd, adv60_usd, realized_vol60, median_full_spread_bps60)
    gdm "equities.ref_fields" (INDUSTRY_SECTOR, LONG_COMP_NAME)
        (/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/...; read via gdm.dataset)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010: prices, sessions)
Output files (in <run dir>/holdings_<n>_<UTC timestamp>/):
    positions.parquet (schedule-0 positions with characteristics), profile.parquet, sectors.parquet,
    turnover.parquet, concentration.parquet, attribution.parquet, buckets.parquet, top_names.parquet,
    compare_100.parquet, holdings_<n>.xlsx, holdings_<n>.pdf (light mode), summary.json;
    log tee'd by the caller to verify_logs/holdings_<n>_v1.0.log

Usage:
    .venv/bin/python research/r3000_headline/holdings_deepdive.py --n 40 --stress-dir <run dir>/stress_40_<ts>

Dependencies: numpy, pandas, matplotlib, openpyxl; gdm; prepare.resolve_dataset.
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
S100 = RUN / 'stress_100_20261101T041600Z'
T0 = time.time()


def log(m):
    print(f'[{time.time() - T0:7.1f}s] {m}', flush=True)


def stats(x):
    x = pd.Series(x).dropna()
    return {'ann_return': x.mean() * 12, 'ann_vol': x.std() * np.sqrt(12), 'sharpe': x.mean() / x.std() * np.sqrt(12), 'months': len(x)}


def periods(idx):
    end = idx.max()
    return {'Full': idx >= idx.min(), '2000-2011': idx.year <= 2011, '2012-2026': idx.year >= 2012,
            '5y': idx > end - pd.DateOffset(years=5), '3y': idx > end - pd.DateOffset(years=3), '1y': idx > end - pd.DateOffset(years=1)}


def main(N, sdir):
    out = RUN / (f'holdings_{N}_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    H = pd.read_parquet(sdir / 'holdings.parquet')
    mon = pd.read_parquet(sdir / 'monthly.parquet')
    mon = mon[mon.variant == 'base_1m']
    log(f'{len(H):,} positions, {H.offset.nunique()} schedules, {H.formation_month.nunique()} formation months')
    H['signed'] = np.where(H.side == 'long', H.ret, -H.ret)

    # --- characteristics at formation month-end
    root, _ = resolve_dataset()
    liq = pd.read_parquet(dataset('equities.small_liq_monthly', must_exist=True),
                          columns=['asof_date', 'security', 'r3000_member', 'nominal_close', 'mcap_usd', 'adv60_usd', 'realized_vol60', 'median_full_spread_bps60'])
    liq = liq[liq.r3000_member].drop(columns='r3000_member')
    liq['security'] = liq.security.astype(str)
    liq['asof_date'] = pd.to_datetime(liq.asof_date)
    sess = pd.read_parquet(root / 'sessions.parquet')
    cal = pd.DatetimeIndex(sess.date)
    months = sorted(liq.asof_date.unique())
    me = {pd.Timestamp(m): cal[cal.searchsorted(pd.Timestamp(m), side='right') - 1] for m in months}
    px = pd.read_parquet(root / 'prices.parquet', columns=['security', 'date', 'total_return_index_gross_dividends'],
                         filters=[('date', 'in', list(set(me.values())))])
    px['security'] = px.security.astype(str)
    tri = px.pivot_table(index='date', columns='security', values='total_return_index_gross_dividends')
    sess2m = {v: k for k, v in me.items()}
    tri.index = pd.DatetimeIndex([sess2m[d] for d in tri.index])
    tri = tri.sort_index()
    r1 = (tri / tri.shift(1) - 1).stack().rename('prior_1m')
    r12 = (tri.shift(1) / tri.shift(12) - 1).stack().rename('prior_12_1m')
    pr = pd.concat([r1, r12], axis=1)
    pr.index.names = ['asof_date', 'security']
    liq = liq.merge(pr.reset_index(), on=['asof_date', 'security'], how='left')
    ref = pd.read_parquet(dataset('equities.ref_fields', must_exist=True), columns=['security', 'INDUSTRY_SECTOR', 'LONG_COMP_NAME'])
    ref['security'] = ref.security.astype(str)
    ref['sector'] = ref.INDUSTRY_SECTOR.where(~ref.INDUSTRY_SECTOR.isin(['Government', 'Diversified']), 'Other').fillna('Other')
    liq = liq.merge(ref[['security', 'sector', 'LONG_COMP_NAME']], on='security', how='left')
    liq['sector'] = liq.sector.fillna('Other')
    elig = liq[liq.adv60_usd >= 20 * 1e6 / N].copy()   # the capacity-screened pool the book picks from
    log(f'eligible pool: median {elig.groupby("asof_date").size().median():.0f} names per month')

    P = H[H.offset == 0].merge(liq.rename(columns={'asof_date': 'formation_month'}), on=['formation_month', 'security'], how='left')
    miss = P.nominal_close.isna().mean()
    log(f'schedule-0 positions {len(P):,}; characteristics missing for {miss:.2%}')
    P.to_parquet(out / 'positions.parquet', index=False)

    # 1. profile
    cols = {'nominal_close': 'price $', 'mcap_usd': 'market cap $', 'adv60_usd': '60d avg $ volume', 'realized_vol60': '60d realized vol',
            'median_full_spread_bps60': 'bid-ask spread bps', 'prior_1m': 'prior 1m return', 'prior_12_1m': 'prior 12-1m return'}
    prof = []
    for per, lo_, hi_ in [('Full', 2000, 2026), ('2000-2011', 2000, 2011), ('2012-2026', 2012, 2026)]:
        for side, d in [('long', P[P.side == 'long']), ('short', P[P.side == 'short']), ('pool', elig)]:
            dm = d.formation_month if side != 'pool' else d.asof_date
            d = d[(dm.dt.year >= lo_) & (dm.dt.year <= hi_)]
            r = {'period': per, 'side': side}
            for c, lab in cols.items():
                r[lab] = d[c].median()
            r['share price < $5'] = (d.nominal_close < 5).mean()
            prof.append(r)
    prof = pd.DataFrame(prof)
    prof.to_parquet(out / 'profile.parquet', index=False)
    log('profile (medians):\n' + prof.round(3).to_string(index=False))

    # 2. sectors: share of positions vs pool, and contribution (signed return / N, averaged per month, annualised)
    nm = P.formation_month.nunique()
    sec = P.groupby(['sector', 'side']).agg(positions=('ret', 'size'), mean_signed=('signed', 'mean'), total_signed=('signed', 'sum')).reset_index()
    sec['share'] = sec.positions / sec.groupby('side').positions.transform('sum')
    sec['contrib_ann'] = sec.total_signed / N / nm * 12
    pool_share = elig.groupby('sector').size() / len(elig)
    sec['pool_share'] = sec.sector.map(pool_share)
    sec.to_parquet(out / 'sectors.parquet', index=False)
    log('sectors:\n' + sec.round(3).sort_values(['side', 'contrib_ann']).to_string(index=False))

    # 3. turnover (schedule 0)
    tov = []
    for side, d in P.groupby('side'):
        g = d.groupby('formation_month').security.apply(set).sort_index()
        prev = None
        for m, s in g.items():
            if prev is not None:
                tov.append({'side': side, 'month': m, 'new_share': len(s - prev) / max(len(s), 1)})
            prev = s
    tov = pd.DataFrame(tov)
    # spell lengths
    spells = []
    for side, d in P.groupby('side'):
        mi = {m: i for i, m in enumerate(sorted(P.formation_month.unique()))}
        for sec_, g in d.groupby('security'):
            idx = sorted(mi[m] for m in g.formation_month)
            run = 1
            for a, b in zip(idx, idx[1:]):
                if b == a + 1:
                    run += 1
                else:
                    spells.append({'side': side, 'months': run})
                    run = 1
            spells.append({'side': side, 'months': run})
    spells = pd.DataFrame(spells)
    tsum = tov.groupby('side').new_share.mean().rename('avg_new_share').to_frame().join(spells.groupby('side').months.mean().rename('avg_spell_months'))
    tsum = tsum.join(spells.assign(one=spells.months == 1).groupby('side').one.mean().rename('share_spells_one_month'))
    tov.to_parquet(out / 'turnover.parquet', index=False)
    log('turnover:\n' + tsum.round(3).to_string())

    # 4. concentration of P&L (all schedules)
    conc = []
    for side in ['long', 'short', 'both']:
        d = H if side == 'both' else H[H.side == side]
        s = d.signed.sort_values(ascending=False).to_numpy()
        tot = s.sum()
        n = len(s)
        r = {'side': side, 'positions': n, 'mean_signed': s.mean(), 'median_signed': np.median(s), 'hit_rate': (s > 0).mean(),
             'p1': np.percentile(s, 1), 'p99': np.percentile(s, 99), 'skew': pd.Series(s).skew(),
             'share_pnl_top1pct': s[:max(1, n // 100)].sum() / tot, 'share_pnl_top5pct': s[:max(1, n // 20)].sum() / tot,
             'share_pnl_bottom1pct': s[-max(1, n // 100):].sum() / tot,
             'mean_signed_ex_top1pct': s[max(1, n // 100):].mean(), 'mean_signed_ex_top_and_bottom1pct': s[max(1, n // 100):-max(1, n // 100)].mean()}
        conc.append(r)
    conc = pd.DataFrame(conc)
    conc.to_parquet(out / 'concentration.parquet', index=False)
    log('concentration:\n' + conc.round(4).T.to_string())
    # book return without the best 1% of positions each month (trim per month, all schedules) -> how fragile
    def trimmed_book(q):
        res = []
        for (k, m), d in H.groupby(['offset', 'formation_month']):
            x = d.signed
            if q > 0:
                x = x[x <= x.quantile(1 - q)]
            res.append({'offset': k, 'formation_month': m, 'r': x.mean()})
        r = pd.DataFrame(res).groupby('formation_month').r.mean()
        return r
    trims = {f'drop best {int(q * 100)}% of positions each month': trimmed_book(q) for q in [0, 0.02, 0.05]}
    trim_tab = pd.DataFrame({k: stats(v * 2) for k, v in trims.items()}).T  # x2: mean signed position * 2 sides = LS approx
    log('trimmed books (mean signed position x 2 = long-short):\n' + trim_tab.round(3).to_string())

    # 5. leg attribution (schedule-averaged)
    avg = mon.groupby('month')[['long', 'short', 'pool', 'LS']].mean()
    avg['long_vs_pool'] = avg.long - avg.pool
    avg['pool_vs_short'] = avg.pool - avg.short
    att = []
    for per, m in periods(avg.index).items():
        for c in ['LS', 'long_vs_pool', 'pool_vs_short', 'long', 'short', 'pool']:
            att.append({'period': per, 'series': c, **stats(avg.loc[m, c])})
    att = pd.DataFrame(att)
    att.to_parquet(out / 'attribution.parquet', index=False)
    log('attribution:\n' + att.pivot_table(index='series', columns='period', values='ann_return').round(3).to_string())
    log(att.pivot_table(index='series', columns='period', values='sharpe').round(2).to_string())

    # 6. buckets: what kind of position makes money (schedule 0)
    bk = []
    P['price_band'] = pd.cut(P.nominal_close, [0, 5, 10, 20, 1e9], labels=['<$5', '$5-10', '$10-20', '>$20'])
    for side in ['long', 'short']:
        d = P[P.side == side].copy()
        d['prior_12_1m_tercile'] = d.groupby('formation_month').prior_12_1m.transform(lambda x: pd.qcut(x.rank(method='first'), 3, labels=['loser', 'mid', 'winner']) if x.notna().sum() >= 3 else np.nan)
        d['prior_1m_tercile'] = d.groupby('formation_month').prior_1m.transform(lambda x: pd.qcut(x.rank(method='first'), 3, labels=['down', 'mid', 'up']) if x.notna().sum() >= 3 else np.nan)
        d['vol_tercile'] = d.groupby('formation_month').realized_vol60.transform(lambda x: pd.qcut(x.rank(method='first'), 3, labels=['calm', 'mid', 'volatile']) if x.notna().sum() >= 3 else np.nan)
        d['size_tercile'] = d.groupby('formation_month').mcap_usd.transform(lambda x: pd.qcut(x.rank(method='first'), 3, labels=['small', 'mid', 'large']) if x.notna().sum() >= 3 else np.nan)
        for by in ['price_band', 'prior_12_1m_tercile', 'prior_1m_tercile', 'vol_tercile', 'size_tercile']:
            for per, m in {'Full': d.formation_month.dt.year >= 2000, '2012-2026': d.formation_month.dt.year >= 2012}.items():
                g = d[m].groupby(by, observed=True).agg(n=('signed', 'size'), mean_signed_m=('signed', 'mean'), hit=('signed', lambda x: (x > 0).mean()))
                g['share_positions'] = g.n / g.n.sum()
                for b_, r in g.iterrows():
                    bk.append({'side': side, 'split': by, 'bucket': str(b_), 'period': per, **r.to_dict()})
    bk = pd.DataFrame(bk)
    bk.to_parquet(out / 'buckets.parquet', index=False)
    log('buckets (mean signed position return per month):\n' + bk.round(4).to_string(index=False))

    # 7. top names (all schedules; contribution in book return units, averaged over schedules)
    H2 = H.merge(ref[['security', 'LONG_COMP_NAME', 'sector']], on='security', how='left')
    tn = H2.groupby(['side', 'security', 'LONG_COMP_NAME', 'sector'], dropna=False).agg(
        months_held=('formation_month', 'nunique'), positions=('signed', 'size'), mean_signed=('signed', 'mean'),
        contrib=('signed', lambda x: x.sum() / N / 20)).reset_index()
    tn['first'] = H2.groupby(['side', 'security']).formation_month.min().reindex(pd.MultiIndex.from_frame(tn[['side', 'security']])).to_numpy()
    tn = tn.sort_values('contrib', ascending=False)
    tn.to_parquet(out / 'top_names.parquet', index=False)
    for side in ['long', 'short']:
        d = tn[tn.side == side]
        log(f'{side}: {len(d)} distinct names; best 10 contributors:\n' + d.head(10)[['LONG_COMP_NAME', 'sector', 'months_held', 'mean_signed', 'contrib']].round(3).to_string(index=False))
        log(f'{side}: worst 10:\n' + d.tail(10)[['LONG_COMP_NAME', 'sector', 'months_held', 'mean_signed', 'contrib']].round(3).to_string(index=False))
        log(f'{side}: most held:\n' + d.sort_values('months_held', ascending=False).head(10)[['LONG_COMP_NAME', 'sector', 'months_held', 'mean_signed', 'contrib']].round(3).to_string(index=False))
    totc = tn.contrib.sum()
    share_top20 = tn.head(20).contrib.sum() / totc
    log(f'total contribution {totc:.3f} (book units, summed over all months); top 20 names = {share_top20:.1%}')

    # 8. compare with the 100-name book
    m100 = pd.read_parquet(S100 / 'monthly.parquet')
    a100 = m100[m100.variant == 'base_1m'].groupby('month').LS.mean()
    cmp_ = []
    for per, m in periods(avg.index).items():
        cmp_.append({'period': per, 'book': f'{N} per side', **stats(avg.loc[m, 'LS'])})
        mm = periods(a100.index)[per]
        cmp_.append({'period': per, 'book': '100 per side', **stats(a100[mm])})
    cmp_ = pd.DataFrame(cmp_)
    corr = float(avg.LS.corr(a100))
    cmp_.to_parquet(out / 'compare_100.parquet', index=False)
    log(f'compare (corr {corr:.2f}):\n' + cmp_.pivot_table(index='book', columns='period', values=['ann_return', 'sharpe']).round(3).to_string())

    with pd.ExcelWriter(out / f'holdings_{N}.xlsx') as xw:
        pd.DataFrame({'note': [f'Full R3000, long top {N} / short bottom {N} CNN scores, $1m per side capacity screen (ADV60 >= ${20e6 / N:,.0f}), monthly, next-open entry, gross.',
                               'Signed return = position return for longs, minus it for shorts. Profiles/buckets use schedule 0; concentration and names use all 20 schedules.',
                               "Sectors are today's Bloomberg INDUSTRY_SECTOR applied backwards (not point-in-time)."]}).to_excel(xw, sheet_name='notes', index=False)
        prof.to_excel(xw, sheet_name='profile', index=False)
        sec.to_excel(xw, sheet_name='sectors', index=False)
        tsum.reset_index().to_excel(xw, sheet_name='turnover', index=False)
        conc.to_excel(xw, sheet_name='concentration', index=False)
        trim_tab.reset_index().to_excel(xw, sheet_name='trimmed_books', index=False)
        att.to_excel(xw, sheet_name='attribution', index=False)
        bk.to_excel(xw, sheet_name='buckets', index=False)
        tn.head(300).to_excel(xw, sheet_name='top_names', index=False)
        tn.tail(300).to_excel(xw, sheet_name='worst_names', index=False)
        cmp_.to_excel(xw, sheet_name='vs_100', index=False)

    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    with PdfPages(out / f'holdings_{N}.pdf') as pdf:
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
        for c, colr in [('LS', '#15211d'), ('long_vs_pool', '#0f7a58'), ('pool_vs_short', '#b8432f')]:
            axs[0].plot((1 + avg[c]).cumprod(), color=colr, label={'LS': 'long-short', 'long_vs_pool': 'long side vs pool', 'pool_vs_short': 'short side vs pool'}[c])
        axs[0].set_yscale('log'); axs[0].legend(frameon=False); axs[0].grid(alpha=0.25)
        axs[0].set_title(f'{N}-name book: where the money comes from (schedule-averaged, gross)')
        axs[1].plot((1 + avg.LS).cumprod(), color='#15211d', label=f'{N} per side')
        axs[1].plot((1 + a100).cumprod(), color='#8a97a8', label='100 per side')
        axs[1].set_yscale('log'); axs[1].legend(frameon=False); axs[1].grid(alpha=0.25); axs[1].set_title(f'{N} vs 100 names per side (corr {corr:.2f})')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axs = plt.subplots(2, 4, figsize=(14, 7))
        for ax, (c, lab) in zip(axs.flat, list(cols.items()) + [('share_lt5', 'share price < $5')]):
            for side, colr in [('long', '#0f7a58'), ('short', '#b8432f')]:
                d = P[P.side == side]
                s_ = (d.nominal_close < 5).groupby(d.formation_month).mean() if c == 'share_lt5' else d.groupby('formation_month')[c].median()
                ax.plot(s_.rolling(12).mean(), color=colr, label=side)
            s_ = (elig.nominal_close < 5).groupby(elig.asof_date).mean() if c == 'share_lt5' else elig.groupby('asof_date')[c].median()
            ax.plot(s_.rolling(12).mean(), color='#8a97a8', label='pool')
            if c in ('mcap_usd', 'adv60_usd'):
                ax.set_yscale('log')
            ax.set_title(f'{lab} (median, 12m smoothed)'); ax.grid(alpha=0.25)
        axs.flat[0].legend(frameon=False)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        for ax, side in zip(axs, ['long', 'short']):
            d = sec[sec.side == side].sort_values('contrib_ann')
            ax.barh(d.sector, d.contrib_ann * 100, color=np.where(d.contrib_ann >= 0, '#0f7a58', '#b8432f'))
            ax.set_title(f'{side} side: contribution by sector, % per year (signed)'); ax.grid(alpha=0.25, axis='x')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axs = plt.subplots(2, 5, figsize=(15, 6.5))
        for i, side in enumerate(['long', 'short']):
            for j, by in enumerate(['price_band', 'prior_12_1m_tercile', 'prior_1m_tercile', 'vol_tercile', 'size_tercile']):
                ax = axs[i, j]
                for k_, (per, colr) in enumerate([('Full', '#1f4e79'), ('2012-2026', '#c55a11')]):
                    d = bk[(bk.side == side) & (bk.split == by) & (bk.period == per)]
                    ax.bar(np.arange(len(d)) + (k_ - 0.5) * 0.4, d.mean_signed_m * 100, 0.4, color=colr, label=per)
                    ax.set_xticks(range(len(d)), d.bucket, fontsize=7)
                ax.axhline(0, color='k', lw=0.6)
                ax.set_title(f'{side}: by {by.replace("_", " ")}', fontsize=8)
        axs[0, 0].legend(frameon=False, fontsize=7)
        fig.suptitle('Mean signed position return, % per month (schedule 0)')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
        for side, colr in [('long', '#0f7a58'), ('short', '#b8432f')]:
            s = H[H.side == side].signed.clip(-1, 2)
            axs[0].hist(s * 100, bins=150, color=colr, alpha=0.5, label=side, density=True)
        axs[0].set_xlim(-80, 120); axs[0].legend(frameon=False); axs[0].set_title('Signed position returns, % (clipped at -100/+200)')
        for side, colr in [('long', '#0f7a58'), ('short', '#b8432f')]:
            t = tov[tov.side == side].set_index('month').new_share
            axs[1].plot(t.rolling(12).mean() * 100, color=colr, label=side)
        axs[1].set_title('Share of names that are new each month, %, 12m smoothed (schedule 0)'); axs[1].legend(frameon=False); axs[1].grid(alpha=0.25)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    summ = {'n': N, 'stress_dir': str(sdir), 'positions': int(len(H)), 'characteristics_missing_share': float(miss),
            'turnover': tsum.round(4).reset_index().to_dict('records'), 'concentration': conc.round(4).to_dict('records'),
            'trimmed': trim_tab.round(4).reset_index().to_dict('records'), 'corr_vs_100': corr, 'top20_names_share': float(share_top20),
            'elapsed_s': round(time.time() - T0, 1), 'files': sorted(p.name for p in out.iterdir())}
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, default=str))
    log(f'DONE {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--stress-dir', required=True)
    a = ap.parse_args()
    main(a.n, Path(a.stress_dir))
