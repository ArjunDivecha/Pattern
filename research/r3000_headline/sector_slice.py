"""
Script: sector_slice.py
Version: 1.1 (2026-10-14)

Description:
    Sector slicing of the R3000 headline run (expanding retraining, 2000-2026). Asks three things:
      1. Is the CNN a sector bet? Rebuild the D10-D1 book after removing sector (and finer
         industry) from the score: (a) score minus its sector mean, deciles over all names;
         also repeated inside large caps (top 1000) and small caps (rest), ranks re-taken there;
         (b) deciles formed inside each sector and pooled, so every decile holds the same sector
         mix; (c) score minus its industry mean (SIC 2-digit major group); (d) all price
         characteristics from neutralize.py plus sector and industry dummies.
      2. How much comes from picking sectors versus picking stocks inside them? Split the
         rank-normalised score into a between-sector part (the sector's mean score) and a
         within-sector part, Fama-MacBeth both against the realised return, and run a
         sector-timing book (long the three sectors with the highest mean score, short the
         three lowest, equal weight within).
      3. Where does it work? The within-sector D10-D1 book for each sector, by period and by
         size (large = top 1000 by point-in-time market cap), plus each sector's share of the
         top and bottom deciles against its share of the universe.
    Same conventions as report.py / neutralize.py: cohorts every 20th session from the first
    test session, equal weight, deciles over all names, returns over known outcomes, delisted
    names earn the return to their last price, gross of all costs. Periods: Full, 5y, 3y, 1y
    (counted back from the last cohort) and the two halves 2000-2011 and 2012-2026.

    Sector sources (Global Data Mart; both are TODAY's classification applied backwards, not
    point-in-time; a company's sector rarely changes, but this is a known look-ahead and is
    flagged in the outputs):
      sector    Bloomberg INDUSTRY_SECTOR (100% of rows; 9 sectors used, the tiny
                'Government'/'Diversified' groups folded into 'Other')
      industry  SEC SIC code via the verified EDGAR CIK map, 2-digit major group (93% of rows);
                a name without a SIC code takes its Bloomberg sector as its industry
    The previous neutralize.py sector control used GICS, which covers only 57% of rows (30% in
    2000) because dead companies are missing from it.

Input files:
    <run dir>/neutralize_<ts>/characteristics.parquet (from research/r3000_headline/neutralize.py v1.1)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/ref_fields/security_reference.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/edgar/data/security_cik_map.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/edgar/data/entities.parquet
Output files (in <run dir>/sector_slice_<UTC timestamp>/):
    books.parquet             book x period: D10-D1 stats, monotonicity, decile excess profile
    fama_macbeth.parquet      between- vs within-sector coefficients (% per 20 sessions), NW t
    size_books.parquet        size bucket x book x period: D10-D1 stats (raw / sector- / industry-demeaned)
    by_sector.parquet         sector x universe x period: within-sector D10-D1 stats
    tilts.parquet             sector x year: share of D1, D10 and the universe
    sector_map.parquet        security -> sector, industry (as used)
    cohort_series.parquet     per cohort date: D10-D1 return of every book
    sector_slice.xlsx         all tables above
    sector_slice.pdf          charts (light mode)
    summary.json

Usage:
    .venv/bin/python research/r3000_headline/sector_slice.py --char FILE

Dependencies: numpy, pandas, matplotlib, openpyxl.
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neutralize import rank_norm, deciles_by_date, book_stats, ALL, DEC  # noqa: E402
from verify_fixed import nw_t, PPY  # noqa: E402

GDM = Path('/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources')
REF = GDM / 'equities/data/ref_fields/security_reference.parquet'
CIK = GDM / 'edgar/data/security_cik_map.parquet'
ENT = GDM / 'edgar/data/entities.parquet'
T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)


def sector_map(secs):
    r = pd.read_parquet(REF, columns=['security', 'INDUSTRY_SECTOR'])
    m = pd.read_parquet(CIK, columns=['security', 'cik', 'verified'])
    e = pd.read_parquet(ENT, columns=['cik', 'sic'])
    m = m[m.verified.fillna(False).astype(bool)].merge(e, on='cik', how='left')
    out = pd.DataFrame({'security': secs}).merge(r, on='security', how='left').merge(m[['security', 'sic']], on='security', how='left')
    out = out.drop_duplicates('security')
    s = out.INDUSTRY_SECTOR.fillna('Other')
    out['sector'] = s.where(~s.isin(['Government', 'Diversified']), 'Other')
    sic2 = out.sic.astype('string').str.strip().str[:2]
    out['has_sic'] = sic2.notna() & (sic2.str.len() == 2)
    out['industry'] = np.where(out.has_sic, 'SIC' + sic2.fillna(''), 'BBG:' + out.sector)
    return out[['security', 'sector', 'industry', 'has_sic', 'sic']]


def demean(x, d, by):
    return x - pd.Series(x, index=d.index).groupby([d.date, d[by]]).transform('mean').to_numpy()


def within_group_deciles(d, col, by):
    """Deciles formed inside each (date, group), pooled: D_k = equal-weight mean of every group's D_k names."""
    g = d.sort_values(['date', by, col, 'security'])
    pos = g.groupby(['date', by]).cumcount().to_numpy()
    n = g.groupby(['date', by])[col].transform('size').to_numpy()
    g = g.assign(dec=pos * 10 // n + 1)
    k = g.dropna(subset=['ret'])
    dd = k.pivot_table(index='date', columns='dec', values='ret', aggfunc='mean')
    dd.columns = [f'D{c}' for c in dd.columns]
    dd['EW'] = k.groupby('date').ret.mean()
    return dd


def residual(d, X, y):
    dates = d.date.to_numpy()
    b = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    res = np.empty(len(d))
    for i in range(len(b) - 1):
        s = slice(b[i], b[i + 1])
        res[s] = y[s] - X[s] @ np.linalg.lstsq(X[s], y[s], rcond=None)[0]
    return res


def period_masks(idx):
    last = idx.max()
    return {'Full': idx == idx, '5y': idx > last - pd.DateOffset(years=5), '3y': idx > last - pd.DateOffset(years=3),
            '1y': idx > last - pd.DateOffset(years=1), '2000-2011': idx.year <= 2011, '2012-2026': idx.year >= 2012}


def ls_stats(x):
    x = x.dropna()
    if len(x) < 3 or x.std() == 0:
        return {'cohorts': len(x), 'ls_ann': np.nan, 'ls_sharpe': np.nan, 'ls_nw_t': np.nan, 'hit': np.nan}
    return {'cohorts': len(x), 'ls_ann': x.mean() * PPY, 'ls_sharpe': x.mean() / x.std() * np.sqrt(PPY),
            'ls_nw_t': nw_t(x), 'hit': float((x > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--char', required=True)
    a = ap.parse_args()
    char = Path(a.char).resolve()
    run = char.parent.parent
    out = run / ('sector_slice_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()

    d = pd.read_parquet(char)
    d['security'] = d.security.astype(str)
    d = d[d.date.isin(d.dropna(subset=['ret']).date.unique())].sort_values(['date', 'security']).reset_index(drop=True)
    sm = sector_map(sorted(d.security.unique()))
    sm.to_parquet(out / 'sector_map.parquet', index=False)
    d = d.drop(columns=['sector']).merge(sm[['security', 'sector', 'industry', 'has_sic']], on='security', how='left')
    d = d.sort_values(['date', 'security']).reset_index(drop=True)
    cov = {'rows': len(d), 'sector_known': float((d.sector != 'Other').mean()), 'sic_known': float(d.has_sic.mean()),
           'n_sectors': int(d.sector.nunique()), 'n_industries': int(d.industry.nunique()),
           'median_names_per_industry_date': float(d.groupby(['date', 'industry']).size().median())}
    log(f'coverage {json.dumps(cov)}')

    rk = rank_norm(d, ['score'] + ALL)
    rk.columns = ['cnn'] + ALL
    cnn = rk.cnn.to_numpy()
    sec_d = pd.get_dummies(d.sector, drop_first=True, dtype=float).to_numpy()
    ind_d = pd.get_dummies(d.industry, drop_first=True, dtype=float).to_numpy()
    one = np.ones((len(d), 1))
    ctrl = rk[ALL].to_numpy()

    # 1. Books
    log('books')
    sigs = {'raw CNN': cnn,
            'sector-demeaned (minus sector mean score)': demean(cnn, d, 'sector'),
            'industry-demeaned (minus SIC 2-digit mean score)': demean(cnn, d, 'industry'),
            'all price controls (as neutralize.py, no sector)': residual(d, np.hstack([one, ctrl]), cnn),
            'all price controls + sector': residual(d, np.hstack([one, ctrl, sec_d]), cnn),
            'all price controls + industry': residual(d, np.hstack([one, ctrl, ind_d]), cnn)}
    decs = {}
    for name, s in sigs.items():
        d['sig'] = s
        decs[name] = deciles_by_date(d, 'sig')
        log(f'{name}: Sharpe {book_stats(decs[name])["ls_sharpe"]:.3f}')
    d['sig'] = cnn
    decs['within-sector deciles (equal sector mix in every decile)'] = within_group_deciles(d, 'sig', 'sector')
    d['sig'] = sigs['all price controls + sector']
    decs['within-sector deciles of the all-controls residual'] = within_group_deciles(d, 'sig', 'sector')
    for name in list(decs)[-2:]:
        log(f'{name}: Sharpe {book_stats(decs[name])["ls_sharpe"]:.3f}')
    raw_sh = book_stats(decs['raw CNN'])['ls_sharpe']
    assert abs(raw_sh - 0.7161) < 5e-4, raw_sh

    # sector-timing book: long the 3 sectors with the highest mean score, short the 3 lowest
    d['sig'] = cnn
    named = d[d.sector != 'Other']
    sm_score = named.groupby(['date', 'sector']).sig.mean().rename('sec_score').reset_index()
    sm_ret = named.dropna(subset=['ret']).groupby(['date', 'sector']).ret.mean().rename('sec_ret').reset_index()
    st = sm_score.merge(sm_ret, on=['date', 'sector'])
    st['r'] = st.groupby('date').sec_score.rank(ascending=False, method='first')
    nsec = st.groupby('date').sector.transform('size')
    timing = st[st.r <= 3].groupby('date').sec_ret.mean() - st[st.r > nsec - 3].groupby('date').sec_ret.mean()

    series = pd.DataFrame({k: v.D10 - v.D1 for k, v in decs.items()})
    series['sector timing (top 3 minus bottom 3 sectors by mean score)'] = timing
    series.to_parquet(out / 'cohort_series.parquet')
    books = []
    for name in series.columns:
        dd = decs.get(name)
        for p, m in period_masks(series.index).items():
            row = {'book': name, 'period': p, **ls_stats(series.loc[m, name])}
            if dd is not None:
                bs = book_stats(dd.loc[dd.index.isin(series.index[m])])
                row.update({'monotonicity': bs['monotonicity'], 'top3_bottom3_sharpe': bs['top3_bottom3_sharpe'],
                            **{k: v for k, v in bs.items() if k.endswith('_excess_ann')}})
            books.append(row)
    books = pd.DataFrame(books)
    corr = {k: float(pd.Series(v).groupby(d.date).apply(lambda s: s.rank().corr(pd.Series(cnn[s.index], index=s.index).rank())).mean())
            for k, v in sigs.items()}
    books['rank_corr_with_raw'] = books.book.map(corr)

    # 2. Between vs within Fama-MacBeth
    log('Fama-MacBeth between vs within')
    between = cnn - demean(cnn, d, 'sector')
    within = cnn - between
    k = d.assign(between=between, within=within, cnn=cnn).dropna(subset=['ret'])
    fm_rows = []
    for spec, cols in {'CNN alone': ['cnn'], 'between + within': ['between', 'within'],
                       'between alone': ['between'], 'within alone': ['within']}.items():
        coefs = []
        for date, g in k.groupby('date', sort=True):
            X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in cols])
            coefs.append(np.linalg.lstsq(X, g.ret.to_numpy(), rcond=None)[0][1:])
        coefs = np.array(coefs)
        for j, c in enumerate(cols):
            fm_rows.append({'spec': spec, 'regressor': c, 'coef_pct_per_20d': coefs[:, j].mean() * 100,
                            'nw_t': nw_t(coefs[:, j]), 'cohorts': len(coefs)})
    fm = pd.DataFrame(fm_rows)
    var_share = float((pd.Series(between).groupby(d.date).var() / pd.Series(cnn).groupby(d.date).var()).mean())
    log(f'between-sector share of score variance {var_share:.3f}')

    # 3. Per sector, within-sector deciles; by size
    log('by sector')
    d['sig'] = cnn
    d['resid'] = sigs['all price controls + sector']
    sec_rows = []
    for sec, g in d[d.sector != 'Other'].groupby('sector'):
        for uni, gg in [('all', g), ('large caps (top 1000)', g[g.cap_rank <= 1000]), ('small caps (rest)', g[g.cap_rank > 1000])]:
            for col, lab in [('sig', 'raw CNN'), ('resid', 'all-controls residual')]:
                if uni != 'all' and col == 'resid':
                    continue
                dd = deciles_by_date(gg, col)
                ls = dd.D10 - dd.D1
                names = gg.groupby('date').size()
                for p, m in period_masks(ls.index).items():
                    sec_rows.append({'sector': sec, 'universe': uni, 'signal': lab, 'period': p,
                                     'median_names': float(names.median()), 'share_of_universe': float(len(g) / len(d)),
                                     **ls_stats(ls[m])})
    by_sector = pd.DataFrame(sec_rows)

    # tilts
    d['raw_decile'] = d.groupby('date').score.transform(lambda s: (s.rank(method='first') - 1) * 10 // len(s) + 1).astype(int)
    d['year'] = d.date.dt.year
    uni_sh = d.groupby(['year', 'sector']).size() / d.groupby('year').size()
    d1_sh = d[d.raw_decile == 1].groupby(['year', 'sector']).size() / d[d.raw_decile == 1].groupby('year').size()
    d10_sh = d[d.raw_decile == 10].groupby(['year', 'sector']).size() / d[d.raw_decile == 10].groupby('year').size()
    tilts = pd.DataFrame({'universe_share': uni_sh, 'D1_share': d1_sh, 'D10_share': d10_sh}).fillna(0.0).reset_index()
    tilts['D10_minus_D1'] = tilts.D10_share - tilts.D1_share

    # sector / industry neutral books inside each size bucket (ranks and means re-taken in the bucket)
    log('size buckets')
    size_rows = []
    for uni, mask in [('large caps (top 1000)', d.cap_rank <= 1000), ('small caps (rest)', d.cap_rank > 1000)]:
        g = d[np.asarray(mask)].reset_index(drop=True)
        c = rank_norm(g, ['score']).score.to_numpy()
        for name, sg in [('raw CNN', c), ('sector-demeaned', demean(c, g, 'sector')), ('industry-demeaned', demean(c, g, 'industry'))]:
            g['sig'] = sg
            ls = (lambda dd: dd.D10 - dd.D1)(deciles_by_date(g, 'sig'))
            for p, m in period_masks(ls.index).items():
                size_rows.append({'universe': uni, 'book': name, 'period': p, **ls_stats(ls[m])})
    size_books = pd.DataFrame(size_rows)
    size_books.to_parquet(out / 'size_books.parquet', index=False)

    books.to_parquet(out / 'books.parquet', index=False)
    fm.to_parquet(out / 'fama_macbeth.parquet', index=False)
    by_sector.to_parquet(out / 'by_sector.parquet', index=False)
    tilts.to_parquet(out / 'tilts.parquet', index=False)
    note = pd.DataFrame({'note': ['Sectors are today\'s Bloomberg INDUSTRY_SECTOR and SIC codes applied backwards (not point-in-time).',
                                  'Gross of all costs. Periods counted back from the last cohort date.',
                                  'Blank cells: undefined (too few cohorts).']})
    with pd.ExcelWriter(out / 'sector_slice.xlsx') as xw:
        note.to_excel(xw, sheet_name='notes', index=False)
        books.to_excel(xw, sheet_name='books', index=False)
        size_books.to_excel(xw, sheet_name='size_books', index=False)
        fm.to_excel(xw, sheet_name='fama_macbeth', index=False)
        by_sector.to_excel(xw, sheet_name='by_sector', index=False)
        tilts.to_excel(xw, sheet_name='tilts', index=False)
        pd.DataFrame([cov]).to_excel(xw, sheet_name='coverage', index=False)

    # charts
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9})
    with PdfPages(out / 'sector_slice.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        show = {'raw CNN': '#1a7f37', 'within-sector deciles (equal sector mix in every decile)': '#2f6fb3',
                'industry-demeaned (minus SIC 2-digit mean score)': '#c46a00', 'all price controls + industry': '#8a3ab9',
                'sector timing (top 3 minus bottom 3 sectors by mean score)': '#777777'}
        for c, col in show.items():
            f = books[(books.book == c) & (books.period == 'Full')].iloc[0]
            ax.plot((1 + series[c].fillna(0)).cumprod(), color=col, lw=1.5, label=f'{c}: {f.ls_ann:.1%}/yr, Sharpe {f.ls_sharpe:.2f}, t {f.ls_nw_t:.1f}')
        ax.set_yscale('log')
        ax.set_title('D10-D1 long-short, growth of $1: raw vs sector- and industry-neutral (gross, log scale)')
        ax.legend(fontsize=7.5, loc='upper left')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        full = by_sector[(by_sector.universe == 'all') & (by_sector.period == 'Full')]
        order = full.pivot(index='sector', columns='signal', values='ls_sharpe').sort_values('raw CNN').index
        for ax, p in zip(axes, ['Full', '2000-2011', '2012-2026']):
            f = by_sector[(by_sector.universe == 'all') & (by_sector.period == p)]
            piv = f.pivot(index='sector', columns='signal', values='ls_sharpe').loc[order]
            tt = f.pivot(index='sector', columns='signal', values='ls_nw_t').loc[piv.index]
            y = np.arange(len(piv))
            ax.barh(y + 0.2, piv['raw CNN'], 0.4, color='#1a7f37', label='raw CNN')
            ax.barh(y - 0.2, piv['all-controls residual'], 0.4, color='#8a3ab9', label='all-controls residual')
            for i, s in enumerate(piv.index):
                ax.text(max(piv.loc[s].max(), 0) + 0.03, i, f"t {tt.loc[s, 'raw CNN']:.1f} / {tt.loc[s, 'all-controls residual']:.1f}", va='center', fontsize=7)
            ax.set_yticks(y, piv.index)
            ax.axvline(0, color='#333', lw=0.8)
            ax.set_title(f'Within-sector D10-D1 Sharpe, {p}', fontsize=9)
            ax.legend(fontsize=7, loc='lower right')
            ax.margins(x=0.25)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5))
        piv = tilts[tilts.sector != 'Other'].pivot(index='sector', columns='year', values='D10_minus_D1') * 100
        lim = np.nanmax(np.abs(piv.to_numpy()))
        im = ax.imshow(piv.to_numpy(), cmap='RdYlGn', vmin=-lim, vmax=lim, aspect='auto')
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=90)
        ax.set_yticks(range(piv.shape[0]), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f'{piv.iat[i, j]:.0f}', ha='center', va='center', fontsize=6)
        ax.set_title('Sector share of the top decile minus share of the bottom decile, percentage points, by year (raw CNN)')
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    summary = {'char': str(char), 'coverage': cov, 'between_sector_share_of_score_variance': var_share,
               'look_ahead_note': 'sector and industry are current classifications applied backwards',
               'books_full': books[books.period == 'Full'][['book', 'ls_ann', 'ls_sharpe', 'ls_nw_t']].round(4).to_dict('records')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))

    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 60)
    print(books[['book', 'period', 'ls_ann', 'ls_sharpe', 'ls_nw_t', 'monotonicity', 'rank_corr_with_raw']].round(3).to_string(index=False))
    print(fm.round(3).to_string(index=False))
    f = by_sector[(by_sector.universe == 'all')]
    print(f.pivot_table(index=['sector', 'signal'], columns='period', values='ls_sharpe').round(2).to_string())
    print(f[f.period == 'Full'][['sector', 'signal', 'median_names', 'share_of_universe', 'ls_ann', 'ls_nw_t']].round(3).to_string(index=False))
    print(by_sector[(by_sector.universe != 'all') & (by_sector.period == 'Full')].pivot(index='sector', columns='universe', values='ls_sharpe').round(2).to_string())
    print(size_books.pivot_table(index=['universe', 'book'], columns='period', values='ls_sharpe').round(2).to_string())
    print(size_books[size_books.period == 'Full'][['universe', 'book', 'ls_ann', 'ls_sharpe', 'ls_nw_t']].round(3).to_string(index=False))
    print(tilts.groupby('sector')[['universe_share', 'D1_share', 'D10_share']].mean().round(3).to_string())
    print('OUTPUT', out)


if __name__ == '__main__':
    main()
