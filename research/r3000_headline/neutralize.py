"""
Script: neutralize.py
Version: 1.1 (2026-10-14)

Description:
    Is the CNN's decile staircase a new signal, or a known factor in disguise? For the R3000
    headline run (expanding retraining, 2000-2026), this builds the standard price-based
    characteristics for every scored stock at every cohort formation date, then:
      1. Fama-MacBeth: per cohort date, regress the realised 20-session simple return on
         rank-normalised signals (each mapped to [-0.5, 0.5], so a coefficient is a
         bottom-to-top return spread), univariate and with controls; Newey-West t across dates.
      2. Neutralised deciles: per date, regress the rank-normalised CNN score on the controls
         (contemporaneous cross-section, as a risk model would), keep the residual, and rebuild
         the full ten-decile profile and the D10-D1 book from it, control group by group and
         all together (with and without sector dummies).
      3. What the deciles hold: average characteristics of each raw CNN decile.
    Same conventions as report.py and verify_expanding.py: cohorts every 20th session from the
    first test session, equal weight, deciles over all names, returns over known outcomes,
    delisted names earn the return to their last price, gross of all costs.

    Characteristics (all from data through the formation close, the same information as the image):
      ret_5d, ret_1m       past 5- and 20-session total return (reversal)
      mom_12_1             total return from 252 to 21 sessions back (momentum)
      vol_60d              std of daily total returns, 60 sessions
      ivol_60d, beta_60d   residual std and beta against the equal-weight daily market, 60 sessions
      max_21d              largest daily return in the past 21 sessions (lottery)
      log_mcap             point-in-time restated shares x split-adjusted close (mcap.py; v1.0 used
                           the traded close, which inflated companies that later split)
      log_dv_60d           log mean daily dollar volume (traded close x volume), 60 sessions
      log_price            log traded close (low-price / distress proxy)
      dist_52wh            close / highest close of the past 252 sessions (George-Hwang)
      sector               GICS sector from the security master (current, not point-in-time;
                           missing = its own 'Unknown' group)
    A missing characteristic is set to the neutral rank (0) for that row; coverage is reported.
    No accounting-based distress measure is available in the snapshot.

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010):
        prices.parquet, quality.parquet, nominal_prices.parquet, security_master.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/index.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/fundamentals.parquet (BS_SH_OUT)
    <run dir>/Y<year>/{COMPLETE.json, ensemble_scores.parquet}
Output files (in <run dir>/neutralize_<UTC timestamp>/):
    characteristics.parquet     one row per scored stock per cohort date: score, return, characteristics
    fama_macbeth.parquet        spec x regressor: mean coefficient (% per 20 sessions), NW t
    neutralized.parquet         spec x universe: D10-D1 stats, ten-decile excess profile, monotonicity, IC
    decile_characteristics.parquet  average characteristics by raw CNN decile
    neutralize.xlsx             all tables above (except the row-level file)
    neutralize.pdf              charts (light mode)
    summary.json; neutralize.log (tee'd by caller)

Usage:
    .venv/bin/python research/r3000_headline/neutralize.py --run-dir DIR

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import resolve_dataset, H  # noqa: E402
from train import load_panel  # noqa: E402
from verify_fixed import nw_t, PPY  # noqa: E402
from mcap import market_caps  # noqa: E402

FUND = ROOT.parent / 'Global Data Mart/sources/equities/data/vintages/2026-06-30/fundamentals.parquet'
DEC = [f'D{k}' for k in range(1, 11)]
GROUPS = {
    'volatility (vol, ivol, beta)': ['vol_60d', 'ivol_60d', 'beta_60d'],
    'lottery (max daily return)': ['max_21d'],
    'reversal (5d, 1m)': ['ret_5d', 'ret_1m'],
    'momentum (12-1)': ['mom_12_1'],
    'size and liquidity': ['log_mcap', 'log_dv_60d'],
    'distress proxies (price, 52w high)': ['log_price', 'dist_52wh'],
}
ALL = [c for g in GROUPS.values() for c in g]
T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)


def rolling_by_security(values, sec_pos, window, how):
    """Rolling statistic over rows of a panel sorted by security then date; NaN until a full window."""
    s = pd.Series(values)
    r = getattr(s.rolling(window, min_periods=window), how)()
    r[sec_pos < window - 1] = np.nan
    return r.to_numpy()


def build_characteristics(root, sc):
    secs = sorted(sc.security.unique())
    log(f'loading daily panel for {len(secs)} securities')
    p = load_panel(root, secs)
    nom = pd.read_parquet(root / 'nominal_prices.parquet', columns=['security', 'date', 'close_nominal'],
                          filters=[('security', 'in', secs)])
    nom['security'] = nom.security.astype(str)
    p = p.merge(nom, on=['security', 'date'], how='left', validate='one_to_one')
    p = p.sort_values(['security', 'date']).reset_index(drop=True)
    pos = p.groupby('security').cumcount().to_numpy()
    tri = p.total_return_index_gross_dividends.to_numpy(float)
    first = pos == 0
    r = np.empty_like(tri)
    r[1:] = tri[1:] / tri[:-1] - 1
    r[first] = np.nan
    p['r'] = r
    # equal-weight daily market (returns clipped at -50%/+100% so a bad print cannot move it)
    mkt = p.assign(rc=p.r.clip(-0.5, 1.0)).groupby('date').rc.mean()
    m = p.date.map(mkt).to_numpy(float)
    log('rolling characteristics')
    rr = np.where(np.isfinite(r), r, np.nan)
    mean_r = rolling_by_security(rr, pos, 60, 'mean')
    mean_m = rolling_by_security(m, pos, 60, 'mean')
    mean_rm = rolling_by_security(rr * m, pos, 60, 'mean')
    mean_mm = rolling_by_security(m * m, pos, 60, 'mean')
    var_r = rolling_by_security(rr, pos, 60, 'var')
    cov = mean_rm - mean_r * mean_m
    var_m = mean_mm - mean_m ** 2
    beta = cov / var_m
    p['vol_60d'] = np.sqrt(var_r)
    p['beta_60d'] = beta
    p['ivol_60d'] = np.sqrt(np.clip(var_r - beta ** 2 * var_m * 60 / 59, 0, None))
    p['max_21d'] = rolling_by_security(rr, pos, 21, 'max')
    g = p.groupby('security').total_return_index_gross_dividends
    p['ret_5d'] = tri / g.shift(5).to_numpy() - 1
    p['ret_1m'] = tri / g.shift(H).to_numpy() - 1
    p['mom_12_1'] = g.shift(21).to_numpy() / g.shift(252).to_numpy() - 1
    dv = p.close_nominal.to_numpy(float) * p.volume.to_numpy(float)
    p['log_dv_60d'] = np.log(rolling_by_security(dv, pos, 60, 'mean'))
    p['log_price'] = np.log(p.close_nominal)
    p['dist_52wh'] = p.close.to_numpy(float) / rolling_by_security(p.close.to_numpy(float), pos, 252, 'max')
    keep = ['security', 'date', 'close_nominal', 'vol_60d', 'beta_60d', 'ivol_60d', 'max_21d', 'ret_5d', 'ret_1m',
            'mom_12_1', 'log_dv_60d', 'log_price', 'dist_52wh']
    out = sc.merge(p[keep], on=['security', 'date'], how='left', validate='one_to_one')
    log('market cap')
    out = out.sort_values('date').reset_index(drop=True)
    mcap = market_caps(root, out)
    out['log_mcap'] = np.log(mcap.where(mcap > 0))
    out['cap_rank'] = mcap.groupby(out.date).rank(ascending=False, method='first')
    sm = pd.read_parquet(root / 'security_master.parquet', columns=['security', 'gics_sector_name'])
    sm['security'] = sm.security.astype(str)
    out = out.merge(sm, on='security', how='left', validate='many_to_one')
    out['sector'] = out.gics_sector_name.fillna('Unknown')
    return out.drop(columns=['gics_sector_name']).sort_values(['date', 'security']).reset_index(drop=True)


def rank_norm(d, cols):
    """Per-date percentile rank mapped to [-0.5, 0.5]; missing -> 0 (neutral)."""
    r = d.groupby('date')[cols].rank(pct=True) - 0.5
    return r.fillna(0.0)


def deciles_by_date(d, col):
    """Per date: ten decile mean returns (deciles over all names, returns over known ones) and EW."""
    rows = []
    for date, g in d.groupby('date', sort=True):
        g = g.sort_values([col, 'security'])
        dec = np.arange(len(g)) * 10 // len(g) + 1
        r = g.ret.to_numpy()
        ok = np.isfinite(r)
        if not ok.any():
            continue
        row = {'date': date, 'EW': r[ok].mean()}
        for k in range(1, 11):
            sel = (dec == k) & ok
            row[f'D{k}'] = r[sel].mean() if sel.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('date')


def book_stats(dd):
    ls = dd.D10 - dd.D1
    ex = dd[DEC].sub(dd.EW, axis=0).mean() * PPY
    return {'ls_ann': ls.mean() * PPY, 'ls_sharpe': ls.mean() / ls.std() * np.sqrt(PPY), 'ls_nw_t': nw_t(ls),
            'top3_bottom3_sharpe': (lambda x: x.mean() / x.std() * np.sqrt(PPY))(dd[DEC[7:]].mean(1) - dd[DEC[:3]].mean(1)),
            'monotonicity': float(pd.Series(dd[DEC].mean().to_numpy()).rank().corr(pd.Series(np.arange(1, 11)))),
            **{f'{k}_excess_ann': v for k, v in ex.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--index-dir', default=str(ROOT / 'cache/r3000_I20_M20_v2_index'))
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    out = run / ('neutralize_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    root, _ = resolve_dataset()
    years = sorted(int(p.name[1:]) for p in run.glob('Y*') if (p / 'COMPLETE.json').exists())
    log(f'years {years[0]}-{years[-1]} ({len(years)})')
    sc = pd.concat([pd.read_parquet(run / f'Y{y}' / 'ensemble_scores.parquet', columns=['row_id', 'security', 'date', 'session_position', 'score'])
                    for y in years])
    idx = pd.read_parquet(Path(a.index_dir) / 'index.parquet', columns=['row_id', 'forward_return_20', 'terminal_within_20', 'return_to_last'])
    sc = sc.merge(idx, on='row_id', validate='one_to_one')
    first = sc.session_position.min()
    sc = sc[(sc.session_position - first) % H == 0].copy()
    known = sc.forward_return_20.notna()
    term = sc.terminal_within_20 & sc.return_to_last.notna() & ~known
    sc['ret'] = np.where(known, sc.forward_return_20, np.where(term, sc.return_to_last, np.nan))
    sc['security'] = sc.security.astype(str)
    sc = sc[['security', 'date', 'score', 'ret']]
    d = build_characteristics(root, sc)
    d.to_parquet(out / 'characteristics.parquet', index=False)
    coverage = {c: float(d[c].notna().mean()) for c in ALL}
    coverage['sector_known'] = float((d.sector != 'Unknown').mean())
    log(f'coverage {json.dumps({k: round(v, 3) for k, v in coverage.items()})}')

    # sanity: decile book from raw scores must match the headline (verify_expanding A)
    d = d[d.date.isin(d.dropna(subset=['ret']).date.unique())].copy()
    rk = rank_norm(d, ['score'] + ALL)
    rk.columns = ['cnn'] + ALL
    rk['date'] = d.date.to_numpy()
    sectors = pd.get_dummies(d.sector, drop_first=True, dtype=float)

    # 1. Fama-MacBeth
    log('Fama-MacBeth')
    specs = {'CNN alone': ['cnn'], **{f'CNN + {g}': ['cnn'] + c for g, c in GROUPS.items()},
             'CNN + all controls': ['cnn'] + ALL, 'CNN + all controls + sector': ['cnn'] + ALL + ['SECTOR'],
             **{f'{c} alone': [c] for c in ALL}, 'all controls (no CNN)': ALL}
    y_all = d.ret.to_numpy() * 100
    dates = d.date.to_numpy()
    bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    fm = []
    for name, cols in specs.items():
        use_sec = 'SECTOR' in cols
        xc = [c for c in cols if c != 'SECTOR']
        X_all = np.column_stack([np.ones(len(d)), rk[xc].to_numpy()] + ([sectors.to_numpy()] if use_sec else []))
        coefs = []
        for i in range(len(bounds) - 1):
            s = slice(bounds[i], bounds[i + 1])
            y, X = y_all[s], X_all[s]
            ok = np.isfinite(y)
            b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
            coefs.append(b[1:1 + len(xc)])
        coefs = np.array(coefs)
        for j, c in enumerate(xc):
            fm.append({'spec': name, 'regressor': c, 'mean_coef_pct': coefs[:, j].mean(), 'nw_t': nw_t(coefs[:, j]), 'dates': len(coefs)})
    fm = pd.DataFrame(fm)
    print(fm[fm.regressor == 'cnn'].round(3).to_string(index=False), flush=True)

    # 2. Neutralised decile books
    log('neutralised deciles')
    nspecs = {'raw CNN': [], **{f'minus {g}': c for g, c in GROUPS.items()}, 'minus all controls': ALL,
              'minus all controls + sector': ALL + ['SECTOR']}
    cnn = rk.cnn.to_numpy()
    res, curves = [], {}
    for name, cols in nspecs.items():
        if cols:
            use_sec = 'SECTOR' in cols
            xc = [c for c in cols if c != 'SECTOR']
            X_all = np.column_stack([np.ones(len(d)), rk[xc].to_numpy()] + ([sectors.to_numpy()] if use_sec else []))
            resid = np.empty(len(d))
            for i in range(len(bounds) - 1):
                s = slice(bounds[i], bounds[i + 1])
                b = np.linalg.lstsq(X_all[s], cnn[s], rcond=None)[0]
                resid[s] = cnn[s] - X_all[s] @ b
        else:
            resid = cnn.copy()
        d['sig'] = resid
        corr_raw = float(d.groupby('date').apply(lambda g: g.sig.rank().corr(g.score.rank()), include_groups=False).mean())
        ic = d.dropna(subset=['ret']).groupby('date').apply(lambda g: g.sig.rank().corr(g.ret.rank()), include_groups=False)
        for uni, mask in [('all names', d.index == d.index), ('large caps (top 1000)', d.cap_rank <= 1000),
                          ('small caps (rest)', d.cap_rank > 1000)]:
            dd = deciles_by_date(d[np.asarray(mask)], 'sig')
            res.append({'spec': name, 'universe': uni, 'rank_corr_with_raw': corr_raw, 'rank_ic': ic.mean(), 'rank_ic_nw_t': nw_t(ic),
                        **book_stats(dd)})
            if uni == 'all names':
                curves[name] = dd
        log(f"{name}: LS Sharpe {res[-3]['ls_sharpe']:.2f} (t {res[-3]['ls_nw_t']:.2f}), IC t {res[-3]['rank_ic_nw_t']:.2f}, corr w raw {corr_raw:.3f}")
    res = pd.DataFrame(res)
    d = d.drop(columns=['sig'])

    # sanity check against the headline
    raw = res[(res.spec == 'raw CNN') & (res.universe == 'all names')].iloc[0]
    log(f"raw CNN all names LS Sharpe {raw.ls_sharpe:.4f} (headline report: 0.7161)")

    # 3. What the raw deciles hold
    d['raw_decile'] = d.groupby('date').score.transform(lambda s: (s.rank(method='first') - 1) * 10 // len(s) + 1).astype(int)
    chars = d.groupby('raw_decile')[ALL].median()
    chars['share_price_below_5'] = d.groupby('raw_decile').close_nominal.apply(lambda s: (s < 5).mean())
    chars['median_cap_rank'] = d.groupby('raw_decile').cap_rank.median()
    sec = pd.crosstab(d.raw_decile, d.sector, normalize='index')

    fm.to_parquet(out / 'fama_macbeth.parquet', index=False)
    res.to_parquet(out / 'neutralized.parquet', index=False)
    chars.reset_index().to_parquet(out / 'decile_characteristics.parquet', index=False)
    with pd.ExcelWriter(out / 'neutralize.xlsx') as xw:
        res.to_excel(xw, sheet_name='neutralized_books', index=False)
        fm.to_excel(xw, sheet_name='fama_macbeth', index=False)
        chars.to_excel(xw, sheet_name='decile_characteristics')
        sec.to_excel(xw, sheet_name='decile_sectors')
        pd.Series(coverage).rename('coverage').to_excel(xw, sheet_name='coverage')

    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9})
    with PdfPages(out / 'neutralize.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        show = ['raw CNN', 'minus volatility (vol, ivol, beta)', 'minus all controls', 'minus all controls + sector']
        w = 0.2
        for i, name in enumerate(show):
            r = res[(res.spec == name) & (res.universe == 'all names')].iloc[0]
            ax.bar(np.arange(1, 11) + (i - 1.5) * w, [r[f'{k}_excess_ann'] * 100 for k in DEC], width=w, label=name)
        ax.axhline(0, color='#333', lw=0.8)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel('decile (1 = lowest score, 10 = highest)')
        ax.set_ylabel('% / yr vs equal weight')
        ax.set_title('Decile excess returns: raw CNN score vs factor-neutralised residual, 2000-2026 (gross)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        for name in show:
            dd = curves[name]
            ax.plot((1 + (dd.D10 - dd.D1)).cumprod(), label=name)
        ax.set_yscale('log')
        ax.set_title('Growth of $1 in the D10-D1 book: raw vs neutralised (gross, log scale)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    summary = {'run_dir': str(run), 'cohort_dates': int(d.date.nunique()), 'rows': len(d), 'coverage': coverage,
               'raw_ls_sharpe_check': float(raw.ls_sharpe),
               'fama_macbeth_cnn': fm[fm.regressor == 'cnn'].round(4).to_dict('records'),
               'neutralized_all_names': res[res.universe == 'all names'][['spec', 'ls_ann', 'ls_sharpe', 'ls_nw_t', 'monotonicity',
                                                                          'rank_ic', 'rank_ic_nw_t', 'rank_corr_with_raw']].round(4).to_dict('records')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option('display.width', 250)
    print(res[['spec', 'universe', 'ls_ann', 'ls_sharpe', 'ls_nw_t', 'monotonicity', 'rank_ic', 'rank_ic_nw_t', 'rank_corr_with_raw',
               'D1_excess_ann', 'D10_excess_ann']].round(3).to_string(index=False))
    print(fm[fm.spec.isin(['CNN + all controls + sector', 'all controls (no CNN)'])].round(3).to_string(index=False))
    print(fm[fm.spec.str.endswith(' alone')].round(3).to_string(index=False))
    print(chars.round(3).to_string())
    print('OUTPUT', out)


if __name__ == '__main__':
    main()
