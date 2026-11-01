"""
Script: top1_pattern.py
Version: 1.0 (2026-10-31)

Description:
    What does a chart the CNN loves look like? For every session 2000-2026, takes the Russell 3000
    members the R3000 headline run ranked in the top 1% of that day's scores (about 30 a day), and for
    comparison the bottom 1% and a random 1% of all members, and shows:

      1. what they went on to do: mean 20-session return in excess of the day's average member, share
         that rose, share that beat the average (all formation days; overlapping, so no t-stats here);
      2. the average image the network sees (share of charts with a white pixel at each position);
      3. the average shape of the 20-day chart: median close path and inter-quartile band on the
         chart's own scale (close on day 1 = 1), position against the 20-day moving average, and the
         volume profile;
      4. simple numbers that describe the shape: return over the chart, over the last 5 days and the
         last day, 20-day volatility, distance from the moving average, where the close sits in the
         chart's high-low range, last-5-day volume against the first 15;
      5. recurring shapes inside the top 1%: k-means (k=6, numpy, 8 restarts) on the chart's log close
         path and volume shares, with each cluster's average image, typical path, share of the top 1%,
         how common the same shape is among all stocks (nearest-centroid share of the random 1%), and
         its forward excess return;
      6. a grid of real top-1% and bottom-1% inputs.

    Charts are rendered with the training renderer from the frozen database (I20, MA20), and paths use
    the total-return index, exactly as the images. Returns follow report.py: the strict 20-session
    forward return, or for a name that delists inside the horizon the return to its last price.
    Gross of costs; descriptive, no new model.

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z/Y<year>/ensemble_scores.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2_index/index.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010 prices, quality)
Output files (in <run dir>/top1_pattern_<UTC timestamp>/):
    top1_pattern.pdf       charts (light mode)
    top1_pattern.xlsx      returns by group, shape statistics, clusters
    group_returns.parquet, shape_stats.parquet, clusters.parquet, mean_images.npz, summary.json
    log tee'd by the caller to top1_pattern.log

Usage:
    .venv/bin/python research/r3000_headline/top1_pattern.py

Dependencies: numpy, pandas, matplotlib, openpyxl; pattern fast renderer; train.load_panel, prepare.resolve_dataset.
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
from pattern.imaging.fast import load_renderer  # noqa: E402
from prepare import resolve_dataset  # noqa: E402
from train import load_panel  # noqa: E402

RUN = ROOT / 'research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z'
INDEX = ROOT / 'cache/r3000_I20_M20_v2_index/index.parquet'
W, MA, HIST = 20, 20, 39
K, RESTARTS = 6, 8
GROUPS = {'top1': 'Top 1% of picks', 'rand1': 'Random 1% (all stocks)', 'bot1': 'Bottom 1% of picks'}
COL = {'top1': '#0f7a58', 'rand1': '#8a97a8', 'bot1': '#b8432f'}
T0 = time.time()


def log(m):
    print(f'[{time.time() - T0:7.1f}s] {m}', flush=True)


def kmeans(X, k, restarts, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        c = X[rng.integers(len(X))][None]
        for _ in range(1, k):                                   # k-means++ seeding
            d = ((X[:, None, :] - c[None]) ** 2).sum(-1).min(1)
            c = np.vstack([c, X[rng.choice(len(X), p=d / d.sum())]])
        for _ in range(100):
            lab = ((X[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
            nc = np.vstack([X[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(nc, c):
                break
            c = nc
        inertia = ((X - c[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, c, lab)
    return best[1], best[2]


def main():
    out = RUN / ('top1_pattern_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    root, _ = resolve_dataset()
    years = range(2000, 2027)
    sc = pd.concat([pd.read_parquet(RUN / f'Y{y}' / 'ensemble_scores.parquet', columns=['row_id', 'security', 'date', 'score']) for y in years])
    idx = pd.read_parquet(INDEX, columns=['row_id', 'forward_return_20', 'terminal_within_20', 'return_to_last'])
    sc = sc.merge(idx, on='row_id', validate='one_to_one')
    sc['security'] = sc.security.astype(str)
    known = sc.forward_return_20.notna()
    term = sc.terminal_within_20 & sc.return_to_last.notna() & ~known
    sc['ret'] = np.where(known, sc.forward_return_20, np.where(term, sc.return_to_last, np.nan))
    sc['pct'] = sc.groupby('date').score.rank(pct=True, method='first')
    rng = np.random.default_rng(7)
    sc['u'] = rng.random(len(sc))
    sc['grp'] = None
    sc.loc[sc.pct > 0.99, 'grp'] = 'top1'
    sc.loc[sc.pct <= 0.01, 'grp'] = 'bot1'
    sc.loc[sc.grp.isna() & (sc.u < 0.01), 'grp'] = 'rand1'
    log(f'scored rows {len(sc):,}; group sizes {sc.grp.value_counts().to_dict()}')

    # 1. what they went on to do
    day_mean = sc.groupby('date').ret.transform('mean')
    sc['xret'] = sc.ret - day_mean
    rows = []
    for g, lab in [('top1', 'Top 1%'), ('top10', 'Top 10%'), ('rand1', 'Random 1%'), ('bot10', 'Bottom 10%'), ('bot1', 'Bottom 1%')]:
        m = {'top10': sc.pct > 0.9, 'bot10': sc.pct <= 0.1}.get(g, sc.grp == g)
        d = sc[m & sc.ret.notna()]
        per_day = d.groupby('date').xret.mean()
        rows.append({'group': lab, 'stock_days': len(d), 'mean_20d_return': d.ret.mean(), 'mean_20d_excess': per_day.mean(),
                     'excess_annualised': per_day.mean() * 252 / 20, 'share_rose': (d.ret > 0).mean(),
                     'share_beat_average': (d.xret > 0).mean(), 'median_20d_return': d.ret.median(),
                     'excess_2000_2011': per_day[per_day.index.year <= 2011].mean() * 252 / 20,
                     'excess_2012_2026': per_day[per_day.index.year >= 2012].mean() * 252 / 20})
    gret = pd.DataFrame(rows)
    gret.to_parquet(out / 'group_returns.parquet', index=False)
    log('returns:\n' + gret.round(4).to_string(index=False))

    # 2-4. images and shapes
    sel = sc[sc.grp.notna()].reset_index(drop=True)
    p = load_panel(root, sorted(sel.security.unique()))
    p['row'] = np.arange(len(p))
    e = sel[['security', 'date']].merge(p[['security', 'date', 'row']], on=['security', 'date'], how='left', validate='one_to_one').row
    assert e.notna().all()
    e = e.to_numpy('int64')
    ohlcv = np.ascontiguousarray(p[['open', 'high', 'low', 'close', 'volume']].to_numpy(float))
    tri = np.ascontiguousarray(p.total_return_index_gross_dividends.to_numpy(float))
    del p
    render = load_renderer(out / 'compiled')
    win = e[:, None] + np.arange(-HIST + 1, 1)[None]             # (n, 39) panel rows
    TR = tri[win]
    scale = TR / ohlcv[win, 3]
    Hh, Ll, V = ohlcv[win, 1] * scale, ohlcv[win, 2] * scale, ohlcv[win, 4]
    ma = np.stack([TR[:, j - MA + 1:j + 1].mean(1) for j in range(MA - 1, HIST)], 1)   # (n, 20)
    base = TR[:, HIST - W][:, None]
    cl, hi, lo, ma_n = TR[:, -W:] / base, Hh[:, -W:] / base, Ll[:, -W:] / base, ma / base
    vol = V[:, -W:]
    vsh = vol / np.maximum(vol.sum(1, keepdims=True), 1e-12)
    dr = np.diff(np.log(TR[:, -W - 1:]), axis=1)
    shape = pd.DataFrame({
        'grp': sel.grp, 'ret_chart_20d': cl[:, -1] - 1, 'ret_last_5d': cl[:, -1] / cl[:, -6] - 1, 'ret_last_day': cl[:, -1] / cl[:, -2] - 1,
        'vol_20d_ann': dr.std(1) * np.sqrt(252), 'close_vs_ma20': cl[:, -1] / ma_n[:, -1] - 1,
        'close_in_range': (cl[:, -1] - lo.min(1)) / np.maximum(hi.max(1) - lo.min(1), 1e-12),
        'volume_last5_vs_first15': vol[:, -5:].mean(1) / np.maximum(vol[:, :15].mean(1), 1e-12),
        'last_day_range': (hi[:, -1] - lo[:, -1]) / cl[:, -1], 'days_down_of_20': (dr < 0).sum(1)})
    stats = shape.groupby('grp').agg(['median', lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    stats.columns = [f'{a}_{ {"median": "median", "<lambda_0>": "p25", "<lambda_1>": "p75"}[b] }' for a, b in stats.columns]
    stats = stats.reindex(list(GROUPS))
    stats.to_parquet(out / 'shape_stats.parquet')
    log('shape medians:\n' + shape.groupby('grp').median().reindex(list(GROUPS)).T.round(3).to_string())

    imgs = {}
    for g in GROUPS:
        ids = np.flatnonzero(sel.grp.to_numpy() == g)
        acc = np.zeros((64, 60))
        for k in range(0, len(ids), 20000):
            acc += (render(ohlcv, tri, e[ids[k:k + 20000]], W, MA)[:, 0] > 0).sum(0)
        imgs[g] = acc / len(ids)
    np.savez_compressed(out / 'mean_images.npz', **imgs)
    log('mean images rendered')

    # 5. recurring shapes in the top 1%
    lc = np.log(cl)
    X = np.hstack([lc[:, 1:], 0.5 * np.log(np.maximum(vsh * W, 1e-6))])   # day-1 close is 1 by construction
    mu, sd = X[sel.grp == 'top1'].mean(0), X[sel.grp == 'top1'].std(0)
    Z = (X - mu) / sd
    top = np.flatnonzero(sel.grp.to_numpy() == 'top1')
    sub = np.random.default_rng(1).choice(top, min(60000, len(top)), replace=False)
    cen, _ = kmeans(Z[sub], K, RESTARTS)
    lab_all = np.concatenate([((Z[i:i + 50000, None, :] - cen[None]) ** 2).sum(-1).argmin(1) for i in range(0, len(Z), 50000)])
    sel['cluster'] = lab_all
    crow = []
    rand = sel.grp == 'rand1'
    for j in range(K):
        m = (sel.grp == 'top1') & (sel.cluster == j)
        d = sel[m]
        crow.append({'cluster': j, 'share_of_top1': m.sum() / (sel.grp == 'top1').sum(),
                     'share_of_all_stocks': (rand & (sel.cluster == j)).sum() / rand.sum(),
                     'mean_20d_excess': d.xret.mean(), 'excess_annualised': d.xret.mean() * 252 / 20,
                     'share_beat_average': (d.xret.dropna() > 0).mean(),
                     **{k: float(shape.loc[m.to_numpy(), k].median()) for k in ['ret_chart_20d', 'ret_last_5d', 'ret_last_day', 'vol_20d_ann', 'close_vs_ma20', 'close_in_range', 'volume_last5_vs_first15']}})
    cl_df = pd.DataFrame(crow)
    cl_df['lift'] = cl_df.share_of_top1 / cl_df.share_of_all_stocks
    cl_df = cl_df.sort_values('share_of_top1', ascending=False).reset_index(drop=True)
    order = cl_df.cluster.tolist()
    cl_df['rank'] = np.arange(1, K + 1)
    cl_df.to_parquet(out / 'clusters.parquet', index=False)
    log('clusters:\n' + cl_df.round(3).to_string(index=False))
    cimg = {}
    for j in order:
        ids = np.flatnonzero(((sel.grp == 'top1') & (sel.cluster == j)).to_numpy())
        ids = np.random.default_rng(j).choice(ids, min(20000, len(ids)), replace=False)
        cimg[j] = (render(ohlcv, tri, e[ids], W, MA)[:, 0] > 0).mean(0)

    with pd.ExcelWriter(out / 'top1_pattern.xlsx') as xw:
        pd.DataFrame({'note': ['R3000 members 2000-2026; groups by that day\'s CNN score percentile. Returns: 20-session forward total return (delisted: return to last price), gross.',
                               'Excess = stock return minus the average member\'s that day; annualised = mean 20-day excess x 252/20 (overlapping formation days).',
                               'Shapes on the chart\'s own scale (total-return close on chart day 1 = 1). Clusters: k-means on log close path + log volume shares of the top 1%; share_of_all_stocks = nearest-centroid share of a random 1% of all members; lift = ratio.']}).to_excel(xw, sheet_name='notes', index=False)
        gret.to_excel(xw, sheet_name='returns', index=False)
        stats.reset_index().to_excel(xw, sheet_name='shape_stats', index=False)
        cl_df.to_excel(xw, sheet_name='clusters', index=False)

    # charts
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.size': 9})
    x = np.arange(1, W + 1)
    with PdfPages(out / 'top1_pattern.pdf') as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        for ax, g in zip(axes, GROUPS):
            ax.imshow(1 - imgs[g], cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
            ax.set_title(f'{GROUPS[g]}: average input image\n(darker = more charts have a mark there)', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
        for g in GROUPS:
            m = (sel.grp == g).to_numpy()
            q = np.nanpercentile(cl[m], [25, 50, 75], axis=0)
            axes[0].fill_between(x, q[0], q[2], color=COL[g], alpha=0.15)
            axes[0].plot(x, q[1], color=COL[g], lw=2, label=GROUPS[g])
            axes[1].plot(x, np.nanmedian(cl[m] / ma_n[m] - 1, 0) * 100, color=COL[g], lw=2, label=GROUPS[g])
            axes[2].plot(x, vsh[m].mean(0) * W, color=COL[g], lw=2, label=GROUPS[g])
        axes[0].set_title('Close over the 20 chart days (day 1 = 1): median and middle half', fontsize=9)
        axes[1].set_title('Close vs its 20-day moving average, % (median)', fontsize=9)
        axes[2].set_title('Volume each day vs the chart average (1 = average)', fontsize=9)
        for ax in axes:
            ax.set_xlabel('chart day (20 = signal day)'); ax.grid(alpha=0.25); ax.legend(fontsize=7)
        axes[1].axhline(0, color='black', lw=0.7); axes[2].axhline(1, color='black', lw=0.7)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.2))
        gr = gret.set_index('group')
        ax.bar(gr.index, gr.excess_annualised * 100, color=['#0f7a58', '#5fae8d', '#8a97a8', '#d9a08f', '#b8432f'])
        for i, (v, s) in enumerate(zip(gr.excess_annualised, gr.share_beat_average)):
            ax.text(i, v * 100 + (0.4 if v >= 0 else -0.9), f'{v:+.1%}/yr\nbeat avg {s:.0%}', ha='center', fontsize=8)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_ylabel('return vs the average stock, % a year')
        ax.set_title('What they went on to do: 20-day return after the signal, in excess of the average R3000 member, 2000-2026, gross', fontsize=9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, K, figsize=(16, 6.4), gridspec_kw={'height_ratios': [1.15, 1]})
        for c, j in enumerate(order):
            r = cl_df[cl_df.cluster == j].iloc[0]
            axes[0, c].imshow(1 - cimg[j], cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
            axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
            axes[0, c].set_title(f'Shape {c + 1}: {r.share_of_top1:.0%} of top 1%\n{r.lift:.1f}x as common as in all stocks', fontsize=8)
            m = ((sel.grp == 'top1') & (sel.cluster == j)).to_numpy()
            q = np.nanpercentile(cl[m], [25, 50, 75], axis=0)
            axes[1, c].fill_between(x, q[0], q[2], color='#0f7a58', alpha=0.15)
            axes[1, c].plot(x, q[1], color='#0f7a58', lw=2)
            axes[1, c].plot(x, np.nanmedian(ma_n[m], 0), color='black', lw=1, ls='--')
            axes[1, c].set_title(f'chart {r.ret_chart_20d:+.0%}, last 5d {r.ret_last_5d:+.0%}\nthen {r.excess_annualised:+.1%}/yr vs avg', fontsize=8)
            axes[1, c].grid(alpha=0.25)
        fig.suptitle('Recurring shapes inside the top 1%: average image, median close path (dashed = 20-day average), and what followed', fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        for g in ['top1', 'bot1']:
            ids = np.random.default_rng(3).choice(np.flatnonzero(sel.grp.to_numpy() == g), 24, replace=False)
            ims = render(ohlcv, tri, e[ids], W, MA)[:, 0]
            fig, axes = plt.subplots(4, 6, figsize=(13, 8))
            for ax, im, i in zip(axes.ravel(), ims, ids):
                ax.imshow(im, cmap='gray', interpolation='nearest', aspect='auto')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f'{sel.security[i].replace(" US Equity", "")} {sel.date[i].date()}\nthen {sel.ret[i]:+.0%}' if np.isfinite(sel.ret[i]) else f'{sel.security[i]} {sel.date[i].date()}', fontsize=7)
            fig.suptitle(f'{GROUPS[g]}: 24 random real inputs (as the network sees them) and the 20-day return that followed', fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    json.dump({'out': str(out), 'groups': sc.grp.value_counts().to_dict(), 'returns': gret.round(5).to_dict('records'),
               'shape_medians': shape.groupby('grp').median().round(4).to_dict(), 'clusters': cl_df.round(4).to_dict('records'),
               'elapsed_s': round(time.time() - T0, 1)}, open(out / 'summary.json', 'w'), indent=1, default=str)
    log(f'DONE {out}')


if __name__ == '__main__':
    main()
