"""
Script: share_page.py
Version: 1.0 (2026-10-17)

Description:
    Builds a one-page, phone-friendly HTML summary of the R3000 CNN chart strategy for sharing
    with friends: what it does, the headline numbers and four charts (growth of $1, the decile
    staircase, calendar-year long-short returns, Sharpe by slice). Every number is read from
    the finished research outputs; nothing is typed by hand. It also renders two real 64x60
    input images (the highest- and lowest-ranked example for one well-known stock) with the
    same renderer the model was trained on, embedded as PNG data URIs. Light mode only.

    Headline book = the staggered book (20 equal tranches, each rebuilt every 20 sessions on
    its own start date), equal-weight decile 10 minus decile 1 of ~3,000 point-in-time Russell
    3000 members, gross of costs, 2000-02 to 2026-06, monthly measure.

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z/
        offsets_20261015T011336Z/{staggered_daily.parquet, staggered_stats.parquet}
        decile_profile_20261014T214954Z/deciles_by_period.parquet
        full_db_scores_20261017T034904Z/Y<year>.parquet   (example images)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (-> pattern_daily_v2_20261010 prices, quality)

Output files:
    <run>/share_page_<UTC timestamp>/pattern_cnn_share.html
    <run>/share_page_<UTC timestamp>/share_data.json   (every number the page shows)

Usage:
    .venv/bin/python research/r3000_headline/share_page.py

Dependencies: numpy, pandas, Pillow; pattern renderer (unmodified).
"""
from pathlib import Path
import base64
import io
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pattern.imaging.fast import load_renderer  # noqa: E402
from prepare import resolve_dataset  # noqa: E402
from train import load_panel  # noqa: E402

RUN = ROOT / 'research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z'
OFF = RUN / 'offsets_20261015T011336Z'
DEC = RUN / 'decile_profile_20261014T214954Z'
FULL = RUN / 'full_db_scores_20261017T034904Z'
EXAMPLE = 'AAPL US Equity'
EXAMPLE_YEAR = 2025


def png_uri(img):
    """uint8 (64,60) -> 4x nearest-neighbour PNG data URI."""
    im = Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def examples():
    d = pd.read_parquet(FULL / f'Y{EXAMPLE_YEAR}.parquet', columns=['security', 'date', 'score', 'r3000_member'])
    d = d[d.r3000_member]
    d['pct'] = d.groupby('date').score.rank(pct=True)
    s = d[d.security == EXAMPLE].sort_values('pct')
    picks = {'sell': s.iloc[0], 'buy': s.iloc[-1]}
    root, _ = resolve_dataset()
    p = load_panel(root, [EXAMPLE])
    render = load_renderer(RUN / 'compiled')
    ohlcv = np.ascontiguousarray(p[['open', 'high', 'low', 'close', 'volume']].to_numpy(float))
    tri = np.ascontiguousarray(p.total_return_index_gross_dividends.to_numpy(float))
    out = {}
    for k, r in picks.items():
        row = int(np.flatnonzero(p.date.to_numpy() == np.datetime64(r.date))[0])
        img = render(ohlcv, tri, np.array([row], dtype='int64'), 20, 20)[0, 0]
        out[k] = {'date': str(r.date.date()), 'pct': round(float(r.pct) * 100, 1), 'score': round(float(r.score), 3),
                  'png': png_uri(img)}
    return out


def main():
    st = pd.read_parquet(OFF / 'staggered_stats.parquet')
    daily = pd.read_parquet(OFF / 'staggered_daily.parquet')
    stat = lambda book, period, leg: st[(st.book == book) & (st.period == period) & (st.leg == leg)].iloc[0]

    raw = daily[daily.book == 'all names, raw'].set_index('date')
    sn = daily[daily.book == 'all names, sector-neutral'].set_index('date').LS.rename('LS_sn')
    df = raw.join(sn)
    growth = (1 + df[['D10', 'D1', 'EW', 'LS', 'LS_sn']]).cumprod()
    gm = growth.resample('ME').last()
    years = (1 + df[['LS', 'LS_sn']]).groupby(df.index.year).prod() - 1

    dec = pd.read_parquet(DEC / 'deciles_by_period.parquet')
    dec = dec[(dec.variant == 'all, bankrupt at last price') & (dec.period == 'Full')].set_index('decile')

    slices = []
    for label, book in [('All stocks', 'all names'), ('Large caps', 'large caps'), ('Small caps', 'small caps')]:
        slices.append({'label': label, 'raw': round(float(stat(f'{book}, raw', 'Full', 'LS').sharpe), 2),
                       'sn': round(float(stat(f'{book}, sector-neutral', 'Full', 'LS').sharpe), 2)})
    periods = []
    for per in ['2000-2011', '2012-2026', '5y', '3y', '1y']:
        r = stat('all names, raw', per, 'LS')
        periods.append({'period': per, 'ann_return': round(float(r.ann_return) * 100, 1), 'sharpe': round(float(r.sharpe), 2)})

    ls, snl = stat('all names, raw', 'Full', 'LS'), stat('all names, sector-neutral', 'Full', 'LS')
    d10, ew = stat('all names, raw', 'Full', 'D10'), stat('all names, raw', 'Full', 'EW')
    data = {
        'span': [str(df.index.min().date()), str(df.index.max().date())],
        'headline': {'ls_ann': round(float(ls.ann_return) * 100, 1), 'ls_vol': round(float(ls.ann_vol) * 100, 1),
                     'ls_sharpe': round(float(ls.sharpe), 2), 'ls_mdd': round(float(ls.max_drawdown) * 100, 0),
                     'ls_t': round(float(ls.nw_t), 1), 'ls_cagr': round(float(ls.cagr) * 100, 1),
                     'sn_ann': round(float(snl.ann_return) * 100, 1), 'sn_sharpe': round(float(snl.sharpe), 2),
                     'sn_mdd': round(float(snl.max_drawdown) * 100, 0),
                     'd10_cagr': round(float(d10.cagr) * 100, 1),
                     'd1_cagr': round(float(stat('all names, raw', 'Full', 'D1').cagr) * 100, 1),'ew_cagr': round(float(ew.cagr) * 100, 1),
                     'd10_growth': round(float(growth.D10.iloc[-1]), 1), 'ew_growth': round(float(growth.EW.iloc[-1]), 1),
                     'ls_growth': round(float(growth.LS.iloc[-1]), 1)},
        'growth': {'dates': [d.strftime('%Y-%m') for d in gm.index],
                   **{c: [round(float(v), 4) for v in gm[c]] for c in gm.columns}},
        'years': {'year': [int(y) for y in years.index], 'LS': [round(float(v) * 100, 1) for v in years.LS],
                  'LS_sn': [round(float(v) * 100, 1) for v in years.LS_sn]},
        'deciles': {'decile': [str(i) for i in range(1, 11)],
                    'ann_return': [round(float(dec.loc[str(i), 'ann_return']) * 100, 1) for i in range(1, 11)],
                    'ew': round(float(dec.loc['EW', 'ann_return']) * 100, 1)},
        'slices': slices, 'periods': periods,
        'positive_years': int((years.LS > 0).sum()), 'n_years': int(len(years)),
        'examples': examples(),
    }
    out = RUN / f"share_page_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%SZ}"
    out.mkdir()
    (out / 'share_data.json').write_text(json.dumps({k: v for k, v in data.items() if k != 'examples'}, indent=1))
    tpl = (Path(__file__).parent / 'share_page_template.html').read_text()
    html = tpl.replace('/*__DATA__*/null', json.dumps(data))
    for k, v in data['headline'].items():
        html = html.replace('{{' + k + '}}', f'{v:g}' if isinstance(v, float) else str(v))
    for k in ('buy', 'sell'):
        for f in ('date', 'pct', 'png', 'score'):
            html = html.replace('{{' + f'{k}_{f}' + '}}', str(data['examples'][k][f]))
    html = (html.replace('{{positive_years}}', str(data['positive_years'])).replace('{{n_years}}', str(data['n_years']))
            .replace('{{p1_sharpe}}', f"{data['periods'][0]['sharpe']:g}").replace('{{p2_sharpe}}', f"{data['periods'][1]['sharpe']:g}")
            .replace('{{y3_ann}}', f"{data['periods'][3]['ann_return']:g}").replace('{{y1_ann}}', f"{data['periods'][4]['ann_return']:g}")
            .replace('{{small_sharpe}}', f"{data['slices'][2]['raw']:g}").replace('{{large_sharpe}}', f"{data['slices'][1]['raw']:g}"))
    if '{{' in html:
        raise ValueError('unfilled placeholder: ' + html[html.index('{{'):html.index('{{') + 40])
    (out / 'pattern_cnn_share.html').write_text(html)
    print(out / 'pattern_cnn_share.html')
    print(json.dumps(data['headline'], indent=1), data['examples']['buy']['date'], data['examples']['buy']['pct'],
          data['examples']['sell']['date'], data['examples']['sell']['pct'], data['positive_years'], data['n_years'])


if __name__ == '__main__':
    main()
