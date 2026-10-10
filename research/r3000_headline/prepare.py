"""
Script: prepare.py
Version: 1.0 (2026-10-10)

Description:
    Builds the sample index and the image cache for the R3000 headline rerun: I20 charts
    (20 sessions, 20-session moving average, volume bars; 64x60 uint8) for every session
    on which a security was a point-in-time Russell 3000 member, 1996 to June 2026.

    A sample (security, date t) is kept when:
      - the security was in the Russell 3000 at the latest month-end on or before t
        (membership_pit.parquet; never a later month);
      - the 39 sessions ending at t (20 chart + 19 moving-average prehistory) are
        consecutive NYSE sessions, every bar passes bar_ok_on_calendar, and all lie in
        one calendar_segment.
    Off-calendar rows (24 since 2000, holiday prints) are dropped before windowing.

    Label columns written to the index:
      forward_return_20   TRI(t+20)/TRI(t)-1 (simple, not log), only when t+20 is a valid
                          bar in the same segment; else NaN. Training uses this only.
      terminal_within_20  True when the security's price history ends before t+20 and
                          the ending is a delisting (not the June 2026 vintage end).
      return_to_last      TRI(last bar)/TRI(t)-1 for those terminal samples. The report
                          applies an explicit terminal assumption to these, so delisted
                          holdings never silently vanish from a portfolio.
    No price floor, liquidity filter or return clipping here; those are report choices.

Input files (read-only; resolved via CURRENT.json, manifest hash checked):
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/pattern_daily_v2_20261010/
        prices.parquet, quality.parquet, calendar_quality.parquet, sessions.parquet,
        membership_pit.parquet, delisting_outcomes.parquet, manifest.json

Output files (in --cache-dir, default
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/cache/r3000_I20_M20_v2/):
    index.parquet   one row per sample: row_id, security, date, session_position,
                    member_month, forward_return_20, terminal_within_20, return_to_last,
                    white_pixels (blank until images are rendered)
    images.bin      uint8 memmap (rows, 1, 64, 60), row_id order (skipped with --index-only)
    parts/*.parquet per-security index parts (resumable build)
    state.json      resumable build state + config fingerprint (dataset manifest hash,
                    renderer source hash); a mismatch refuses to reuse the directory
    summary.json    sample counts by year, members without samples, terminal counts
    prepare.log     (tee'd by the caller)

Usage:
    .venv/bin/python research/r3000_headline/prepare.py [--cache-dir DIR] [--limit N] [--index-only]
    --limit N keeps the first N member securities (alphabetical) for a smoke test.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pattern.imaging.fast import load_renderer  # noqa: E402
from pattern.imaging.renderer import render_window  # noqa: E402

WINDOW, MA, H = 20, 20, 20
HISTORY = WINDOW + MA - 1
START = pd.Timestamp('1996-01-01')
T0 = time.time()


def log(msg):
    print(f'[{time.time()-T0:8.1f}s] {msg}', flush=True)


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def resolve_dataset():
    ref = json.loads((ROOT / 'data/frozen/CURRENT.json').read_text())
    root = Path(ref['path'])
    if hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest() != ref['manifest_sha256']:
        raise ValueError('Frozen manifest checksum mismatch')
    if json.loads((root / 'manifest.json').read_text())['version'] < 2:
        raise ValueError('R3000 headline needs the v2 snapshot (membership, NYSE calendar)')
    return root, ref


def sample_index(g, member_set_by_month, month_ends, last_vintage_date):
    """Valid I20 sample ends and labels for one security (on-calendar rows only)."""
    pos = g.session_index.to_numpy('int64')
    seg = g.calendar_segment.to_numpy()
    good = g.bar_ok_on_calendar.to_numpy(bool)
    endpoint = g.return_endpoint_ok.to_numpy(bool) & ~g.off_exchange_calendar.to_numpy(bool)
    tri = g.total_return_index_gross_dividends.to_numpy(float)
    n = len(g)
    if n < HISTORY:
        return None
    bad = np.r_[0, np.cumsum(~good)]
    ends = np.arange(HISTORY - 1, n)
    starts = ends - HISTORY + 1
    ok = (pos[ends] - pos[starts] == HISTORY - 1) & (seg[ends] == seg[starts]) & (bad[ends + 1] - bad[starts] == 0)
    dates = g.date.to_numpy()[ends]
    m = np.searchsorted(month_ends, dates, side='right') - 1
    ok &= m >= 0
    ok &= np.isin(m, member_set_by_month)
    ends, m = ends[ok], m[ok]
    if not len(ends):
        return None
    out = pd.DataFrame({'security': g.security.iat[0], 'date': g.date.to_numpy()[ends],
                        'session_position': pos[ends].astype('int32'),
                        'member_month': month_ends[m]})
    # Strict forward label: the row H sessions later, same segment, valid endpoints.
    fwd = np.full(len(ends), np.nan)
    stop = ends + H
    inside = stop < n
    v = np.flatnonzero(inside)
    e, t = ends[v], stop[v]
    # Row stop is H rows later; it must also be exactly H sessions later.
    good_pair = (pos[t] - pos[e] == H) & (seg[t] == seg[e]) & endpoint[t] & endpoint[e]
    fwd[v[good_pair]] = tri[t[good_pair]] / tri[e[good_pair]] - 1
    # Terminal: no bar at or beyond t+H sessions, and the history ends before the vintage end.
    last = n - 1
    ends_early = (pos[last] - pos[ends] < H) & (g.date.iat[last] < last_vintage_date)
    term = ends_early & np.isnan(fwd)
    ret_last = np.full(len(ends), np.nan)
    tl = term & endpoint[last] & endpoint[ends] & (seg[last] == seg[ends])
    ret_last[tl] = tri[last] / tri[ends[tl]] - 1
    fwd[~np.isfinite(fwd)] = np.nan
    out['forward_return_20'] = fwd
    out['terminal_within_20'] = term
    out['return_to_last'] = ret_last
    return out, ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', default=str(ROOT / 'cache/r3000_I20_M20_v2'))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--index-only', action='store_true')
    a = ap.parse_args()
    root, ref = resolve_dataset()
    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'parts').mkdir(exist_ok=True)
    renderer_src = ROOT / 'pattern/imaging/fast_renderer.c'
    cfg = {'dataset_id': ref['dataset_id'], 'manifest_sha256': ref['manifest_sha256'], 'window': WINDOW, 'ma': MA,
           'horizon': H, 'start': str(START.date()), 'limit': a.limit, 'index_only': a.index_only,
           'prepare_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'renderer_sha256': hashlib.sha256(renderer_src.read_bytes()).hexdigest()}
    state_path = cache / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else \
        {'config': cfg, 'last_security': None, 'rows': 0, 'stocks_processed': 0, 'state': 'building'}
    if state['config'] != cfg:
        raise ValueError('Cache configuration changed; use a new --cache-dir')
    if state['state'] == 'complete':
        log('cache already complete')
        return

    mem = pd.read_parquet(root / 'membership_pit.parquet', columns=['date', 'security'])
    month_ends = np.sort(mem.date.unique()).astype('datetime64[ns]')
    mem['m'] = np.searchsorted(month_ends, mem.date.to_numpy().astype('datetime64[ns]'))
    members = sorted(mem.security.unique())
    if a.limit:
        members = members[:a.limit]
        mem = mem[mem.security.isin(members)]
    month_sets = mem.groupby('security').m.apply(lambda s: np.sort(s.unique())).to_dict()
    last_vintage_date = pd.Timestamp(json.loads((root / 'manifest.json').read_text())['last_date'])

    filters = [('date', '>=', START), ('security', 'in', members)]
    log(f'loading v2 panel for {len(members):,} member securities')
    p = pd.read_parquet(root / 'prices.parquet', filters=filters,
                        columns=['security', 'date', 'open', 'high', 'low', 'close', 'volume', 'total_return_index_gross_dividends'])
    for name, cols in [('quality.parquet', ['security', 'date', 'bar_ok_on_calendar', 'return_endpoint_ok', 'off_exchange_calendar']),
                       ('calendar_quality.parquet', ['security', 'date', 'session_index', 'calendar_segment'])]:
        p = p.merge(pd.read_parquet(root / name, columns=cols, filters=filters), on=['security', 'date'], validate='one_to_one')
    p = p[~p.off_exchange_calendar].sort_values(['security', 'date']).reset_index(drop=True)
    p['security'] = p.security.astype(str)
    log(f'panel: {len(p):,} on-calendar rows, {p.security.nunique():,} priced members')

    render = None if a.index_only else load_renderer(cache / 'compiled')
    image_file = cache / 'images.bin'
    if not a.index_only and not image_file.exists():
        image_file.touch()
    fh = None if a.index_only else open(image_file, 'r+b')
    if fh:
        fh.truncate(state['rows'] * 64 * 3 * WINDOW)
        fh.seek(0, 2)
    started, initial, verified = time.monotonic(), state['rows'], 0
    for security, g in p.groupby('security', sort=True):
        if state['last_security'] is not None and security <= state['last_security']:
            continue
        g = g.reset_index(drop=True)
        res = sample_index(g, month_sets.get(security, np.array([], int)), month_ends, last_vintage_date)
        if res is not None:
            index, ends = res
            if fh:
                ohlcv = g[['open', 'high', 'low', 'close', 'volume']].to_numpy(float)
                tri = g.total_return_index_gross_dividends.to_numpy(float)
                counts = []
                for k in range(0, len(ends), 2048):
                    block = ends[k:k + 2048]
                    images = render(ohlcv, tri, block, WINDOW, MA)
                    if verified < 200:
                        e = int(block[0])
                        ref_img = render_window(ohlcv[e - HISTORY + 1:e + 1], tri[e - HISTORY + 1:e + 1],
                                                window=WINDOW, width=3 * WINDOW, ma_length=MA)
                        if not np.array_equal(images[0], ref_img):
                            raise AssertionError(f'Render parity failure {security} row {e}')
                        verified += 1
                    images.tofile(fh)
                    counts.extend(np.count_nonzero(images, axis=(1, 2, 3)).tolist())
                index['white_pixels'] = np.array(counts, dtype='int16')
                fh.flush()
                os.fsync(fh.fileno())
            else:
                index['white_pixels'] = np.full(len(index), -1, dtype='int16')
            index['row_id'] = np.arange(state['rows'], state['rows'] + len(index), dtype='int64')
            part = cache / 'parts' / f"{state['stocks_processed']:05d}.parquet"
            index.to_parquet(str(part) + '.tmp', index=False)
            Path(str(part) + '.tmp').replace(part)
            state['rows'] += len(index)
        state.update(last_security=security, stocks_processed=state['stocks_processed'] + 1,
                     updated_utc=pd.Timestamp.now(tz='UTC').isoformat(), render_parity_checks=verified)
        atomic_json(state_path, state)
        if state['stocks_processed'] % 250 == 0:
            rate = (state['rows'] - initial) / max(time.monotonic() - started, 1)
            log(f"{state['stocks_processed']:,} stocks, {state['rows']:,} samples, {rate:,.0f} samples/s")
    if fh:
        fh.close()

    writer = None
    for part in sorted((cache / 'parts').glob('*.parquet')):
        t = pq.read_table(part)
        writer = writer or pq.ParquetWriter(str(cache / 'index.parquet') + '.tmp', t.schema)
        writer.write_table(t)
    writer.close()
    Path(str(cache / 'index.parquet') + '.tmp').replace(cache / 'index.parquet')
    idx = pd.read_parquet(cache / 'index.parquet')
    assert np.array_equal(idx.row_id.to_numpy(), np.arange(len(idx))) and len(idx) == state['rows']
    assert not idx.duplicated(['security', 'date']).any()
    if not a.index_only:
        assert image_file.stat().st_size == state['rows'] * 64 * 3 * WINDOW
    mem2000 = mem[mem.date >= '2000-01-01']
    have = set(idx.security.unique())
    by_year = idx.groupby(idx.date.dt.year).agg(samples=('row_id', 'size'), securities=('security', 'nunique'),
                                                 dates=('date', 'nunique'), known_label=('forward_return_20', 'count'),
                                                 terminal=('terminal_within_20', 'sum'))
    by_year['avg_names_per_date'] = by_year.samples / by_year.dates
    by_year.to_csv(cache / 'samples_by_year.csv')
    summary = {'config': cfg, 'samples': len(idx), 'securities': idx.security.nunique(),
               'first_date': str(idx.date.min().date()), 'last_date': str(idx.date.max().date()),
               'known_label_share': float(idx.forward_return_20.notna().mean()),
               'terminal_within_20': int(idx.terminal_within_20.sum()),
               'terminal_with_return_to_last': int(idx.return_to_last.notna().sum()),
               'members_since_2000_without_any_sample': int(len(set(mem2000.security) - have)),
               'images_bytes': state['rows'] * 64 * 3 * WINDOW, 'render_parity_checks': verified,
               'elapsed_seconds': time.time() - T0}
    atomic_json(cache / 'summary.json', summary)
    state['state'] = 'complete'
    state['completed_utc'] = pd.Timestamp.now(tz='UTC').isoformat()
    atomic_json(state_path, state)
    log('DONE ' + json.dumps({k: v for k, v in summary.items() if k != 'config'}, default=str))
    print(by_year.to_string(), flush=True)


if __name__ == '__main__':
    main()
