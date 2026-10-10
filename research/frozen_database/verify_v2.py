"""
Script: verify_v2.py
Version: 2.0 (2026-10-10)

Description:
    Independently verifies a Pattern v2 frozen snapshot (pattern_daily_v2_*) without
    touching GDM or any vendor. Checks every file hash in manifest.json, that the
    daily tables (prices, nominal_prices, quality, calendar_quality) share exactly
    the same security/date keys, that screening flags behave as documented, that the
    exchange calendar is strictly increasing, and that no fixed-window fields leaked in.

Input files: <snapshot dir>/manifest.json and every file it inventories, e.g.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/pattern_daily_v2_20261010/
Output files: none (returns / prints a JSON result; freeze_v2.py saves it as verification.json)

Usage:
    .venv/bin/python verify_v2.py "/Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/pattern_daily_v2_20261010"
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def verify(root):
    root = Path(root)
    m = json.loads((root / 'manifest.json').read_text())
    checks = []
    for name, info in m['files'].items():
        assert sha(root / name) == info['sha256'], name
    checks.append('file hashes')
    p = pd.read_parquet(root / 'prices.parquet', columns=['security', 'date'])
    assert len(p) == m['rows'] and p.security.nunique() == m['securities']
    assert not p.isna().any().any() and not p.duplicated().any()
    p = p.sort_values(['security', 'date']).reset_index(drop=True)
    checks.append('price keys unique and complete')
    for name in ['nominal_prices.parquet', 'quality.parquet', 'calendar_quality.parquet']:
        o = pd.read_parquet(root / name, columns=['security', 'date']).sort_values(['security', 'date']).reset_index(drop=True)
        assert len(o) == len(p) and (o.security.astype(str).values == p.security.astype(str).values).all() \
            and (o.date.values == p.date.values).all(), name
        checks.append(f'{name} keys == price keys')
    for name in ['quality.parquet', 'calendar_quality.parquet']:
        cols = pq.read_schema(root / name).names
        assert not any(t in c for c in cols for t in ('_20_', '_39_', 'past_', 'forward_', 'target_')), name
    checks.append('no fixed-window fields')
    q = pd.read_parquet(root / 'quality.parquet', columns=[
        'return_pair_ok', 'screened_total_return', 'current_renderer_bar_ok', 'ohlc_nonpositive',
        'ohlc_inconsistent', 'off_exchange_calendar', 'bar_ok_on_calendar'])
    assert q.loc[~q.return_pair_ok, 'screened_total_return'].isna().all()
    assert not q.loc[q.ohlc_nonpositive | q.ohlc_inconsistent, 'current_renderer_bar_ok'].any()
    assert (q.bar_ok_on_calendar == (q.current_renderer_bar_ok & ~q.off_exchange_calendar)).all()
    checks.append('screening flags consistent')
    c = pd.read_parquet(root / 'calendar_quality.parquet')
    assert (c.off_exchange_calendar == c.session_index.isna()).all()
    s = pd.read_parquet(root / 'sessions.parquet')
    assert s.date.is_monotonic_increasing and not s.date.duplicated().any()
    assert (s.session_index.values == np.arange(len(s))).all()
    on = c.dropna(subset=['session_index'])
    d = on.session_index.astype('int64').to_numpy()
    same = on.security.to_numpy()[1:] == on.security.to_numpy()[:-1]
    seg_same = on.calendar_segment.to_numpy()[1:] == on.calendar_segment.to_numpy()[:-1]
    # Inside one calendar segment, consecutive rows are consecutive sessions.
    assert (np.diff(d)[same & seg_same] == 1).all()
    checks.append('calendar segments contiguous')
    assert m['lookback_sessions'] is None and m['forecast_horizon_sessions'] is None
    assert m['source_rows_dropped'] == 0 and m['vendor_calls'] == 0
    checks.append('manifest invariants')
    return {'passed_checks': len(checks), 'checks': checks, 'rows_verified': len(p),
            'all_file_hashes_match': True,
            'scope': 'Frozen artifact integrity and screening behaviour; not independent vendor price certification'}


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1]), indent=2))
