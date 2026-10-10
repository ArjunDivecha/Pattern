"""
Script: freeze_v2.py
Version: 2.0 (2026-10-10)

Description:
    Builds Pattern's second frozen daily research database, pattern_daily_v2_20261010,
    from the rebuilt Global Data Mart (GDM) June 30, 2026 research vintage. That vintage
    is survivorship-certified for the Russell 3000 from 2000-01-01, covering 9,510 names.
    v1 (pattern_daily_v1_20260918) is left untouched.

    Differences from v1:
      - prices come from the rebuilt vintage (37.6M rows, 9,470 names vs 29.6M / 6,751);
      - the session calendar is the real NYSE calendar (exchange_calendars XNYS), not
        the empirical ">=100 active names" approximation;
      - the quality layer is recomputed here, with the same v1.0 rules, on all rows;
      - new tables: point-in-time Russell 3000 membership (full 11,011-name RAY file
        and the vintage's own filtered copy), traded (nominal) prices reconstructed by
        GDM, delisting outcomes, identity segments and the GDM vintage manifest.

    Nothing is filtered, floored, clipped or winsorized. Every source row is kept and
    flagged. No vendor (Bloomberg) calls. The build refuses to overwrite an existing
    version, writes to a .building staging dir and renames atomically at the end.

Input files (read-only, SHA256 checked before and after):
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/prices.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/security_master.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/membership.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/universe.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/manifest.json
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/ray_membership_pit.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/research_extras/nominal_prices.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/research_extras/delisting_outcomes.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/research_extras/identity_flags.parquet
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/research_extras/build_summary.json
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/frozen_database/CONTRACT_v2.md
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/frozen_database/verify_v2.py

Output files (all inside the new snapshot dir, made read-only):
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/pattern_daily_v2_20261010/
        prices.parquet, nominal_prices.parquet, quality.parquet, calendar_quality.parquet,
        sessions.parquet, calendar_evidence.parquet, off_calendar_rows.parquet,
        membership_pit.parquet, membership_vintage.parquet, universe.parquet,
        security_master.parquet, delisting_outcomes.parquet, identity_flags.parquet,
        gdm_manifest.json, gdm_research_extras_summary.json, build_summary.json,
        manifest.json, verification.json, README.md, freeze_v2.py, verify_v2.py
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/CURRENT.json (repointed to v2 with --make-current)
    Build log: stdout (run with tee into a log file)

Dependencies: pandas, numpy, pyarrow, exchange_calendars (4.13.2)

Usage:
    .venv/bin/python research/frozen_database/freeze_v2.py --make-current 2>&1 | tee <log>

Notes:
    The quality rules are imported from research/data_audit_20260917/build_quality_layer.py
    (quality_layer(), rule version 1.0) so v1 and v2 flags mean exactly the same thing.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import exchange_calendars as xc

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GDM = ROOT.parent / 'Global Data Mart' / 'sources/equities/data'
VINTAGE = GDM / 'vintages/2026-06-30'
EXTRAS = GDM / 'research_extras'
DEST = ROOT / 'data/frozen/pattern_daily_v2_20261010'
EXPECTED_GDM_MANIFEST = 'f190f267ea447bb5b06e752df1fe6568c2d300d6dcbba95d8a36c232a0e2a896'
CALENDAR = 'XNYS'
sys.path.insert(0, str(ROOT / 'research/data_audit_20260917'))
from build_quality_layer import quality_layer, RULE_VERSION, GAP_DAYS  # noqa: E402

T0 = time.time()


def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(df, path):
    tmp = path.with_suffix('.tmp.parquet')
    df.to_parquet(tmp, index=False, compression='zstd')
    tmp.replace(path)


def calendar_flags(q, sessions):
    """Session positions on the exchange calendar. All rows kept; nothing imputed.

    calendar_segment breaks at a missing session or a >180-day identity boundary. A chart or label window is valid only inside one segment.
    """
    out = q[['security', 'date']].copy()
    pos = pd.Series(np.arange(len(sessions), dtype='int32'), index=pd.DatetimeIndex(sessions))
    out['session_index'] = q.date.map(pos).astype('Int32')
    out['off_exchange_calendar'] = out.session_index.isna()
    on = out.loc[~out.off_exchange_calendar, ['security', 'session_index']]
    delta_on = on.session_index - on.groupby('security', sort=False).session_index.shift()
    out['missing_sessions_before'] = pd.Series(pd.NA, index=out.index, dtype='Int32')
    out.loc[on.index, 'missing_sessions_before'] = (delta_on - 1).clip(lower=0).astype('Int32')
    first_on = delta_on.isna()  # first on-calendar row of a security
    # Off-calendar rows are ignored by session-reindexed windows, so they do not break
    # a segment; only a missing session or an identity boundary does.
    brk = pd.Series(False, index=out.index)
    brk.loc[on.index] = delta_on.ne(1) | first_on
    brk = brk | q.unresolved_identity_boundary.to_numpy()
    out['calendar_segment'] = brk.groupby(out.security, sort=False).cumsum().astype('int32')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--make-current', action='store_true')
    args = ap.parse_args()
    if DEST.exists():
        raise FileExistsError(f'{DEST} exists; frozen versions are never overwritten')
    staging = DEST.with_name(DEST.name + '.building')
    if staging.exists():
        raise FileExistsError(f'{staging} exists from an interrupted build; inspect and remove it by hand')
    gm = digest(VINTAGE / 'manifest.json')
    if gm != EXPECTED_GDM_MANIFEST:
        raise ValueError(f'GDM vintage manifest changed ({gm}); re-check the vintage before freezing')
    staging.mkdir(parents=True)
    provenance = {}
    sources = {
        'prices.parquet': VINTAGE / 'prices.parquet',
        'security_master.parquet': VINTAGE / 'security_master.parquet',
        'membership_vintage.parquet': VINTAGE / 'membership.parquet',
        'universe.parquet': VINTAGE / 'universe.parquet',
        'gdm_manifest.json': VINTAGE / 'manifest.json',
        'membership_pit.parquet': GDM / 'ray_membership_pit.parquet',
        'nominal_prices.parquet': EXTRAS / 'nominal_prices.parquet',
        'delisting_outcomes.parquet': EXTRAS / 'delisting_outcomes.parquet',
        'identity_flags.parquet': EXTRAS / 'identity_flags.parquet',
        'gdm_research_extras_summary.json': EXTRAS / 'build_summary.json',
    }
    before = {}
    for name, src in sources.items():
        before[name] = digest(src)
        shutil.copyfile(src, staging / name)
        assert digest(staging / name) == before[name], name
        provenance[name] = {'source': str(src), 'source_sha256': before[name]}
        log(f'copied {name}')

    log('loading prices')
    raw = pd.read_parquet(staging / 'prices.parquet')
    q = quality_layer(raw)
    log(f'quality layer built: {len(q):,} rows')
    assert len(q) == len(raw)
    keys = q[['security', 'date']]

    first, last = raw.date.min(), raw.date.max()
    cal = xc.get_calendar(CALENDAR, start=str(first.date()), end=str(last.date()))
    sessions = pd.DatetimeIndex(cal.sessions.tz_localize(None) if cal.sessions.tz else cal.sessions)
    c = calendar_flags(q, sessions)
    log(f'calendar flags built on {len(sessions):,} {CALENDAR} sessions')

    # Calendar evidence: names priced on every session, and every row dated off-calendar.
    active = (np.isfinite(raw.close) & raw.close.gt(0) & np.isfinite(raw.volume) & raw.volume.gt(0))
    per_day = pd.DataFrame({'date': raw.date, 'active': active}).groupby('date').agg(
        observed_names=('active', 'size'), active_names=('active', 'sum')).reset_index()
    ev = pd.DataFrame({'date': sessions}).merge(per_day, on='date', how='left')
    ev[['observed_names', 'active_names']] = ev[['observed_names', 'active_names']].fillna(0).astype('int64')
    ev['session_index'] = np.arange(len(ev), dtype='int32')
    off = q.loc[c.off_exchange_calendar, ['security', 'date']].merge(
        raw[['security', 'date', 'close', 'volume']], on=['security', 'date'])
    off['weekday'] = off.date.dt.day_name()
    del raw

    q = q.merge(c[['security', 'date', 'off_exchange_calendar']], on=['security', 'date'], validate='one_to_one')
    q['bar_ok_on_calendar'] = q.current_renderer_bar_ok & ~q.off_exchange_calendar
    write(q, staging / 'quality.parquet')
    write(c, staging / 'calendar_quality.parquet')
    write(pd.DataFrame({'date': sessions, 'session_index': np.arange(len(sessions), dtype='int32')}),
          staging / 'sessions.parquet')
    write(ev, staging / 'calendar_evidence.parquet')
    write(off, staging / 'off_calendar_rows.parquet')
    log('quality and calendar tables written')

    # Cross-checks recorded in build_summary.json; hard failures are asserts.
    nom = pd.read_parquet(staging / 'nominal_prices.parquet', columns=['security', 'date', 'close_nominal', 'basis_source'])
    nom = nom.sort_values(['security', 'date']).reset_index(drop=True)
    assert len(nom) == len(keys) and (nom.security.astype(str).values == keys.security.astype(str).values).all() \
        and (nom.date.values == keys.date.values).all(), 'nominal_prices keys differ from prices keys'
    mem = pd.read_parquet(staging / 'membership_pit.parquet')
    priced = set(keys.security.unique())
    mem['priced'] = mem.security.isin(priced)
    m2000 = mem[mem.date >= '2000-01-01']
    seg_q = int(q.groupby('security').segment_id.max().add(1).sum())
    seg_gdm = len(pd.read_parquet(staging / 'identity_flags.parquet'))
    sessions_low = ev[(ev.date >= '2000-01-01') & (ev.active_names < 1000)]
    summary = {
        'dataset_id': DEST.name, 'quality_rule_version': RULE_VERSION,
        'identity_gap_threshold_calendar_days': GAP_DAYS, 'calendar': CALENDAR,
        'exchange_calendars_version': xc.__version__,
        'rows': len(q), 'securities': int(keys.security.nunique()),
        'flags': {k: int(q[k].sum()) for k in q.select_dtypes('bool').columns},
        'off_exchange_calendar_rows': int(c.off_exchange_calendar.sum()),
        'off_exchange_calendar_rows_since_2000': int((c.off_exchange_calendar & c.date.ge('2000-01-01')).sum()),
        'rows_with_missing_sessions_before': int(c.missing_sessions_before.fillna(0).gt(0).sum()),
        'sessions_since_2000_with_under_1000_active_names': len(sessions_low),
        'nominal_prices_keys_match_prices': True,
        'nominal_basis_rows': nom.basis_source.value_counts().to_dict(),
        'membership_pit_names': int(mem.security.nunique()),
        'membership_pit_month_ends': int(mem.date.nunique()),
        'membership_pit_member_months_since_2000_priced_share': float(m2000.priced.mean()),
        'membership_pit_names_never_priced': int(mem.groupby('security').priced.max().eq(False).sum()),
        'identity_segments_quality_layer': seg_q, 'identity_segments_gdm': seg_gdm,
        'source_rows_dropped': 0, 'vendor_calls': 0,
    }
    (staging / 'build_summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    log('build summary: ' + json.dumps({k: summary[k] for k in list(summary)[:12] if k != 'flags'}, default=str))

    for name, src in sources.items():
        assert digest(src) == before[name], f'source changed during build: {src}'
    shutil.copyfile(Path(__file__), staging / 'freeze_v2.py')
    shutil.copyfile(HERE / 'verify_v2.py', staging / 'verify_v2.py')
    shutil.copyfile(HERE / 'CONTRACT_v2.md', staging / 'README.md')
    manifest = {
        'dataset_id': DEST.name, 'status': 'FROZEN_RESEARCH_BASELINE', 'version': 2,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'rows': len(keys), 'securities': summary['securities'],
        'first_date': str(keys.date.min().date()), 'last_date': str(keys.date.max().date()),
        'source_rows_dropped': 0, 'vendor_calls': 0,
        'lookback_sessions': None, 'forecast_horizon_sessions': None,
        'horizons': 'Chosen independently per experiment; no precomputed labels or fixed window filters',
        'calendar_status': f'Exchange calendar {CALENDAR} from exchange_calendars {xc.__version__}',
        'coverage_status': 'GDM survivorship certificate for the Russell 3000, 2000-01-01 to 2026-06-30; 1996-1999 not certified',
        'gdm_vintage_manifest_sha256': gm,
        'supersedes': 'pattern_daily_v1_20260918 (kept unchanged)',
        'sources': provenance, 'files': {},
    }
    for f in sorted(staging.iterdir()):
        manifest['files'][f.name] = {'sha256': digest(f), 'bytes': f.stat().st_size}
        if f.suffix == '.parquet':
            manifest['files'][f.name]['rows'] = pq.ParquetFile(f).metadata.num_rows
            manifest['files'][f.name]['schema'] = str(pq.read_schema(f))
    (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    log('manifest written; verifying')
    sys.path.insert(0, str(HERE))
    from verify_v2 import verify
    result = verify(staging)
    (staging / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    staging.rename(DEST)
    for f in DEST.iterdir():
        f.chmod(0o444)
    if args.make_current:
        pointer = DEST.parent / 'CURRENT.json'
        (DEST.parent / 'CURRENT.tmp.json').write_text(json.dumps(
            {'dataset_id': DEST.name, 'path': str(DEST), 'manifest_sha256': digest(DEST / 'manifest.json')}, indent=2) + '\n')
        (DEST.parent / 'CURRENT.tmp.json').replace(pointer)
        log(f'CURRENT.json now points at {DEST.name}')
    log('DONE ' + json.dumps(result))


if __name__ == '__main__':
    main()
