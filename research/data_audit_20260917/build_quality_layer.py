"""Build a non-destructive, local-only quality sidecar for GDM prices.

Input: gdm.dataset('equities.research_vintage_prices').
Output: timestamped quality_layer_* directory beside this script.
No vendor requests, source corrections, price floors, or winsorization.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RULE_VERSION = '1.0'
GAP_DAYS = 180  # conservative screening boundary, not a proven issuer change


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def quality_layer(raw):
    required = ['security', 'date', 'open', 'high', 'low', 'close',
                'volume', 'total_return_index_gross_dividends']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if raw[['security', 'date']].isna().any().any():
        raise ValueError('Null security/date key')
    if raw.duplicated(['security', 'date']).any():
        raise ValueError('Duplicate security/date key: resolve before generating sidecar')
    p = raw[required].sort_values(['security', 'date']).reset_index(drop=True)
    q = p[['security', 'date']].copy()
    values = p[['open', 'high', 'low', 'close']]
    q['ohlc_missing_or_nonfinite'] = ~np.isfinite(values).all(axis=1)
    q['ohlc_nonpositive'] = values.le(0).any(axis=1)
    # Relative tolerance only absorbs floating point representation noise.
    tol = values.abs().max(axis=1) * 1e-10
    q['ohlc_inconsistent'] = (
        p.high.lt(p.low - tol) | p.open.gt(p.high + tol) |
        p.open.lt(p.low - tol) | p.close.gt(p.high + tol) |
        p.close.lt(p.low - tol)
    ) & ~q.ohlc_missing_or_nonfinite
    q['volume_missing_or_nonfinite'] = ~np.isfinite(p.volume)
    q['volume_negative'] = p.volume.lt(0)
    q['volume_zero'] = p.volume.eq(0)  # informational; no tradability filter
    tri = p.total_return_index_gross_dividends
    q['tri_invalid'] = ~np.isfinite(tri) | tri.le(0)
    q['price_bar_ok'] = ~(q.ohlc_missing_or_nonfinite | q.ohlc_nonpositive | q.ohlc_inconsistent)
    q['image_bar_ok'] = q.price_bar_ok & ~(q.volume_missing_or_nonfinite | q.volume_negative)
    # The current renderer uses TRI in chart normalization.
    q['current_renderer_bar_ok'] = q.image_bar_ok & ~q.tri_invalid
    group = p.groupby('security', sort=False)
    q['previous_date'] = group.date.shift()
    q['gap_days'] = (p.date - q.previous_date).dt.days.astype('Int32')
    q['unresolved_identity_boundary'] = q.gap_days.gt(GAP_DAYS).fillna(False)
    q['segment_id'] = q.groupby('security', sort=False).unresolved_identity_boundary.cumsum().astype('int32')
    # A segment id is local to security, not an asserted persistent security id.
    q['return_endpoint_ok'] = q.price_bar_ok & ~q.tri_invalid
    prev_ok = q.groupby('security', sort=False).return_endpoint_ok.shift(fill_value=False)
    q['return_pair_ok'] = q.return_endpoint_ok & prev_ok & ~q.unresolved_identity_boundary
    prev_close = group.close.shift()
    prev_tri = group.total_return_index_gross_dividends.shift()
    # Values are next-observation returns, NOT guaranteed one trading day apart.
    q['observed_price_return'] = (p.close / prev_close - 1).where(prev_close.gt(0) & p.close.gt(0))
    q['observed_total_return'] = (tri / prev_tri - 1).where(prev_tri.gt(0) & tri.gt(0))
    q['screened_price_return'] = q.observed_price_return.where(q.return_pair_ok)
    q['screened_total_return'] = q.observed_total_return.where(q.return_pair_ok)
    for col in ['observed_price_return', 'observed_total_return', 'screened_price_return', 'screened_total_return']:
        q[col] = q[col].where(np.isfinite(q[col]))
    q['extreme_return_gt_100pct'] = q.observed_total_return.abs().gt(1)
    q['short_gap_over_7d'] = q.gap_days.gt(7).fillna(False) & ~q.unresolved_identity_boundary
    q['unchanged_close_zero_volume'] = p.close.eq(prev_close) & q.volume_zero
    return q


def window_ok(q, length, forward=False):
    """Validity for contiguous OBSERVATIONS, not a trading-calendar guarantee.

    Past window includes current row. Forward length means length intervals and
    length+1 endpoints. Reject any invalid renderer row or segment boundary.
    Missing end-of-series endpoints yield False. No rows are dropped or bridged.
    """
    if length < 1:
        raise ValueError('length must be positive')
    n = length + 1 if forward else length
    grouped = q.groupby(['security', 'segment_id'], sort=False)
    out = pd.Series(False, index=q.index)
    for _, rows in grouped:
        valid = rows.current_renderer_bar_ok
        if forward:
            result = valid.iloc[::-1].rolling(n, min_periods=n).sum().eq(n).iloc[::-1]
        else:
            result = valid.rolling(n, min_periods=n).sum().eq(n)
        out.loc[rows.index] = result
    return out


def main():
    sys.path.insert(0, '/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart')
    from gdm import dataset
    source = dataset('equities.research_vintage_prices', acknowledge='survivorship-incomplete')
    source_hash = hash_file(source)
    print('Loading original rows; source SHA256', source_hash, flush=True)
    raw = pd.read_parquet(source)
    q = quality_layer(raw)
    print('Quality flags built; checking windows', flush=True)
    q['chart_20_observations_ok'] = window_ok(q, 20)
    q['renderer_39_observations_ok'] = window_ok(q, 39)
    q['forward_20_observation_intervals_ok'] = window_ok(q, 20, forward=True)
    out = ROOT / ('quality_layer_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    target = out / 'quality.parquet'
    q.to_parquet(out/'quality.tmp.parquet', index=False)
    (out/'quality.tmp.parquet').replace(target)
    flags = {c: int(q[c].sum()) for c in q.select_dtypes(include='bool').columns}
    cases = pd.read_csv(ROOT/'reviewed_cases.csv', parse_dates=['date'])
    check = cases[['case_id','category','security','date']].merge(q, on=['security','date'], how='left', validate='one_to_one')
    assert len(check) == 30 and check.price_bar_ok.notna().all()
    assert (~check.loc[check.category.eq('OHLC inconsistency'),'price_bar_ok']).all()
    assert (~check.loc[check.category.eq('Nonpositive price'),'price_bar_ok']).all()
    assert check.loc[check.category.eq('Long gap'),'unresolved_identity_boundary'].all()
    assert check.loc[check.category.eq('Long gap'),'screened_total_return'].isna().all()
    assert check.loc[check.category.isin(['Split control','Dividend control']),'return_pair_ok'].all()
    assert check.loc[check.category.eq('Extreme return'),'extreme_return_gt_100pct'].all()
    check.to_csv(out/'audit_case_validation.csv', index=False)
    assert q.loc[~q.return_pair_ok,'screened_total_return'].isna().all()
    assert np.isfinite(q.screened_total_return.dropna()).all()
    assert len(q) == len(raw)
    assert hash_file(source) == source_hash, 'Source hash changed'
    # Read back the delivered output, not just its in-memory input.
    back = pd.read_parquet(target, columns=['security','date','unresolved_identity_boundary','screened_total_return'])
    assert len(back)==len(q)
    assert back.loc[back.unresolved_identity_boundary,'screened_total_return'].isna().all()
    summary = {'rule_version':RULE_VERSION,'source_registry_key':'equities.research_vintage_prices',
               'source_path':str(source),'source_sha256_before_and_after':source_hash,
               'output_sha256':hash_file(target),'rows':len(q),'securities':q.security.nunique(),
               'flags':flags,'identity_gap_threshold_calendar_days':GAP_DAYS,
               'audit_cases_validated':30,'bloomberg_calls':0,'source_rows_dropped':0,
               'scope':'Quality sidecar only; not a certified clean or tradable panel',
               'returns':'Adjacent observed rows; blocked at invalid endpoints or >180-day boundaries',
               'unresolved':'Identity within segments, shorter gaps, adjustments, executable prices, terminal returns and tradability'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    report = ['# Price-quality layer — rules v1.0', '',
              'Raw source unchanged (SHA256 checked before and after); all original rows retained. No Bloomberg calls.', '',
              'This is a sidecar keyed by security/date. Join one-to-one to the original GDM prices. Do not treat missing screened returns as zero or drop held positions from a P&L ledger.', '',
              '## Rules applied', '',
              '- Flag missing/nonfinite/nonpositive OHLC; flag high/low/open/close contradictions with relative numerical tolerance 1e-10.',
              '- Image flags additionally require finite nonnegative volume; zero volume is flagged but not excluded. Current-renderer flags additionally require positive finite TRI.',
              '- Start an unresolved segment boundary after a gap over 180 calendar days. This is a conservative safeguard, not confirmation of ticker reuse.',
              '- Screened returns require valid price bars and positive finite TRI at both endpoints, with no unresolved boundary. Keep observed diagnostic returns separately.',
              '- Window guards require every observation valid and all observations within one segment. 20-row chart and 39-row current-renderer input guards are separate (20-bar chart plus MA prehistory). Forward guard uses 21 endpoints for 20 observation intervals.',
              '- No replacements, return clipping, price floor, dollar-volume filter, universe selection, or model changes.', '',
              '## Limits', '',
              'These are observation windows, not fixed trading-calendar horizons. Gaps of 8–180 days and unchanged-close/zero-volume observations are flagged, not resolved. Adjustment conventions, stable identity, terminal returns, and trading eligibility remain open. No valid-to-valid return bridges over a removed invalid row because no rows are removed.', '',
              '## Counts', '', '| Flag | Rows |','|---|---:|']
    report += [f'| {k} | {v:,} |' for k,v in flags.items()]
    report += ['', '## Validation', '', 'Thirty audit cases checked; entire output row count and blocked-boundary returns checked after Parquet read-back; source hash unchanged. See audit_case_validation.csv and summary.json.']
    (out/'README.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print('OUTPUT',out,flush=True)


if __name__ == '__main__':
    main()
