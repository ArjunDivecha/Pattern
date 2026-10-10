"""Local-only provisional market calendar and fixed-session missingness audit.
Does not certify an exchange calendar or create model labels/portfolio returns.
"""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from build_quality_layer import hash_file
ROOT = Path(__file__).resolve().parent
QUALITY = ROOT/'quality_layer_20260918T064856Z/quality.parquet'


def calendar_flags(q, sessions):
    """Retain all keys. No imputation; guards count explicit calendar positions."""
    q = q.sort_values(['security', 'date']).reset_index(drop=True)
    out = q[['security','date']].copy()
    positions = pd.Series(np.arange(len(sessions)), index=pd.DatetimeIndex(sessions))
    out['session_index'] = q.date.map(positions).astype('Int32')
    out['off_candidate_calendar'] = out.session_index.isna()
    group = out.groupby('security', sort=False)
    delta = out.session_index - group.session_index.shift()
    out['missing_sessions_before'] = (delta-1).clip(lower=0).astype('Int32')
    # Off-calendar observations and identity boundaries break continuity.
    boundary = delta.ne(1).fillna(True) | q.unresolved_identity_boundary | out.off_candidate_calendar
    out['calendar_segment'] = boundary.groupby(q.security, sort=False).cumsum().astype('int32')
    # Global segment number prevents crossing either security or gap boundaries.
    segment = boundary.to_numpy(dtype=bool).cumsum()
    valid = (q.current_renderer_bar_ok & ~out.off_candidate_calendar).astype('int8')
    for n in (20,39):
        same = np.zeros(len(q),dtype=bool)
        if len(q)>=n:
            same[n-1:] = segment[n-1:] == segment[:len(q)-n+1]
        out[f'past_{n}_sessions_complete'] = valid.rolling(n,min_periods=n).sum().eq(n).to_numpy() & same
    endpoints = (q.return_endpoint_ok & ~out.off_candidate_calendar).astype('int8')
    same = np.zeros(len(q),dtype=bool)
    if len(q)>20:
        same[:-20] = segment[:-20] == segment[20:]
    out['forward_20_sessions_complete'] = endpoints.iloc[::-1].rolling(21,min_periods=21).sum().eq(21).iloc[::-1].to_numpy() & same
    pos = out.session_index.to_numpy(dtype=float, na_value=np.nan)
    target = pos + 20
    eligible = np.isfinite(target) & (target < len(sessions))
    dates = np.full(len(q), np.datetime64('NaT'), dtype='datetime64[ns]')
    dates[eligible] = pd.DatetimeIndex(sessions).values[target[eligible].astype(int)]
    out['target_20_session_date'] = dates
    out['target_beyond_calendar'] = out.session_index.notna() & ~eligible
    return out


def main():
    sys.path.insert(0,'/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart')
    from gdm import dataset
    source = dataset('equities.research_vintage_prices',acknowledge='survivorship-incomplete')
    before = hash_file(source)
    p = pd.read_parquet(source,columns=['security','date','close','volume'])
    p['active'] = np.isfinite(p.close) & p.close.gt(0) & np.isfinite(p.volume) & p.volume.gt(0)
    daily = p.groupby('date').agg(observed_names=('security','size'),active_names=('active','sum')).reset_index()
    daily['weekday'] = daily.date.dt.dayofweek.lt(5)
    # This is explicitly an empirical threshold, not an exchange holiday rule.
    daily['candidate_session'] = daily.weekday & daily.active_names.ge(100)
    sessions = pd.DatetimeIndex(daily.loc[daily.candidate_session,'date'])
    if sessions.empty:
        raise ValueError('No candidate sessions')
    weekdays = pd.date_range(sessions.min(),sessions.max(),freq='B')
    excluded = pd.DataFrame({'date':weekdays}).merge(daily,on='date',how='left')
    excluded = excluded.loc[~excluded.candidate_session.fillna(False).astype(bool)]
    del p
    q = pd.read_parquet(QUALITY,columns=['security','date','unresolved_identity_boundary','current_renderer_bar_ok','return_endpoint_ok','chart_20_observations_ok','renderer_39_observations_ok','forward_20_observation_intervals_ok'])
    result = calendar_flags(q,sessions)
    out = ROOT/('calendar_layer_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    daily.to_csv(out/'calendar_evidence.csv',index=False)
    excluded.to_csv(out/'excluded_weekdays_review.csv',index=False)
    pd.DataFrame({'date':sessions}).to_parquet(out/'candidate_sessions.parquet',index=False)
    result.to_parquet(out/'calendar_quality.parquet',index=False)
    back = pd.read_parquet(out/'calendar_quality.parquet')
    pd.testing.assert_frame_equal(result,back)
    assert len(back)==len(q) and not back.duplicated(['security','date']).any()
    assert not back.loc[back.off_candidate_calendar,'past_20_sessions_complete'].any()
    assert not back.loc[back.target_beyond_calendar,'forward_20_sessions_complete'].any()
    assert hash_file(source)==before
    q = q.sort_values(['security','date']).reset_index(drop=True)
    summary = {'status':'PROVISIONAL: empirical calendar, not exchange-certified',
        'source_sha256_before_and_after':before,'quality_source':str(QUALITY),
        'calendar_rule':'Weekday with >=100 finite positive-close, positive-volume observations',
        'calendar_start':str(sessions.min().date()),'calendar_end':str(sessions.max().date()),
        'sessions':len(sessions),'rows':len(back),'source_rows_dropped':0,'vendor_calls':0,
        'excluded_weekdays_needing_classification':len(excluded),
        'flags':{c:int(back[c].sum()) for c in back.select_dtypes('bool')},
        'rows_with_missing_sessions_before':int(back.missing_sessions_before.gt(0).sum()),
        'observation_chart20_pass_but_session_guard_fail':int((q.chart_20_observations_ok & ~back.past_20_sessions_complete).sum()),
        'output_sha256':hash_file(out/'calendar_quality.parquet')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print('OUTPUT',out,flush=True)

if __name__=='__main__':
    main()
