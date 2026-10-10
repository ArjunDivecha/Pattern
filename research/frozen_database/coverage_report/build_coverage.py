from pathlib import Path
import pandas as pd
import json
r=Path('/Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/frozen/pattern_daily_v1_20260918')
p=pd.read_parquet(r/'prices.parquet',columns=['security','date','volume'])
q=pd.read_parquet(r/'quality.parquet',columns=['security','date','current_renderer_bar_ok','ohlc_missing_or_nonfinite','ohlc_nonpositive','ohlc_inconsistent','volume_missing_or_nonfinite','volume_zero','tri_invalid','extreme_return_gt_100pct','unresolved_identity_boundary'])
c=pd.read_parquet(r/'calendar_quality.parquet',columns=['security','date','missing_sessions_before'])
p=p.merge(q,on=['security','date'],validate='one_to_one').merge(c,on=['security','date'],validate='one_to_one')
sessions=pd.read_parquet(r/'candidate_sessions.parquet').date
p['positive_volume_usable']=p.current_renderer_bar_ok & p.volume.gt(0)
p['missing_session_before']=p.missing_sessions_before.gt(0).fillna(False)
p['year']=p.date.dt.year
flags=['current_renderer_bar_ok','positive_volume_usable','ohlc_missing_or_nonfinite','ohlc_nonpositive','ohlc_inconsistent','volume_missing_or_nonfinite','volume_zero','tri_invalid','extreme_return_gt_100pct','unresolved_identity_boundary','missing_session_before']
daily=p.groupby('date').agg(stocks=('security','size'),usable=('current_renderer_bar_ok','sum'),positive_volume_usable=('positive_volume_usable','sum'))
daily['year']=daily.index.year
d=daily.loc[daily.index.isin(sessions)]
y=d.groupby('year').agg(sessions=('stocks','size'),average_stocks=('stocks','mean'),minimum_stocks=('stocks','min'),maximum_stocks=('stocks','max'),average_usable=('usable','mean'),average_positive_volume_usable=('positive_volume_usable','mean'))
y['year_end_stocks']=d.groupby('year').stocks.last()
y['year_end_usable']=d.groupby('year').usable.last()
y['distinct_stocks_during_year']=p.groupby('year').security.nunique()
annual_flags=p.groupby('year')[flags].sum(); annual_flags['observations']=p.groupby('year').size()
out=r.parents[2]/'research/frozen_database/coverage_report';out.mkdir(exist_ok=True)
y.round(2).to_csv(out/'annual_coverage.csv')
daily.to_csv(out/'daily_coverage.csv')
annual_flags.to_csv(out/'annual_quality.csv')
print('ANNUAL\n'+y.loc[1990:].round(1).to_string())
print('EARLY\n'+y.loc[:1989].round(1).to_string())
print('QUALITY\n'+annual_flags.sum().to_string())
print('RECENT QUALITY\n'+annual_flags.loc[[1996,2000,2010,2020,2025,2026]].to_string())
print('LATEST\n'+d.tail(5).to_string())
print('PATH',out)
