"""Freeze the relative-model A/B universe, gross baseline and lagged calibration.
Only existing data is used; no broker orders or model retraining.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/cost_aware_ab_2020_2026_1m'

def monotone(values,weights):
 blocks=[]
 for i,(v,w) in enumerate(zip(values,weights)):
  blocks.append([i,i,float(v),float(w)])
  while len(blocks)>1 and blocks[-2][2]>blocks[-1][2]:
   b=blocks.pop();a=blocks.pop();w=a[3]+b[3]
   blocks.append([a[0],b[1],(a[2]*a[3]+b[2]*b[3])/w,w])
 result=np.empty(len(values))
 for lo,hi,v,w in blocks:result[lo:hi+1]=v
 return result

def build():
 OUT.mkdir(parents=True,exist_ok=True)
 liquidity=pd.read_parquet(ROOT/'runs/annual_rolling5_absolute_vs_relative/liquidity/adv20.parquet')
 panels=[]
 for y in range(2015,2027):
  rel=ROOT/f'runs/relative_rolling5_seed101_latest/{y}_rolling5_seed101'
  assert json.loads((rel/'status.json').read_text())['state']=='complete'
  s=pd.read_parquet(rel/'scores.parquet')
  f=pd.read_parquet(ROOT/f'runs/full_rolling_clean_2003_2026_seed101/{y}_rolling5_seed101/sample.parquet',columns=['security','date','partition','close','entry_date','exit_date','return_outlier'])
  s=s.merge(f[f.partition.eq('comparison')].drop(columns='partition'),on=['security','date'],validate='one_to_one')
  s=s[s.close.ge(5)].copy();s=s.sort_values(['date','score','security'])
  s['rank']=s.groupby('date').cumcount()+1;s['universe_n']=s.groupby('date').security.transform('size')
  s['rank_pct']=(s['rank']-.5)/s.universe_n
  s['bin']=(s.rank_pct*100).astype(int).clip(0,99)
  panels.append(s)
 allp=pd.concat(panels,ignore_index=True)
 test=allp[allp.date.dt.year.ge(2020)].merge(liquidity,on=['security','date'],validate='one_to_one')
 assert not test.duplicated(['security','date']).any()
 calibrations=[]
 for y,g in test.groupby(test.date.dt.year):
  cutoff=g.date.min();past=allp[(allp.date>=pd.Timestamp(y-5,1,1))&(allp.exit_date<cutoff)]
  daily=past.groupby(['date','bin']).forward_return.mean().reset_index()
  bins=daily.groupby('bin').forward_return.agg(['mean','count']).reindex(range(100))
  assert bins['mean'].notna().all() and bins['count'].gt(0).all()
  fitted=monotone(bins['mean'].to_numpy(),bins['count'].to_numpy())
  for b,v in enumerate(fitted):calibrations.append({'year':int(y),'bin':b,'expected_relative_return':v,'calibration_max_exit':past.exit_date.max(),'first_formation':cutoff,'date_bin_count':int(bins.loc[b,'count'])})
 cal=pd.DataFrame(calibrations);test['year']=test.date.dt.year
 test=test.merge(cal[['year','bin','expected_relative_return']],on=['year','bin'],validate='many_to_one')
 test.to_parquet(OUT/'eligible_panel.parquet',index=False);cal.to_csv(OUT/'calibration.csv',index=False)
 holdings=[];cohorts=[];candidates=[]
 for date,g in test.groupby('date',sort=True):
  g=g.sort_values(['score','security']);a=g[g.adv20.ge(1e6)&g.valid_days.eq(20)]
  assert len(a)>=100
  for leg,sel in [('long',a.tail(50)),('short',a.head(50))]:
   sel=sel.copy();sel['leg']=leg;holdings.append(sel)
  top=a.tail(50).absolute_forward_return;bottom=a.head(50).absolute_forward_return
  cohorts.append({'date':date,'entry_date':g.entry_date.iloc[0],'exit_date':g.exit_date.iloc[0],'long_return':top.mean(),'short_underlying_return':bottom.mean(),'gross_spread':top.mean()-bottom.mean(),'strict_gross_spread':top.mean()-bottom.mean() if top.notna().all() and bottom.notna().all() else np.nan,'missing_long':int(top.isna().sum()),'missing_short':int(bottom.isna().sum())})
  assert len(g)>=500
  for leg,sel in [('long',g.tail(250)),('short',g.head(250))]:
   sel=sel.copy();sel['leg']=leg;candidates.append(sel)
 h=pd.concat(holdings,ignore_index=True);c=pd.DataFrame(cohorts);pool=pd.concat(candidates,ignore_index=True)
 h.to_parquet(OUT/'A_holdings.parquet',index=False);c.to_csv(OUT/'A_cohorts.csv',index=False);pool.to_parquet(OUT/'B_candidates.parquet',index=False)
 req=pd.concat([h[['security','date','exit_date']],pool[['security','date','exit_date']]]).drop_duplicates(['security','date'])
 req=req.groupby('security').agg(first_formation=('date','min'),last_exit=('exit_date','max'),formation_dates=('date','nunique')).reset_index()
 req['symbol']=req.security.str.replace(' US Equity','',regex=False).str.replace('/',' ',regex=False)
 req.to_csv(OUT/'broker_requests.csv',index=False)
 old=pd.read_parquet(ROOT/'runs/annual_rolling5_absolute_vs_relative/top_bottom50_adv1m/relative_holdings.parquet');old=old[old.date.dt.year.ge(2020)].copy();old['leg']=old.leg.map({'top':'long','bottom':'short'})
 assert set(zip(h.date,h.security,h.leg))==set(zip(old.date,old.security,old.leg))
 assert h.groupby(['date','leg']).size().eq(50).all()
 stats={'first_formation':str(c.date.min().date()),'last_formation':str(c.date.max().date()),'last_exit':str(c.exit_date.max().date()),'cohorts':len(c),'gross_ann_arithmetic_pct':c.gross_spread.mean()*5040,'holding_period_sharpe':c.gross_spread.mean()/c.gross_spread.std()*np.sqrt(50.4),'missing_outcome_cohorts':int(c.strict_gross_spread.isna().sum()),'missing_long_outcomes':int(c.missing_long.sum()),'missing_short_outcomes':int(c.missing_short.sum()),'B_candidate_rows':len(pool),'broker_unique_securities':len(req),'baseline_exact_holdings_match':True,'calibration_cutoffs_verified':bool((cal.calibration_max_exit<cal.first_formation).all())}
 (OUT/'baseline_summary.json').write_text(json.dumps(stats,indent=2))
 config={'capital_per_side':1000000,'positions_per_side':50,'price_floor':5,'A_adv20_floor':1000000,'B_adv20_floor':None,'B_candidates_per_side':250,'B_selection':'50 greatest estimated net relative returns per side from fixed top/bottom250 signal candidates; equal $20k sizing','signal':'existing annual relative rolling5 seed101 models; no retraining','calibration':'100 cross-sectional score-rank bins, monotone pooled adjacent violators, daily-equal bin means from prior five years of out-of-sample predictions, all labels realized before first formation in test year','execution':'same next-open five-session holding cohorts as original; full round trips initially, no turnover-buffer change','report_columns':['A gross requested baseline','A net diagnostic','B gross diagnostic','B net requested comparison'],'costs':'observed prior formation-date or earlier bid/ask and fee history; fixed commission assumptions; explicit impact model, not actual fills','missing_cost_policy':'never zero-fill; record coverage and block primary full-period result if incomplete','borrow_availability':'historical quantities not yet sourced; cannot use current inventory historically','fee_rate_units':'unverified; net comparison blocked until resolved','latest_available_outcome':stats['last_exit'],'state':'prepared_not_backtested_B'}
 (OUT/'protocol.json').write_text(json.dumps(config,indent=2));print(json.dumps(stats,indent=2))
if __name__=='__main__':build()
