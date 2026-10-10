"""Post-outcome exclusion diagnostic; never changes predictions or training artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/annual_expanding_rolling_50k_20260918'
OUT=RUN/'outlier_review';OUT.mkdir(exist_ok=True)
config=json.loads((RUN/'config.json').read_text())
cohorts=[];excluded=[];profiles=[]
for job in config['jobs']:
 s=pd.read_parquet(RUN/job['name']/'scores.parquet').sort_values(['date','score','security']).copy()
 s['tile']=s.groupby('date').cumcount()*10//s.groupby('date').score.transform('size')+1
 for cutoff in [np.inf,2.,1.,.5]:
  x=s.copy();bad=x.forward_return.abs()>cutoff
  if cutoff==1:
   z=x.loc[bad].copy();z['job']=job['name'];excluded.append(z)
  x.loc[bad,'forward_return']=np.nan
  tile=x.groupby(['date','tile']).forward_return.mean().unstack()
  ic=x.groupby('date').apply(lambda g:g.score.corr(g.forward_return,method='spearman'),include_groups=False)
  c=pd.DataFrame({'top':tile[10],'bottom':tile[1],'ic':ic});c['spread']=c.top-c.bottom
  c['method']=job['method'];c['seed']=job['seed'];c['year']=job['year'];c['cutoff']=cutoff
  cohorts.append(c.reset_index())
  prof=tile.mean().rename('mean_return').reset_index();prof['method']=job['method'];prof['cutoff']=cutoff;prof['cohorts']=len(tile);profiles.append(prof)
  if np.isinf(cutoff):
   original=pd.read_parquet(RUN/job['name']/'cohorts.parquet')
   np.testing.assert_allclose(c.spread,original.spread,rtol=1e-10,atol=1e-12)
c=pd.concat(cohorts,ignore_index=True);c.to_csv(OUT/'cohorts.csv',index=False)
ex=pd.concat(excluded,ignore_index=True);ex.to_csv(OUT/'excluded_observations_100pct.csv',index=False)
def summary(g):
 return pd.Series({'top_ann':g.top.mean()*50.4,'bottom_ann':g.bottom.mean()*50.4,'spread_ann':g.spread.mean()*50.4,'sharpe':g.spread.mean()/g.spread.std()*np.sqrt(50.4),'ic':g.ic.mean(),'positive_fraction':g.spread.gt(0).mean(),'cohorts':len(g)})
a=c.groupby(['cutoff','method','date'],as_index=False)[['top','bottom','spread','ic']].mean()
r=a.groupby(['cutoff','method']).apply(summary,include_groups=False);r.to_csv(OUT/'summary.csv')
c[c.cutoff==1].groupby(['method','seed']).apply(summary,include_groups=False).to_csv(OUT/'seeds_100pct.csv')
a[a.cutoff==1].assign(year=lambda x:pd.to_datetime(x.date).dt.year).groupby(['year','method']).apply(summary,include_groups=False).to_csv(OUT/'years_100pct.csv')
print(r.to_string());print('Unique excluded',len(ex.drop_duplicates(['date','security'])));print('Unique evaluated',len(s.drop_duplicates(['date','security'])))
print('SEEDS');print(pd.read_csv(OUT/'seeds_100pct.csv').to_string(index=False))
print('YEARS');print(pd.read_csv(OUT/'years_100pct.csv')[['year','method','spread_ann']].to_string(index=False))
(OUT/'methodology.txt').write_text('Retrospective exclusion, not winsorization. Fixed original score deciles. Flag abs(five-day return)>cutoff; reweight remaining observed names within each leg. Equal capital across three seed portfolios, not ensemble scores. IC averaged across seed/date. Annualization mean*50.4; Sharpe std(ddof=1), sqrt(50.4). Missing outcomes remain excluded. No retraining. Baseline reconciled for all60fits. 2022 partial. Thresholds chosen after inspecting results; not an investable backtest.\n')
