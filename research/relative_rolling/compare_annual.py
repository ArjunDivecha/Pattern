"""Matched annual trailing-five-year absolute vs relative target comparison.
Rebuild deciles from scores; preserve raw outcomes and disclose missing returns.
--wait refreshes the report once the last already-authorized fit arrives.
"""
from pathlib import Path
import sys,json,time,fcntl
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from pattern.research.development import evaluate
from pattern.research.full_cache import atomic_json
ABS=ROOT/'runs/full_rolling_clean_2003_2026_seed101'
REL=ROOT/'runs/relative_rolling5_seed101_latest'
OUT=ROOT/'runs/annual_rolling5_absolute_vs_relative';OUT.mkdir(exist_ok=True)

def build():
 cal=pd.DatetimeIndex(pd.read_parquet(ABS/'candidate_sessions.parquet').date)
 rows=[];checks=[];pending=[];cohorts={};ics={}
 for year in range(2003,2027):
  name=f'{year}_rolling5_seed101';ap=ABS/name;rp=REL/name
  if not (rp/'status.json').exists() or json.loads((rp/'status.json').read_text()).get('state')!='complete':
   pending.append(year);continue
  aj=json.loads((ap/'prepared.json').read_text());rj=json.loads((rp/'prepared.json').read_text())
  for k in ['start','train_end','train_rows','evaluation_rows','seed']:assert aj[k]==rj[k],(year,k)
  traincols=['security','date','partition'];ats=pd.read_parquet(ap/'sample.parquet',columns=traincols);rts=pd.read_parquet(rp/'sample.parquet',columns=traincols);pd.testing.assert_frame_equal(ats,rts)
  a=pd.read_parquet(ap/'scores.parquet');r=pd.read_parquet(rp/'scores.parquet');r['forward_return']=r.absolute_forward_return
  keys=['date','security'];a=a.sort_values(keys).reset_index(drop=True);r=r.sort_values(keys).reset_index(drop=True)
  pd.testing.assert_frame_equal(a[keys],r[keys]);assert np.allclose(a.forward_return,r.forward_return,equal_nan=True)
  assert pd.Timestamp(aj['train_end'])<a.date.min()
  checks.append({'year':year,'matched_rows':len(a),'matched_training_rows':aj['train_rows'],'train_start':aj['start'],'train_end':aj['train_end'],'unknown_outcomes':int(a.forward_return.isna().sum()),'flagged_outcomes_retained':True})
  for target,s,prefix,path in [('absolute',a,'',ap),('relative',r,'absolute_',rp)]:
   m,t=evaluate(s[['date','security','score','forward_return']],cal,5,pd.DatetimeIndex(s.date.unique()),10)
   old=json.loads((path/(prefix+'metrics.json')).read_text());assert np.isclose(m['spread'],old['spread']),year
   c=t['cohorts'];cohorts.setdefault(target,[]).append(c);ics.setdefault(target,[]).append(t['rank_ic']);t['tile_profile'].to_csv(OUT/f'{year}_{target}_deciles.csv')
   rows.append({'year':year,'target':target,'train_start':aj['start'],'train_end':aj['train_end'],'cohorts':len(c),'top_mean_5d_pct':m['top']*100,'bottom_mean_5d_pct':m['bottom']*100,'spread_mean_5d_pct':m['spread']*100,'spread_ann_arithmetic_pct':m['spread']*50.4*100,'holding_period_sharpe':m['observed_cohort_sharpe'],'rank_ic':m['rank_ic'],'unresolved_extreme_cohorts':m['unresolved_extreme_cohorts'],'unknown_outcomes':m['unknown_outcomes'],'partial_year':year==2026})
 df=pd.DataFrame(rows);df.to_csv(OUT/'annual_metrics.csv',index=False)
 wide=df.pivot(index='year',columns='target',values=['spread_ann_arithmetic_pct','holding_period_sharpe','rank_ic']);wide.columns=['_'.join(x) for x in wide.columns];wide=wide.reindex(range(2003,2027));wide['status']=['pending' if y in pending else ('partial year' if y==2026 else 'complete') for y in wide.index];wide.to_csv(OUT/'year_by_year.csv')
 pooled=[]
 for target,parts in cohorts.items():
  c=pd.concat(parts).sort_index();ic=pd.concat(ics[target]).sort_index();c.to_parquet(OUT/f'{target}_cohorts.parquet');ic.to_parquet(OUT/f'{target}_ic.parquet')
  for label,mask in [('all_completed',np.ones(len(c),bool)),('2003_2019',c.index.year<=2019),('2020_onward',c.index.year>=2020)]:
   x=c.loc[mask];i=ic.loc[x.index];pooled.append({'period':label,'target':target,'cohorts':len(x),'top_mean_5d_pct':x.top.mean()*100,'bottom_mean_5d_pct':x.bottom.mean()*100,'spread_ann_arithmetic_pct':x.spread.mean()*50.4*100,'holding_period_sharpe':x.spread.mean()/x.spread.std()*np.sqrt(50.4),'rank_ic':i.rank_ic.mean(),'unresolved_extreme_cohorts':int(x.strict_spread.isna().sum())})
 pd.DataFrame(pooled).to_csv(OUT/'summary.csv',index=False)
 audit={'matched_years':len(checks),'pending_years':pending,'matched_scored_rows':sum(c['matched_rows'] for c in checks),'checks':checks,'policy':'Annual retraining on trailing five years, seed101, epoch20. Same stocks/dates and raw next-open five-session returns. Formation eligibility only; no new filters/costs. Missing returns disclosed and excluded from observed leg means after sorting.','annualization':'Arithmetic mean spread times 252/5, not CAGR. Sharpe is holding-period diagnostic, not daily NAV Sharpe. Annual schedules restart each January; no continuous wealth curve claimed.','generated_at':pd.Timestamp.now(tz='UTC').isoformat()};atomic_json(OUT/'verification.json',audit)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(13,5));x=np.arange(24)
 ax.bar(x-.2,wide.spread_ann_arithmetic_pct_absolute,.4,label='Absolute target',color='#596b7d');ax.bar(x+.2,wide.spread_ann_arithmetic_pct_relative,.4,label='Relative target',color='#198475');ax.axhline(0,color='black',lw=.6);ax.set_xticks(x,wide.index.astype(str),rotation=60);ax.set_ylabel('Annualized arithmetic D10−D1 spread (%)');ax.set_title('Annual trailing-five-year models • gross observed returns');ax.legend();ax.grid(axis='y',alpha=.2);fig.text(.01,.01,'2026 partial. Missing outcomes remain. '+('Pending: '+str(pending) if pending else 'All years complete.'),fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'annual_comparison.png',dpi=140);plt.close(fig)
 (OUT/'report.md').write_text('# Annual five-year rolling comparison\n\n'+audit['policy']+'\n\n'+audit['annualization']+'\n\nPending years: '+str(pending)+'\n\n'+pd.DataFrame(pooled).round(4).to_string(index=False)+'\n\n'+wide.round(3).to_string())
 print(json.dumps({'complete_year_pairs':len(checks),'pending':pending,'rows':audit['matched_scored_rows']}),flush=True)
 return pending
if __name__=='__main__':
 lock=(OUT/'compare.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pending=json.loads((OUT/'verification.json').read_text())['pending_years'] if '--wait-only' in sys.argv else build()
 while pending and ('--wait' in sys.argv or '--wait-only' in sys.argv):
  time.sleep(60)
  if any((REL/f'{y}_rolling5_seed101/status.json').exists() and json.loads((REL/f'{y}_rolling5_seed101/status.json').read_text()).get('state')=='complete' for y in pending):pending=build()
