"""Annual rolling5 target comparison: equal-weight top50 minus bottom50.
Same eligible universe/schedules as deciles; no buffer; ADV20 >= $1m screen.
Observed outcomes retained, including outlier flags; missing outcomes disclosed.
"""
from pathlib import Path
import sys,json,time,fcntl
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from pattern.research.full_cache import atomic_json
ABS=ROOT/'runs/full_rolling_clean_2003_2026_seed101';REL=ROOT/'runs/relative_rolling5_seed101_latest'
OUT=ROOT/'runs/annual_rolling5_absolute_vs_relative/top_bottom50_adv1m';OUT.mkdir(parents=True,exist_ok=True)

def summarize(c):
 assert c.top.notna().all() and c.bottom.notna().all()
 return {'cohorts':len(c),'top_ann_arithmetic_pct':c.top.mean()*5040,'bottom_ann_arithmetic_pct':c.bottom.mean()*5040,'spread_ann_arithmetic_pct':c.spread.mean()*5040,'spread_vol_ann_pct':c.spread.std()*np.sqrt(50.4)*100,'holding_period_sharpe':c.spread.mean()/c.spread.std()*np.sqrt(50.4),'positive_spread_fraction':c.spread.gt(0).mean(),'unresolved_cohorts':int(c.strict_spread.isna().sum()),'unknown_top_holdings':int(c.top_unknown.sum()),'unknown_bottom_holdings':int(c.bottom_unknown.sum()),'flagged_top_holdings':int(c.top_flagged.sum()),'flagged_bottom_holdings':int(c.bottom_flagged.sum())}

def build():
 allc={};assignments={};audit=[];pending=[];screens=[]
 liquidity=pd.read_parquet(ROOT/'runs/annual_rolling5_absolute_vs_relative/liquidity/adv20.parquet')
 for year in range(2003,2027):
  name=f'{year}_rolling5_seed101';ap=ABS/name;rp=REL/name
  status=rp/'status.json'
  if not status.exists() or json.loads(status.read_text()).get('state')!='complete':pending.append(year);continue
  a=pd.read_parquet(ap/'scores.parquet').sort_values(['date','security']).reset_index(drop=True)
  r=pd.read_parquet(rp/'scores.parquet').sort_values(['date','security']).reset_index(drop=True);r['forward_return']=r.absolute_forward_return
  pd.testing.assert_frame_equal(a[['date','security']],r[['date','security']]);assert np.allclose(a.forward_return,r.forward_return,equal_nan=True)
  sample=pd.read_parquet(ap/'sample.parquet',columns=['security','date','partition','return_outlier','close']);flags=sample[sample.partition.eq('comparison')].drop(columns='partition')
  for target,s in [('absolute',a),('relative',r)]:
   s=s.merge(flags,on=['security','date'],validate='one_to_one').merge(liquidity,on=['security','date'],validate='one_to_one');s['forward_return']=s.forward_return.where(np.isfinite(s.forward_return));assert np.isfinite(s.score).all() and not s.duplicated(['date','security']).any()
   before=s.groupby('date').size();s=s[s.adv20.ge(1e6)&s.valid_days.eq(20)].copy();after=s.groupby('date').size();assert before.index.equals(after.index) and after.ge(100).all()
   if target=='absolute':screens.append(pd.DataFrame({'before':before,'after':after}))
   cs=[];sels=[]
   for date,g in s.groupby('date',sort=True):
    g=g.sort_values(['score','security']);assert len(g)>=100
    bottom=g.iloc[:50].copy();top=g.iloc[-50:].copy();assert not set(top.security)&set(bottom.security)
    top['leg']='top';bottom['leg']='bottom';sels.extend([top,bottom]);t=top.forward_return;b=bottom.forward_return
    cs.append({'date':date,'top':t.mean(),'bottom':b.mean(),'spread':t.mean()-b.mean(),'strict_spread':t.mean()-b.mean() if t.notna().all() and b.notna().all() else np.nan,'top_unknown':t.isna().sum(),'bottom_unknown':b.isna().sum(),'top_flagged':top.return_outlier.sum(),'bottom_flagged':bottom.return_outlier.sum()})
   c=pd.DataFrame(cs).set_index('date');sel=pd.concat(sels);assert sel.groupby(['date','leg']).size().eq(50).all()
   allc.setdefault(target,[]).append(c);assignments.setdefault(target,[]).append(sel)
  audit.append({'year':year,'matched_score_rows':len(a),'dates':a.date.nunique()})
 annual=[];pooled=[];overlap=[]
 for target,parts in allc.items():
  c=pd.concat(parts).sort_index();c.to_csv(OUT/f'{target}_cohorts.csv');sel=pd.concat(assignments[target],ignore_index=True);sel.to_parquet(OUT/f'{target}_holdings.parquet',index=False);assignments[target]=sel
  for year,g in c.groupby(c.index.year):annual.append({'year':year,'target':target,**summarize(g)})
  for label,mask in [('all_completed',np.ones(len(c),bool)),('2003_2019',c.index.year<=2019),('2020_onward',c.index.year>=2020)]:pooled.append({'period':label,'target':target,**summarize(c.loc[mask])})
 pd.concat(screens).to_csv(OUT/'screen_counts.csv')
 annual=pd.DataFrame(annual);annual.to_csv(OUT/'annual_metrics.csv',index=False);pd.DataFrame(pooled).to_csv(OUT/'summary.csv',index=False)
 wide=annual.pivot(index='year',columns='target',values='spread_ann_arithmetic_pct').reindex(range(2003,2027));wide['status']=['pending' if y in pending else ('partial' if y==2026 else 'complete') for y in wide.index];wide.to_csv(OUT/'year_by_year.csv')
 sets={k:{(d,leg):set(g.security) for (d,leg),g in s.groupby(['date','leg'])} for k,s in assignments.items()}
 assert sets['absolute'].keys()==sets['relative'].keys()
 for (date,leg),a in sets['absolute'].items():b=sets['relative'][(date,leg)];overlap.append({'date':date,'leg':leg,'common_names':len(a&b),'fraction':len(a&b)/50})
 pd.DataFrame(overlap).to_csv(OUT/'holdings_overlap.csv',index=False)
 meta={'pending_years':pending,'matched_years':len(audit),'checks':audit,'construction':'Annual trailing-five-year models, 50 highest vs 50 lowest scores; 50 names per leg every date, equal weights, no turnover buffer; ADV20 >= $1m after original eligibility and five-session next-open returns as decile analysis. No costs. Dollar volume is raw close times volume averaged across 20 observations through formation close; all 20 valid required. Outlier returns retained.','missing_policy':'Rank first; observed leg mean excludes unknown outcomes. Strict spread is missing if any selected return is missing.','annualization':'Arithmetic mean spread times 252/5; not CAGR. Holding-period Sharpe; annual schedules restart in January, not a continuous NAV simulation.','time':pd.Timestamp.now(tz='UTC').isoformat()};atomic_json(OUT/'verification.json',meta)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(13,5));x=np.arange(24);ax.bar(x-.2,wide.absolute,.4,color='#596b7d',label='Absolute target');ax.bar(x+.2,wide.relative,.4,color='#198475',label='Relative target');ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,wide.index.astype(str),rotation=60);ax.set_ylabel('Annualized arithmetic top50 − bottom50 spread (%)');ax.set_title('Annual trailing-five-year models • 50 stocks per side; ADV20 ≥ $1m');ax.legend();ax.grid(axis='y',alpha=.2);fig.text(.01,.01,'Gross observed outcomes; 2026 partial. Pending years: '+str(pending));fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'annual_comparison.png',dpi=140);plt.close(fig)
 print(pd.DataFrame(pooled).round(4).to_string(index=False),flush=True)
 return pending
if __name__=='__main__':
 lock=(OUT/'compare.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pending=json.loads((OUT/'verification.json').read_text())['pending_years'] if '--wait-only' in sys.argv else build()
 while pending and '--wait-only' in sys.argv:
  time.sleep(60)
  if any((REL/f'{y}_rolling5_seed101/status.json').exists() and json.loads((REL/f'{y}_rolling5_seed101/status.json').read_text()).get('state')=='complete' for y in pending):pending=build()
