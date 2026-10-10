"""Same-period random-substitution test; no fill or execution simulation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.full_history.tail20_analysis import load_scores
OUT=ROOT/'runs/full_history_single_2019cutoff_seed101/comparison_to_rolling5/substitution_20_v1'
PPY=50.4
REPS=1000
SEED=20260920

def stats(top,bottom):
 s=top-bottom
 wealth=np.cumprod(1+s,axis=1)
 peak=np.maximum.accumulate(np.concatenate([np.ones((len(s),1)),wealth],axis=1),axis=1)[:,1:]
 return dict(top_ann=top.mean(1)*PPY,bottom_ann=bottom.mean(1)*PPY,
 spread_ann=s.mean(1)*PPY,sharpe=s.mean(1)/s.std(1,ddof=1)*np.sqrt(PPY),
 max_drawdown=(wealth/peak-1).min(1),positive_fraction=(s>0).mean(1))

def main():
 OUT.mkdir(exist_ok=True,parents=True)
 config=dict(repetitions=REPS,random_seed=SEED,pools=[20,50,100,200],positions_per_side=20,
 selection='Uniform random 20 without replacement per side per date; independent across dates; no buffer.',
 screens='Only $5 adjusted-close floor on existing formation-eligible scores. Upstream exclusions remain.',
 basis='Observed raw next-open five-session returns; missing returns omitted within legs as in prior comparison. $1 long/$1 short; no costs.',
 interpretation='Post-hoc same-period substitution, not fills. Simulation ranges describe stock-choice variation, not confidence in future performance.')
 (OUT/'config.json').write_text(json.dumps(config,indent=2))
 frames=load_scores()
 con=duckdb.connect()
 q=con.execute("SELECT security,date,close FROM read_parquet(?)",[str(OUT.parent/'quality_tail20_buffer/quality_features.parquet')]).fetchdf()
 summaries=[]; icrows=[]; checks=[]; plot_data={}
 for model,frame in frames.items():
  frame=frame.merge(q,on=['security','date'],validate='one_to_one')
  frame=frame[frame.close.ge(5)].sort_values(['date','score','security'])
  dates=sorted(frame.date.unique()); assert len(dates)==328
  groups=[g for _,g in frame.groupby('date',sort=True)]
  exact=None
  for pool in [20,50,100,200]:
   rng=np.random.default_rng(SEED+pool)
   legs=np.empty((2,REPS,len(dates))); missing=np.zeros(REPS,dtype=int)
   expected=np.empty((2,len(dates)))
   for t,g in enumerate(groups):
    assert len(g)>=2*pool
    for leg,tail in enumerate([g.tail(pool),g.head(pool)]):
     vals=tail.forward_return.to_numpy(float)
     expected[leg,t]=np.nanmean(vals)
     idx=np.argsort(rng.random((REPS,pool)),axis=1)[:,:20]
     assert (np.diff(np.sort(idx,axis=1),axis=1)>0).all()
     selected=vals[idx]; missing+=np.isnan(selected).sum(1)
     legs[leg,:,t]=np.nanmean(selected,axis=1)
     icrows.append(dict(model=model,pool=pool,date=dates[t],leg=['top','bottom'][leg],
       rank_ic=tail.score.corr(tail.forward_return,method='spearman'),missing_pool=int(np.isnan(vals).sum())))
   st=stats(*legs)
   if pool==20:
    exact=st['spread_ann'][0]
    old=json.loads((OUT.parent/'top_bottom_20/summary.json').read_text())['models'][model]
    assert np.isclose(exact,old['spread_ann_arithmetic'])
    assert np.isclose(st['sharpe'][0],old['spread_sharpe'])
    checks.append(model+': exact top20 return and Sharpe reproduced')
   path=pd.DataFrame(st); path['missing_holdings']=missing
   path.to_csv(OUT/f'{model}_pool{pool}_paths.csv',index=False)
   np.savez_compressed(OUT/f'{model}_pool{pool}_cohorts.npz',top=legs[0],bottom=legs[1],dates=np.array(dates,dtype='datetime64[D]'))
   row=dict(model=model,pool=pool,paths=REPS,dates=len(dates),prob_positive=float((st['spread_ann']>0).mean()),
    prob_beats_exact20=float((st['spread_ann']>exact+1e-10).mean()),
    exact_pool_expected_ann=float((expected[0]-expected[1]).mean()*PPY),
    missing_holdings_mean=float(missing.mean()))
   for k,v in st.items():
    row[k+'_mean']=float(v.mean())
    for label,p in [('p05',.05),('median',.5),('p95',.95)]: row[k+'_'+label]=float(np.quantile(v,p))
   # Uniform sampling should reproduce the full pool's expected arithmetic return.
   se=st['spread_ann'].std(ddof=1)/np.sqrt(REPS)
   assert abs(row['spread_ann_mean']-row['exact_pool_expected_ann'])<max(6*se,1e-8)
   summaries.append(row)
   plot_data[model,pool]=row
  checks.append(model+': all pool means reconcile within six Monte Carlo standard errors')
 summary=pd.DataFrame(summaries); summary.to_csv(OUT/'summary.csv',index=False)
 ic=pd.DataFrame(icrows); ic.to_csv(OUT/'ic_by_date.csv',index=False)
 ic.groupby(['model','pool','leg']).rank_ic.mean().to_csv(OUT/'mean_pool_ic.csv')
 fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for ax,model in zip(axes,frames):
  sub=summary[summary.model.eq(model)]
  y=sub.spread_ann_median.to_numpy()*100
  ax.errorbar(range(4),y,yerr=np.array([y-sub.spread_ann_p05.to_numpy()*100,sub.spread_ann_p95.to_numpy()*100-y]),fmt='o-',capsize=5)
  ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(4),['Exact 20','20 of 50','20 of 100','20 of 200'])
  ax.set_title(model.replace('_',' ')); ax.grid(alpha=.2)
 axes[0].set_ylabel('Annualized long-short arithmetic return (%)')
 fig.suptitle('Same 328 dates, 2020–2026 • median and 5–95% random-selection range')
 fig.tight_layout();fig.savefig(OUT/'substitution.png',dpi=160);plt.close(fig)
 (OUT/'verification.json').write_text(json.dumps(dict(checks=checks,unique_positions=True,all_dates_328=True,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 print(summary[['model','pool','spread_ann_mean','spread_ann_p05','spread_ann_p95','sharpe_median','max_drawdown_median','positive_fraction_median','missing_holdings_mean']].to_string(index=False))
 print('CHECKS',checks)
if __name__=='__main__': main()
