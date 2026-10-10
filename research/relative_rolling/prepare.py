"""Prepare relative labels without modifying the frozen panel or absolute runs."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import duckdb
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pattern.research.full_cache import atomic_json
BASE=ROOT/'runs/full_rolling_clean_2003_2026_seed101'
OUT=ROOT/'runs/relative_rolling5_seed101_latest'

def main():
 OUT.mkdir(exist_ok=True)
 cfg=json.loads((BASE/'config.json').read_text())
 con=duckdb.connect()
 ep=ROOT/'data/research_universe/clean_I20_R5_v1/eligibility.parquet'
 bench=con.execute('''SELECT date, count(*) eligible_n, count(next_open_return_5_raw) known_n,
 avg(next_open_return_5_raw) benchmark_return, sum(CASE WHEN return_outlier THEN 1 ELSE 0 END) flagged_n
 FROM read_parquet(?) WHERE formation_eligible GROUP BY date ORDER BY date''',[str(ep)]).fetchdf()
 bench.to_parquet(OUT/'benchmark.parquet',index=False)
 def convert(s):
  s=s.merge(bench,on='date',how='left',validate='many_to_one',sort=False)
  s['absolute_forward_return_5']=s.forward_return_5
  s['forward_return_5']=s.absolute_forward_return_5-s.benchmark_return
  return s
 jobs=[]
 for i,j in enumerate(cfg['jobs']):
  j={**j,'worker':f'local{i%2}','target':'relative_outperformance'}
  source=BASE/j['name'];dest=OUT/j['name'];dest.mkdir(exist_ok=True)
  s0=pd.read_parquet(source/'sample.parquet');s=convert(s0)
  assert np.array_equal(s.sample_id,s0.sample_id)
  assert np.array_equal(s.loc[s.partition.eq('train'),'forward_return_5'].notna(),s0.loc[s0.partition.eq('train'),'forward_return_5'].notna())
  s.to_parquet(dest/'sample.parquet',index=False)
  if not (dest/'images.npy').exists(): (dest/'images.npy').symlink_to(source/'images.npy')
  j['label_changed_fraction']=float(((s.forward_return_5>0)!=(s.absolute_forward_return_5>0))[s.partition.eq('train')].mean())
  atomic_json(dest/'prepared.json',j);jobs.append(j)
 cal=pd.DatetimeIndex(pd.read_parquet(BASE/'candidate_sessions.parquet').date)
 pd.DataFrame({'date':cal}).to_parquet(OUT/'candidate_sessions.parquet',index=False)
 latest=cal.max();start=cal.searchsorted(latest-pd.DateOffset(years=5))
 d=pd.read_parquet(ep,filters=[('date','>=',cal[start])]);pos=cal.get_indexer(d.date)
 # Anchor latest fit to the most recent fully observed label formation date.
 valid=d.training_eligible.fillna(False)&d.exit_date.le(latest)
 endpos=int(pos[valid].max())
 tr=d[valid&((endpos-pos)%5==0)].copy();tr['partition']='train';tr['forward_return_5']=tr.next_open_return_5_clean
 live=d[d.date.eq(latest)&d.formation_eligible].copy();live['partition']='live';live['forward_return_5']=np.nan
 s=pd.concat([tr,live],ignore_index=True);s['sample_id']=np.arange(len(s));s=convert(s)
 j=dict(name='latest_rolling5_seed101',year=2026,method='rolling5',seed=101,worker='local0',target='relative_outperformance',
 start=str(tr.date.min().date()),train_end=str(tr.date.max().date()),score_date=str(latest.date()),train_rows=len(tr),evaluation_rows=len(live),deployment=True)
 p=OUT/j['name'];p.mkdir(exist_ok=True);s.to_parquet(p/'sample.parquet',index=False)
 from research.full_history.train_single import SourceImages
 source=SourceImages(s);m=np.lib.format.open_memmap(p/'images.npy',mode='w+',dtype='uint8',shape=(len(s),1,64,60))
 for a in range(0,len(s),2048):
  ids=np.arange(a,min(a+2048,len(s)));m[ids]=source.get(ids)
 m.flush();del m
 atomic_json(p/'prepared.json',j)
 jobs.insert(0,j)
 cfg.update(jobs=jobs,target='relative_outperformance',benchmark_policy='Equal-weight mean of raw known five-session next-open returns over all formation-eligible stocks each date; missing outcomes omitted and counts disclosed; flagged extremes retained. Training stock exclusions unchanged.',latest_data_date=str(latest.date()),historical_comparison='24 matched rolling annual fits; extra latest fit is deployment only')
 atomic_json(OUT/'config.json',cfg)
 atomic_json(OUT/'preflight.json',dict(state='prepared',fits=len(jobs),matched_fits=24,unchanged_historical_samples=True,unchanged_images=True,latest=j,training_started=False))
 print(json.dumps({'fits':len(jobs),'latest':j,'label_changes':[x.get('label_changed_fraction') for x in jobs[1:]]},indent=2),flush=True)
if __name__=='__main__':main()
