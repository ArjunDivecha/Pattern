from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np
from pattern.research.full_cache import atomic_json
OUT=ROOT/'runs/full_rolling_clean_2003_2026_seed101'
OUT.mkdir(exist_ok=True)
ref=json.loads((ROOT/'data/frozen/CURRENT.json').read_text());cal=pd.DatetimeIndex(pd.read_parquet(Path(ref['path'])/'candidate_sessions.parquet').date)
pd.DataFrame({'date':cal}).to_parquet(OUT/'candidate_sessions.parquet',index=False)
d=pd.read_parquet(ROOT/'data/research_universe/clean_I20_R5_v1/eligibility.parquet');pos=cal.get_indexer(d.date);assert (pos>=0).all();d['pos']=pos
maps={}
for shard in d.cache_shard.unique():
 st=json.loads((ROOT/'cache'/shard/'state.json').read_text());maps[shard]=np.memmap(ROOT/'cache'/shard/'images.bin',mode='r',dtype='uint8',shape=(st['rows'],1,64,60))
jobs=[]
for year in range(2003,2027):
 first=cal.searchsorted(f'{year}-01-01');start=cal.searchsorted(f'{year-5}-01-01');cutoff=first-44
 train=d[d.training_eligible & d.pos.between(start,cutoff)&((d.pos-start)%5==0)].copy();train['partition']='train';train['forward_return_5']=train.next_open_return_5_clean
 end=min(cal.searchsorted(f'{year+1}-01-01')-1,len(cal)-6)
 test=d[d.formation_eligible & d.pos.between(first,end)&((d.pos-first)%5==0)].copy();test['partition']='comparison';test['forward_return_5']=test.next_open_return_5_raw
 assert train.pos.max()+5<test.pos.min()-38
 assert len(train)>0 and len(test)>0
 name=f'{year}_rolling5_seed101';worker='air' if year in [2003,2009,2015,2021] else f'local{year%2}'
 job={'name':name,'year':year,'method':'rolling5','seed':101,'worker':worker,'start':str(cal[start].date()),'train_end':str(cal[cutoff].date()),'evaluation_end':str(cal[end].date()),'train_rows':len(train),'evaluation_rows':len(test)};jobs.append(job)
 p=OUT/name;p.mkdir(exist_ok=True)
 if (p/'prepared.json').exists():continue
 s=pd.concat([train,test],ignore_index=True);s['sample_id']=np.arange(len(s));s.to_parquet(p/'sample.parquet',index=False)
 m=np.lib.format.open_memmap(p/'images.npy',mode='w+',dtype='uint8',shape=(len(s),1,64,60))
 for shard,ix in s.groupby('cache_shard').groups.items():
  ids=np.asarray(ix)
  for a in range(0,len(ids),1024):
   ii=ids[a:a+1024];m[ii]=maps[shard][s.row_id.to_numpy()[ii]]
 m.flush();del m
 atomic_json(p/'prepared.json',job);print(job,flush=True)
atomic_json(OUT/'config.json',{'jobs':jobs,'dataset':ref,'epochs':20,'validation':False,'seed':101,'formation_stride':5,'clean_overlay':'clean_I20_R5_v1','evaluation':'raw outcomes; flagged outcomes disclosed; missing holdings retain strict-metric warnings','final_year_partial':True})
atomic_json(OUT/'preflight.json',{'fits':len(jobs),'unique_ownership':True,'purged':True,'all_eligible_stocks':True,'train_rows':sum(j['train_rows'] for j in jobs)})
