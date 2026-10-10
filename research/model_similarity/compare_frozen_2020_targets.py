"""Compare absolute and relative I20/R5 models trained on identical 2015-2019 rows.
Both 2020 checkpoints remain fixed throughout 2020-2025. No fitting.
"""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from pattern.models.cnn import ChartCNN
from pattern.config import ModelConfig,ImageConfig
from pattern.research.development import evaluate
from pattern.research.full_cache import atomic_json
ABS=ROOT/'runs/full_rolling_clean_2003_2026_seed101';REL=ROOT/'runs/relative_rolling5_seed101_latest';OUT=ROOT/'runs/frozen_2020_rolling5_absolute_vs_relative';OUT.mkdir(exist_ok=True)
name='2020_rolling5_seed101';sa=pd.read_parquet(ABS/name/'sample.parquet');sr=pd.read_parquet(REL/name/'sample.parquet');pd.testing.assert_frame_equal(sa[['security','date','partition']],sr[['security','date','partition']]);assert np.array_equal(np.isfinite(sa.forward_return_5),np.isfinite(sr.forward_return_5))
cal=pd.DatetimeIndex(pd.read_parquet(ABS/'candidate_sessions.parquet').date)
modela=ChartCNN(ModelConfig(),ImageConfig(window=20,width=60));modelr=ChartCNN(ModelConfig(),ImageConfig(window=20,width=60))
device=torch.device('mps');torch.set_num_threads(2)
models={'absolute':modela,'relative':modelr};norms={}
for key,base in [('absolute',ABS),('relative',REL)]:
 models[key].load_state_dict(torch.load(base/name/'final.pt',map_location='cpu',weights_only=True));models[key].to(device).eval()
 if key=='absolute':
  n=json.loads((ROOT/'runs/model_similarity_25stocks_2020_2025'/name/'complete.json').read_text());norms[key]=(np.float32(n['mean']),np.float32(n['std']))
 else:
  c=torch.load(base/name/'last.pt',map_location='cpu',weights_only=False);norms[key]=(np.float32(c['normalization_mean']),np.float32(c['normalization_std']));del c
assert np.allclose(norms['absolute'],norms['relative'])
atomic_json(OUT/'manifest.json',{'training_start':'2015-01-02','training_end':'2019-10-29','training_rows':807922,'model_year':2020,'evaluation_years':list(range(2020,2026)),'policy':'Both checkpoints fixed; same stock-date rows; raw next-open returns; 10 equal-count tiles; original annual schedules; no costs','annualization':'252/5 multiplied by mean cohort return; arithmetic, not CAGR','missing':'Observed available returns after ranking; strict spread withheld for missing leg returns'})
rows=[];allc={};t=time.time()
for year in range(2020,2026):
 p=ABS/f'{year}_rolling5_seed101';s=pd.read_parquet(p/'sample.parquet');ids=np.flatnonzero(s.partition.eq('comparison'));sample=s.iloc[ids][['security','date','forward_return_5']].rename(columns={'forward_return_5':'forward_return'}).reset_index(drop=True);images=np.load(p/'images.npy',mmap_mode='r')
 for key,model in models.items():
  f=OUT/f'{key}_{year}_scores.parquet'
  if not f.exists():
   if year==2020:
    original=pd.read_parquet((ABS if key=='absolute' else REL)/name/'scores.parquet');x=sample.merge(original[['security','date','score']],on=['security','date'],validate='one_to_one');assert len(x)==len(sample)
   else:
    mean,std=norms[key];scores=np.empty(len(ids),dtype='float32')
    with torch.inference_mode():
     for a in range(0,len(ids),512):
      ix=ids[a:a+512];v=torch.from_numpy((images[ix].astype('float32')-mean)/std).to(device);scores[a:a+len(ix)]=model(v).softmax(1)[:,1].cpu().numpy()
      if a%25600==0:atomic_json(OUT/'status.json',{'state':'scoring','model':key,'year':year,'rows':a+len(ix),'of':len(ids),'seconds':time.time()-t})
    assert np.isfinite(scores).all();x=sample.copy();x['score']=scores
   x.to_parquet(f,index=False)
  x=pd.read_parquet(f);metrics,tables=evaluate(x,cal,5,pd.DatetimeIndex(x.date.unique()),10);metrics['basis']='Raw next-open outcomes; $1 long/$1 short; no costs; observed outcomes'
  rows.append({'model':key,'year':year,**metrics});tables['cohorts'].to_parquet(OUT/f'{key}_{year}_cohorts.parquet');allc.setdefault(key,[]).append(tables['cohorts']);print(year,key,'complete',round(time.time()-t),flush=True)
 del images
pd.DataFrame(rows).to_csv(OUT/'by_year.csv',index=False)
pooled=[]
for key,cs in allc.items():
 c=pd.concat(cs);pooled.append({'model':key,'cohorts':len(c),'top':c.top.mean(),'bottom':c.bottom.mean(),'spread':c.spread.mean(),'annualized_arithmetic_spread':c.spread.mean()*50.4,'holding_period_sharpe':c.spread.mean()/c.spread.std()*np.sqrt(50.4),'unresolved_cohorts':int(c.strict_spread.isna().sum())});c.to_parquet(OUT/f'{key}_cohorts.parquet')
pd.DataFrame(pooled).to_csv(OUT/'summary.csv',index=False);atomic_json(OUT/'status.json',{'state':'complete','seconds':time.time()-t});print(pd.DataFrame(pooled).to_string(index=False),flush=True)
