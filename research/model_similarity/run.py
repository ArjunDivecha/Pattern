"""Compare fixed rolling models on identical 2020-2025 annual schedules.
Inference only; overlap is explicitly disclosed, never presented as OOS.
"""
from pathlib import Path
import sys,json,time,itertools,fcntl
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from pattern.config import ModelConfig,ImageConfig
from pattern.models.cnn import ChartCNN
from pattern.research.full_cache import atomic_json
from pattern.research.development import evaluate
BASE=ROOT/'runs/full_rolling_clean_2003_2026_seed101'
OUT=ROOT/'runs/model_similarity_absolute_2020_2025'

def main():
 OUT.mkdir(exist_ok=True);lock=(OUT/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 cfg=json.loads((BASE/'config.json').read_text());cal=pd.DatetimeIndex(pd.read_parquet(BASE/'candidate_sessions.parquet').date)
 targets={}
 for year in range(2020,2026):
  p=BASE/f'{year}_rolling5_seed101';s=pd.read_parquet(p/'sample.parquet');ids=np.flatnonzero(s.partition.eq('comparison'))
  targets[year]=(s.iloc[ids][['security','date','forward_return_5']].rename(columns={'forward_return_5':'forward_return'}).reset_index(drop=True),ids,p)
 manifest={'models':len(cfg['jobs']),'target_years':list(targets),'target_rows':sum(len(x[0]) for x in targets.values()),'device':'cpu','threads':2,'epochs':20,'target':'absolute direction','schedule':'existing annual five-session formation schedule; gaps/overlaps across year boundaries retained','outcomes':'raw next-open returns; no new screens or costs','overlap_warning':'Models with training dates in or after evaluation are descriptive comparisons, not out-of-sample tests'}
 atomic_json(OUT/'manifest.json',manifest)
 for job in cfg['jobs']:
  name=job['name'];p=BASE/name;dest=OUT/name;dest.mkdir(exist_ok=True)
  if (dest/'complete.json').exists():continue
  assert json.loads((p/'status.json').read_text())['state']=='complete'
  t=time.time();atomic_json(OUT/'status.json',{'state':'normalizing','model':name})
  s=pd.read_parquet(p/'sample.parquet',columns=['partition','forward_return_5']);ids=np.flatnonzero(s.partition.eq('train')&np.isfinite(s.forward_return_5));images=np.load(p/'images.npy',mmap_mode='r')
  count=sum(np.count_nonzero(images[ids[a:a+1024]]) for a in range(0,len(ids),1024));fraction=count/(len(ids)*np.prod(images.shape[1:]));mean=np.float32(255*fraction);std=np.float32(255*np.sqrt(fraction*(1-fraction)));del images,s
  model=ChartCNN(ModelConfig(),ImageConfig(window=20,width=60));model.load_state_dict(torch.load(p/'final.pt',map_location='cpu',weights_only=True));model.eval()
  summaries=[]
  for year,(sample,ix,src) in targets.items():
   f=dest/f'scores_{year}.parquet'
   if not f.exists():
    images=np.load(src/'images.npy',mmap_mode='r');scores=np.empty(len(ix),dtype='float32');started=time.time()
    with torch.inference_mode():
     for a in range(0,len(ix),256):
      x=torch.from_numpy((images[ix[a:a+256]].astype('float32')-mean)/std);scores[a:a+len(x)]=model(x).softmax(1)[:,1].numpy()
      if a%25600==0:
       atomic_json(OUT/'status.json',{'state':'scoring','model':name,'year':year,'rows':min(a+256,len(ix)),'total_rows':len(ix),'seconds':time.time()-started})
    result=sample.copy();result['score']=scores;assert np.isfinite(scores).all()
    if year==job['year']:
     old=pd.read_parquet(p/'scores.parquet');check=result.merge(old,on=['security','date'],suffixes=('','_old'),validate='one_to_one');err=float(np.max(np.abs(check.score-check.score_old)));assert len(check)==len(result) and err<0.001,(name,err)
     atomic_json(dest/'reproduction_check.json',{'max_score_difference':err,'rows':len(check),'tolerance':0.001})
    result.to_parquet(f,index=False);del images
   result=pd.read_parquet(f)
   metrics,tables=evaluate(result,cal,5,pd.DatetimeIndex(result.date.unique()),10)
   metrics['basis']='Raw next-open five-session returns, gross; original annual schedule'
   summaries.append({'model':name,'model_year':job['year'],'test_year':year,'train_start':job['start'],'train_end':job['train_end'],'strictly_forward':pd.Timestamp(job['train_end'])<result.date.min(),**metrics})
   tables['cohorts'].to_parquet(dest/f'cohorts_{year}.parquet')
  pd.DataFrame(summaries).to_csv(dest/'metrics_by_year.csv',index=False);atomic_json(dest/'complete.json',{'seconds':time.time()-t,'mean':float(mean),'std':float(std)});print(name,'complete',time.time()-t,flush=True)
 summaries=pd.concat([pd.read_csv(OUT/j['name']/'metrics_by_year.csv') for j in cfg['jobs']],ignore_index=True);summaries.to_csv(OUT/'metrics_by_year.csv',index=False)
 # Stream one year at a time to bound memory; retain per-date pair diagnostics.
 pairrows=[]
 for year in targets:
  ranks={};tails={};returns={}
  for j in cfg['jobs']:
   name=j['name'];s=pd.read_parquet(OUT/name/f'scores_{year}.parquet').sort_values(['date','security']).set_index(['date','security']);ranks[name]=s.score.groupby(level='date').rank(pct=True)
   tails[name]={}
   for date,g in s.reset_index().groupby('date'):
    g=g.sort_values(['score','security']);n=len(g);tile=np.arange(n)*10//n+1
    tails[name][date]={'bottom20':set(g.security.iloc[:20]),'top20':set(g.security.iloc[-20:]),'bottom_decile':set(g.security[tile==1]),'top_decile':set(g.security[tile==10])}
   returns[name]=pd.read_parquet(OUT/name/f'cohorts_{year}.parquet').spread
  for a,b in itertools.combinations(ranks,2):
   assert ranks[a].index.equals(ranks[b].index)
   for date,x in ranks[a].groupby(level='date'):
    row={'year':year,'date':date,'model_a':a,'model_b':b,'rank_correlation':x.corr(ranks[b].loc[x.index])}
    for k in ['top20','bottom20','top_decile','bottom_decile']:
     u,v=tails[a][date][k],tails[b][date][k];row[k+'_overlap']=len(u&v)/len(u)
    pairrows.append(row)
  pd.DataFrame(returns).corr().to_csv(OUT/f'portfolio_correlation_{year}.csv')
 pairs=pd.DataFrame(pairrows);pairs.to_parquet(OUT/'pairwise_by_date.parquet',index=False)
 pairs.groupby(['model_a','model_b'])[['rank_correlation','top20_overlap','bottom20_overlap','top_decile_overlap','bottom_decile_overlap']].mean().to_csv(OUT/'pairwise_summary.csv')
 atomic_json(OUT/'status.json',{'state':'complete','models':len(cfg['jobs'])})
if __name__=='__main__':main()
