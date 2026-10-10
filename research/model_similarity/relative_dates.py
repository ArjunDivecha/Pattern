"""Small repeated-date model agreement diagnostic, no fitting or return selection."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from pattern.config import ModelConfig,ImageConfig
from pattern.models.cnn import ChartCNN
BASE=ROOT/'runs/full_rolling_clean_2003_2026_seed101'
PRIOR=ROOT/'runs/model_similarity_25stocks_2020_2025'
REL=ROOT/'runs/relative_rolling5_seed101_latest'
OUT=ROOT/'runs/relative_similarity_24dates_25stocks';OUT.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
rng=np.random.default_rng(101);rows=[];arrays=[]
for year in range(2020,2026):
 p=BASE/f'{year}_rolling5_seed101';s=pd.read_parquet(p/'sample.parquet');im=np.load(p/'images.npy',mmap_mode='r');ss=s[s.partition.eq('comparison')]
 dates=sorted(ss.date.unique());chosen=[dates[i] for i in np.linspace(0,len(dates)-1,4).round().astype(int)]
 for date in chosen:
  g=ss[ss.date.eq(date)].sort_values('security');ids=rng.choice(g.index.to_numpy(),25,replace=False);rows.append(s.loc[ids,['security','date']]);arrays.append(np.asarray(im[ids]).copy())
sample=pd.concat(rows,ignore_index=True);images=np.concatenate(arrays);assert len(sample)==600 and not sample.duplicated().any();sample.to_csv(OUT/'observations.csv',index=False)
original=pd.read_csv(ROOT/'runs/model_similarity_24dates_25stocks/observations.csv',parse_dates=['date']);pd.testing.assert_frame_equal(sample.reset_index(drop=True),original)
preds={};checks=[];start=time.time()
for year in range(2003,2027):
 name=f'{year}_rolling5_seed101'
 if json.loads((REL/name/'status.json').read_text()).get('state')!='complete':continue
 cp=torch.load(REL/name/'last.pt',map_location='cpu',weights_only=False);norm={'mean':cp['normalization_mean'],'std':cp['normalization_std']};del cp
 model=ChartCNN(ModelConfig(),ImageConfig(window=20,width=60));model.load_state_dict(torch.load(REL/name/'final.pt',map_location='cpu',weights_only=True));model.eval();chunks=[]
 with torch.inference_mode():
  for a in range(0,len(sample),100):chunks.append(model(torch.from_numpy((images[a:a+100].astype('float32')-np.float32(norm['mean']))/np.float32(norm['std']))).softmax(1)[:,1].numpy())
 score=np.concatenate(chunks);assert np.isfinite(score).all();preds[str(year)]=score
 if 2020<=year<=2025:
  old=pd.read_parquet(REL/name/'scores.parquet');v=sample.copy();v['new']=score;v=v[v.date.dt.year.eq(year)].merge(old,on=['security','date'],validate='one_to_one');err=float(abs(v.new-v.score).max());assert len(v)==100 and err<.001;print('REPRODUCTION',year,err,flush=True);checks.append({'year':year,'rows':len(v),'max_error':err})
 print(year,round(time.time()-start,1),flush=True)
pred=pd.DataFrame(preds);pd.concat([sample,pred],axis=1).to_parquet(OUT/'predictions.parquet',index=False)
records=[]
for date,ix in sample.groupby('date').groups.items():
 for method in ['pearson','spearman']:
  c=pred.loc[ix].corr(method=method);c.to_csv(OUT/f'{str(date.date())}_{method}.csv')
  for a in range(len(pred.columns)):
   for b in range(a+1,len(pred.columns)):records.append({'date':date,'method':method,'model_a':int(pred.columns[a]),'model_b':int(pred.columns[b]),'gap':int(pred.columns[b])-int(pred.columns[a]),'correlation':c.iloc[a,b],'both_forward':int(pred.columns[b])<=date.year})
r=pd.DataFrame(records);r.to_csv(OUT/'pairwise.csv',index=False)
summary=[]
for date,g in r[r.method.eq('spearman')].groupby('date'):
 summary.append({'date':str(date.date()),'all_pairs_mean':g.correlation.mean(),'adjacent_mean':g[g.gap.eq(1)].correlation.mean(),'forward_adjacent_mean':g[g.gap.eq(1)&g.both_forward].correlation.mean()})
pd.DataFrame(summary).to_csv(OUT/'by_date.csv',index=False)
(OUT/'verification.json').write_text(json.dumps({'checks':checks,'models':len(pred.columns),'dates':24,'stocks_per_date':25,'predictions':len(pred)*len(pred.columns),'seconds':time.time()-start,'selection':'25 random eligible stocks per date, identical observations across models; four evenly spaced available formation dates per year 2020-2025; seed101; no returns used'},indent=2))
print(pd.DataFrame(summary).to_string(index=False),flush=True)
