"""Relative-target variant of the matched absolute training loop."""
from pathlib import Path
import sys,os,json,time,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from pattern.config import ModelConfig,ImageConfig
from pattern.models.cnn import ChartCNN
from pattern.research.full_cache import atomic_json
from pattern.research.checkpoint import atomic_save
from pattern.research.development import evaluate
from research.next_open_lab.run import BASIS
OUT=ROOT/'runs/relative_rolling5_seed101_latest'
def fit(job,cfg):
    p=OUT/job['name'];sample=pd.read_parquet(p/'sample.parquet');ret=sample.forward_return_5.to_numpy()
    train=np.flatnonzero(sample.partition.eq('train')&np.isfinite(ret));test=np.flatnonzero(sample.partition.isin(['comparison','live']))
    images=np.load(p/'images.npy',mmap_mode='r')
    fraction=sum(np.count_nonzero(images[train[a:a+2048]]) for a in range(0,len(train),2048))/(len(train)*np.prod(images.shape[1:]));mean=np.float32(255*fraction);std=np.float32(255*np.sqrt(fraction*(1-fraction)))
    assert std>0
    seed=job['seed'];torch.manual_seed(seed);device=torch.device('mps');assert torch.backends.mps.is_available();torch.set_num_threads(3)
    model=ChartCNN(ModelConfig(),ImageConfig(window=20,width=60)).to(device);opt=torch.optim.Adam(model.parameters(),lr=1e-5)
    import wandb
    wb=wandb.init(project='pattern-cnn',entity='arjun-divecha-dancing-elephant',id=hashlib.sha256(str(p).encode()).hexdigest()[:16],resume='allow',name=job['name'],group=OUT.name,dir=str(p),config={**job,'epochs':20,'training_candidates':len(train),'validation':False,'return_basis':'stock five-session next-open return minus equal-weight eligible benchmark','benchmark_policy':cfg['benchmark_policy']},settings=wandb.Settings(disable_git=True,save_code=False))
    atomic_json(p/'wandb.json',{'url':wb.url});history=[];start_epoch=1
    if (p/'last.pt').exists():
        cp=torch.load(p/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);history=cp['history'];start_epoch=cp['epoch']+1
    def inputs(ids):return torch.from_numpy((images[ids].astype('float32')-mean)/std).to(device)
    for epoch in range(start_epoch,21):
        t=time.monotonic();torch.manual_seed(seed+epoch*10000);order=train.copy();np.random.default_rng(seed+epoch).shuffle(order);model.train();loss_sum=0.;correct=0
        for n in range(0,len(order),256):
            ids=order[n:n+256];y=torch.from_numpy((ret[ids]>0).astype('int64')).to(device);logits=model(inputs(ids));loss=torch.nn.functional.cross_entropy(logits,y)
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();loss_sum+=loss.item()*len(ids);correct+=(logits.argmax(1)==y).sum().item()
        row={'epoch':epoch,'train_loss':loss_sum/len(train),'train_accuracy':correct/len(train),'seconds':time.monotonic()-t};history.append(row)
        atomic_save(p/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'history':history,'normalization_mean':float(mean),'normalization_std':float(std),'target':'relative_outperformance'})
        pd.DataFrame(history).to_csv(p/'epochs.csv',index=False);wb.log(row);atomic_json(p/'status.json',{'state':'training','epoch':epoch});print(job['name'],row,flush=True)
    atomic_save(p/'final.pt',model.state_dict());model.eval();scores=[]
    with torch.no_grad():
        for n in range(0,len(test),256):scores.extend(torch.softmax(model(inputs(test[n:n+256])),1)[:,1].cpu().numpy())
    result=sample.iloc[test][['security','date']].copy();result['score']=scores;result['forward_return']=ret[test];result['absolute_forward_return']=sample.iloc[test].absolute_forward_return_5.to_numpy();result['benchmark_return']=sample.iloc[test].benchmark_return.to_numpy()
    result.to_parquet(p/'scores.parquet',index=False)
    if job.get('deployment'):
        metrics={'score_date':job['score_date'],'scored_stocks':len(result),'deployment_only':True}
        result.sort_values('score',ascending=False).to_csv(p/'latest_predictions.csv',index=False)
    else:
        cal=pd.DatetimeIndex(pd.read_parquet(OUT/'candidate_sessions.parquet').date);schedule=pd.DatetimeIndex(result.date.unique()).sort_values()
        metrics,tables=evaluate(result,cal,5,schedule,10);metrics['basis']='Relative to equal-weight universe; observed outcomes; no costs'
        for name,table in tables.items():table.to_parquet(p/f'{name}.parquet',index=True)
        absolute=result.copy();absolute['forward_return']=absolute.absolute_forward_return
        abs_metrics,abs_tables=evaluate(absolute,cal,5,schedule,10)
        atomic_json(p/'absolute_metrics.json',abs_metrics)
        for name,table in abs_tables.items():table.to_parquet(p/f'absolute_{name}.parquet',index=True)
    atomic_json(p/'metrics.json',metrics)
    for k,v in metrics.items():wb.summary[k]=v
    wb.finish();atomic_json(p/'status.json',{'state':'complete','epochs':20,'known_train':len(train)});del model,opt,images;torch.mps.empty_cache()

if __name__=='__main__':
 import fcntl
 lock=(OUT/(sys.argv[1]+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=json.loads((OUT/'config.json').read_text());worker=sys.argv[1]
 for job in cfg['jobs']:
  if job['worker']!=worker:continue
  p=OUT/job['name'];status=p/'status.json'
  if status.exists() and json.loads(status.read_text()).get('state')=='complete':continue
  while not (p/'prepared.json').exists():time.sleep(5)
  atomic_json(status,{'state':'loading','worker':worker})
  try:fit(job,cfg)
  except BaseException as e:atomic_json(status,{'state':'failed','error':repr(e)});raise
