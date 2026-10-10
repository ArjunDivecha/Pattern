"""Fixed-epoch annual expanding versus rolling study; no validation selection."""
from pathlib import Path
import sys,os,json,time,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from pattern.config import ModelConfig,ImageConfig
from pattern.models.cnn import ChartCNN
from pattern.research.full_cache import atomic_json
from pattern.research.development import sample_partition,cohort_schedule,evaluate
from pattern.research.checkpoint import atomic_save
from research.next_open_lab.run import open_return,BASIS
OUT=ROOT/'runs/annual_expanding_rolling_50k_20260918'
CACHE=ROOT/'cache/frozen_full_I20_M20_v1'
SEEDS=[101,202,303]

def prepare():
    OUT.mkdir(exist_ok=False)
    ref=json.loads((ROOT/'data/frozen/CURRENT.json').read_text());frozen=Path(ref['path'])
    state=json.loads((CACHE/'state.json').read_text())
    assert state['state']=='complete' and state['config']['source_manifest_sha256']==ref['manifest_sha256']
    assert hashlib.sha256((frozen/'manifest.json').read_bytes()).hexdigest()==ref['manifest_sha256']
    cal=pd.DatetimeIndex(pd.read_parquet(frozen/'candidate_sessions.parquet').date)
    index=pd.read_parquet(CACHE/'index.parquet',columns=['security','date','session_position','row_id','forward_return_5'])
    jobs=[];pieces=[]
    for year in range(2013,2023):
        first=cal.searchsorted(f'{year}-01-01');last=min(pd.Timestamp(f'{year}-12-31'),index.date.max())
        schedule=cohort_schedule(cal,cal[first],last,5)
        comparison=sample_partition(index,str(cal[first].date()),str(last.date()),500,71,dates=schedule)
        comparison['partition']='comparison'
        for method in ['expanding','rolling5']:
            start=f'{1997 if method=="expanding" else year-5}-01-01'
            # Image includes 20 bars plus 19 earlier bars needed for MA20.
            cutoff=cal[first-44]
            for seed in SEEDS:
                train=sample_partition(index,start,str(cutoff.date()),500,seed,count=100)
                train['partition']='train';s=pd.concat([train,comparison],ignore_index=True)
                assert train.session_position.max()+5 < comparison.session_position.min()-38
                s['entry_date']=cal[s.session_position.to_numpy()+1];s['exit_date']=cal[s.session_position.to_numpy()+5]
                assert (s.entry_date>s.date).all() and s.exit_date.max()<pd.Timestamp('2023-01-01')
                job=f'{year}_{method}_seed{seed}';s['job']=job;pieces.append(s)
                jobs.append({'name':job,'year':year,'method':method,'seed':seed,'start':start,'train_end':str(cutoff.date()),'evaluation_end':str(last.date())})
    selected=pd.concat(pieces,ignore_index=True);del index,pieces
    keys=pd.concat([selected[['security','entry_date']].rename(columns={'entry_date':'date'}),selected[['security','exit_date']].rename(columns={'exit_date':'date'})]).drop_duplicates()
    print('Loading frozen prices for selected endpoints',flush=True)
    prices=pd.read_parquet(frozen/'prices.parquet',columns=['security','date','open','high','low','close','total_return_index_gross_dividends'],filters=[('date','>=',pd.Timestamp('1997-01-01')),('date','<',pd.Timestamp('2023-01-01'))])
    prices=keys.merge(prices,on=['security','date'],how='left',validate='one_to_one')
    e=prices.rename(columns={'date':'entry_date','open':'entry_open','close':'entry_close','high':'entry_high','low':'entry_low','total_return_index_gross_dividends':'entry_tri'})
    x=prices[['security','date','total_return_index_gross_dividends']].rename(columns={'date':'exit_date','total_return_index_gross_dividends':'exit_tri'})
    selected=selected.merge(e,on=['security','entry_date'],how='left',validate='many_to_one').merge(x,on=['security','exit_date'],how='left',validate='many_to_one')
    r=open_return(selected.entry_open,selected.entry_close,selected.entry_tri,selected.exit_tri)
    good=(selected.entry_low>0)&(selected.entry_low<=selected.entry_open)&(selected.entry_open<=selected.entry_high)&(selected.entry_low<=selected.entry_close)&(selected.entry_close<=selected.entry_high)
    r[~np.isfinite(selected.forward_return_5)|~good.to_numpy()]=np.nan;selected['forward_return_5']=r
    audits=[]
    for job,s in selected.groupby('job',sort=False):
        p=OUT/job;p.mkdir();s=s.drop(columns='job').reset_index(drop=True);s['sample_id']=np.arange(len(s));s.to_parquet(p/'sample.parquet',index=False)
        train=s[s.partition.eq('train')];assert len(train)==50000 and not train.duplicated(['security','date']).any()
        audits.append({'job':job,'training_candidates':len(train),'known_train':int(train.forward_return_5.notna().sum()),'evaluation_rows':int(s.partition.eq('comparison').sum())})
    for year in range(2013,2023):
        frames=[]
        for job in [j for j in jobs if j['year']==year]:
            s=pd.read_parquet(OUT/job['name']/'sample.parquet');frames.append(s[s.partition.eq('comparison')].reset_index(drop=True))
        for f in frames[1:]:pd.testing.assert_frame_equal(frames[0],f)
    pd.DataFrame(audits).to_csv(OUT/'sample_audits.csv',index=False)
    atomic_json(OUT/'config.json',{'epochs':20,'validation':False,'checkpoint':'final epoch only','training_candidates':50000,'seeds':SEEDS,'basis':BASIS,'dataset':ref,'cache_rows':state['rows'],'jobs':jobs,'final_year_partial':True})
    atomic_json(OUT/'preflight.json',{'fits':60,'identical_evaluation_per_year':True,'training_outcomes_before_evaluation_inputs':True,'future_period_2023_untouched':True})

def fit(job,cfg):
    p=OUT/job['name'];sample=pd.read_parquet(p/'sample.parquet');ret=sample.forward_return_5.to_numpy()
    train=np.flatnonzero(sample.partition.eq('train')&np.isfinite(ret));test=np.flatnonzero(sample.partition.eq('comparison'))
    src=np.memmap(CACHE/'images.bin',mode='r',dtype='uint8',shape=(cfg['cache_rows'],1,64,60))
    images=np.empty((len(sample),1,64,60),dtype='uint8')
    for n in range(0,len(sample),1024):images[n:n+1024]=src[sample.row_id.to_numpy()[n:n+1024]]
    fraction=np.count_nonzero(images[train])/images[train].size;mean=np.float32(255*fraction);std=np.float32(255*np.sqrt(fraction*(1-fraction)))
    assert std>0
    seed=job['seed'];torch.manual_seed(seed);device=torch.device('mps');assert torch.backends.mps.is_available();torch.set_num_threads(3)
    model=ChartCNN(ModelConfig(),ImageConfig(window=20,width=60)).to(device);opt=torch.optim.Adam(model.parameters(),lr=1e-5)
    import wandb
    wb=wandb.init(project='pattern-cnn',entity='arjun-divecha-dancing-elephant',id=hashlib.sha256(str(p).encode()).hexdigest()[:16],resume='allow',name=job['name'],group=OUT.name,dir=str(p),config={**job,'epochs':20,'training_candidates':50000,'validation':False,'return_basis':BASIS},settings=wandb.Settings(disable_git=True,save_code=False))
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
        atomic_save(p/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'history':history})
        pd.DataFrame(history).to_csv(p/'epochs.csv',index=False);wb.log(row);atomic_json(p/'status.json',{'state':'training','epoch':epoch});print(job['name'],row,flush=True)
    atomic_save(p/'final.pt',model.state_dict());model.eval();scores=[]
    with torch.no_grad():
        for n in range(0,len(test),256):scores.extend(torch.softmax(model(inputs(test[n:n+256])),1)[:,1].cpu().numpy())
    result=sample.iloc[test][['security','date']].copy();result['score']=scores;result['forward_return']=ret[test]
    ref=cfg['dataset'];cal=pd.DatetimeIndex(pd.read_parquet(Path(ref['path'])/'candidate_sessions.parquet').date);schedule=pd.DatetimeIndex(result.date.unique()).sort_values()
    metrics,tables=evaluate(result,cal,5,schedule,10);metrics['basis']=BASIS
    result.to_parquet(p/'scores.parquet',index=False);atomic_json(p/'metrics.json',metrics)
    for name,table in tables.items():table.to_parquet(p/f'{name}.parquet',index=True)
    for k,v in metrics.items():wb.summary[k]=v
    wb.finish();atomic_json(p/'status.json',{'state':'complete','epochs':20,'known_train':len(train)});del model,opt,images;torch.mps.empty_cache()

def main():
    if len(sys.argv)>1:
        seed=int(sys.argv[1]);cfg=json.loads((OUT/'config.json').read_text())
        for job in [j for j in cfg['jobs'] if j['seed']==seed]:
            try:fit(job,cfg)
            except BaseException as e:atomic_json(OUT/job['name']/'status.json',{'state':'failed','error':repr(e)});raise
        return
    prepare();children=[]
    for seed in SEEDS:
        with (OUT/f'worker_{seed}.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',__file__,str(seed)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        children.append((seed,p))
    atomic_json(OUT/'status.json',{'state':'training','fits':60,'workers':{str(s):p.pid for s,p in children}})
    codes={str(s):p.wait() for s,p in children};cfg=json.loads((OUT/'config.json').read_text());rows=[]
    for job in cfg['jobs']:
        f=OUT/job['name']/'metrics.json'
        if f.exists():rows.append({**job,**json.loads(f.read_text())})
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False);atomic_json(OUT/'status.json',{'state':'complete' if len(rows)==60 and not any(codes.values()) else 'failed','completed':len(rows),'returncodes':codes})

if __name__=='__main__':main()
