"""Three fresh daily fits: signal t close, enter t+1 open, exit t+5 close."""
from pathlib import Path
import sys,json,os,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from pattern.research.full_cache import atomic_json
from pattern.research.development import evaluate,verify_splits,cohort_schedule
from research.development_lab.run import train_and_score
from pattern.research import tracking
BASIS='Gross next-session open to fifth-session close; entry-day close/open times subsequent close-to-close TRI; no costs; $1 long/$1 short. Observed outcomes only; adjusted-price research proxy.'

def open_return(entry_open,entry_close,entry_tri,exit_tri):
    a=np.asarray([entry_open,entry_close,entry_tri,exit_tri],dtype=float)
    valid=np.isfinite(a).all(axis=0)&(a>0).all(axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):r=(a[1]/a[0])*(a[3]/a[2])-1
    return np.where(valid & np.isfinite(r),r,np.nan)

def main():
    out=ROOT/'runs/next_open_I20_R5_20260918';out.mkdir(exist_ok=False)
    atomic_json(out/'status.json',{'state':'preparing','pid':os.getpid(),'basis':BASIS})
    try:
        ref=json.loads((ROOT/'data/frozen/CURRENT.json').read_text());frozen=Path(ref['path'])
        assert hashlib.sha256((frozen/'manifest.json').read_bytes()).hexdigest()==ref['manifest_sha256']
        calendar=pd.DatetimeIndex(pd.read_parquet(frozen/'candidate_sessions.parquet').date)
        sources=ROOT/'runs/resample_stability_20260918'
        samples={seed:pd.read_parquet(sources/f'daily_sample_{seed}/sample.parquet') for seed in [101,202,303]}
        for sample in samples.values():
            pos=calendar.get_indexer(sample.date);assert (pos>=0).all()
            sample['entry_date']=calendar[pos+1];sample['exit_date']=calendar[pos+5]
            assert (sample.entry_date>sample.date).all() and (sample.exit_date>=sample.entry_date).all()
        print('Loading frozen open/close/TRI for exact-date joins',flush=True)
        prices=pd.read_parquet(frozen/'prices.parquet',columns=['security','date','open','high','low','close','total_return_index_gross_dividends'],filters=[('date','>=',pd.Timestamp('1997-01-01')),('date','<',pd.Timestamp('2023-01-01'))])
        keys=pd.concat([s[['security','entry_date']].rename(columns={'entry_date':'date'}) for s in samples.values()]+[s[['security','exit_date']].rename(columns={'exit_date':'date'}) for s in samples.values()]).drop_duplicates()
        prices=keys.merge(prices,on=['security','date'],how='left',validate='one_to_one')
        entries=prices.rename(columns={'date':'entry_date','open':'entry_open','close':'entry_close','high':'entry_high','low':'entry_low','total_return_index_gross_dividends':'entry_tri'})
        exits=prices[['security','date','total_return_index_gross_dividends']].rename(columns={'date':'exit_date','total_return_index_gross_dividends':'exit_tri'})
        del prices,keys
        for seed,sample in samples.items():
            source=sources/f'daily_sample_{seed}';run=out/f'sample_{seed}';run.mkdir()
            cfg=json.loads((source/'config.json').read_text());cfg['sample_seed']=seed;cfg['seeds']=[seed];cfg['threads']=4
            cfg['purpose']='Corrected timing: next open to fifth-session close; fresh training'
            cfg['return_basis']=BASIS;cfg['experiments'][0]['name']='I20_R5_next_open_close'
            cfg.pop('prepared_sample_dir',None);cfg.pop('reference_sample',None)
            joined=sample.merge(entries,on=['security','entry_date'],how='left',validate='many_to_one').merge(exits,on=['security','exit_date'],how='left',validate='many_to_one')
            old=joined.forward_return_5.to_numpy().copy()
            returns=open_return(joined.entry_open,joined.entry_close,joined.entry_tri,joined.exit_tri)
            goodbar=(joined.entry_low>0)&(joined.entry_low<=joined.entry_open)&(joined.entry_open<=joined.entry_high)&(joined.entry_low<=joined.entry_close)&(joined.entry_close<=joined.entry_high)
            returns[~np.isfinite(old)|~goodbar.to_numpy()]=np.nan
            joined['forward_return_5']=returns
            joined=joined.drop(columns=['forward_return_20'],errors='ignore')
            verify_splits({p:joined[joined.partition.eq(p)] for p in ['train','validation','comparison']},cfg['max_history'],cfg['max_horizon'])
            joined.to_parquet(run/'sample.parquet',index=False)
            os.link(source/'images.npy',run/'images.npy');atomic_json(run/'config.json',cfg)
            known=np.isfinite(old)&np.isfinite(returns)
            atomic_json(run/'label_audit.json',{'sample_seed':seed,'training_candidates':int(joined.partition.eq('train').sum()),'known_training_labels':int((joined.partition.eq('train')&np.isfinite(returns)).sum()),'sign_flips_all_partitions':int(((old>0)!=(returns>0))[known].sum()),'known_pairs':int(known.sum()),'basis':BASIS,'entry_strictly_after_input':True,'same_images':True,'no_checkpoint_reuse':True})
            print('PREPARED',seed,(run/'label_audit.json').read_text(),flush=True)
        for seed in [202,303]:
            a=pd.read_parquet(out/'sample_101/sample.parquet');b=pd.read_parquet(out/f'sample_{seed}/sample.parquet')
            pd.testing.assert_frame_equal(a[a.partition.ne('train')].reset_index(drop=True),b[b.partition.ne('train')].reset_index(drop=True))
        atomic_json(out/'preflight.json',{'three_samples':True,'identical_evaluation_labels_and_rows':True,'basis':BASIS})
        del samples,entries,exits,joined,a,b
        assert torch.backends.mps.is_available(),'MPS unavailable'
        torch.set_num_threads(4);results=[]
        for seed in [101,202,303]:
            run=out/f'sample_{seed}';cfg=json.loads((run/'config.json').read_text());exp=cfg['experiments'][0]
            sample=pd.read_parquet(run/'sample.parquet');images=np.load(run/'images.npy',mmap_mode='r')
            schedule=cohort_schedule(calendar,cfg['comparison']['start'],cfg['comparison']['end'],5)
            atomic_json(out/'status.json',{'state':'training','seed':seed,'pid':os.getpid(),'basis':BASIS})
            dest=run/exp['name']/f'seed_{seed}'
            scores=train_and_score(cfg,exp,seed,sample,images,20,dest,torch.device('mps'),schedule)
            metrics,tables=evaluate(scores,calendar,5,schedule,cfg['ntiles']);metrics['basis']=BASIS
            tables['cohorts']['entry_date']=calendar[calendar.get_indexer(tables['cohorts'].index)+1]
            atomic_json(dest/'metrics.json',metrics)
            for name,table in tables.items():table.to_parquet(dest/f'{name}.parquet',index=True)
            tracking.finish_metrics(cfg,dest,metrics,tables)
            results.append({'seed':seed,**metrics});pd.DataFrame(results).to_csv(out/'comparison.csv',index=False)
            atomic_json(run/'status.json',{'state':'complete'})
        atomic_json(out/'status.json',{'state':'complete','fits':3,'basis':BASIS})
    except BaseException as e:
        atomic_json(out/'status.json',{'state':'failed','error':repr(e)});raise

if __name__=='__main__':main()
