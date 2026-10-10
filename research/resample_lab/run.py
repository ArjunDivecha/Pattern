"""Prepare independent fixed training draws and queue unchanged daily/monthly fits."""
from pathlib import Path
import sys, json, os, time, subprocess, argparse, hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from pattern.research.full_cache import atomic_json
from pattern.research.development import sample_partition, verify_splits
from pattern.research.monthly import candidate_windows, sample_months, verify_monthly_splits, render_monthly

BASES={'daily':ROOT/'runs/development_I20_R5_long_20260918','monthly':ROOT/'runs/monthly_M12_R1_v1_20260918'}

def prepare(kind, root):
    base=BASES[kind]; original=pd.read_parquet(base/'sample.parquet')
    basecfg=json.loads((base/'config.json').read_text())
    ref=json.loads((ROOT/basecfg['dataset_pointer']).read_text())
    assert hashlib.sha256((Path(ref['path'])/'manifest.json').read_bytes()).hexdigest()==ref['manifest_sha256']
    fixed=original[original.partition.ne('train')].copy()
    oldimages=np.load(base/'images.npy',mmap_mode='r')
    if kind=='daily':
        cache=ROOT/basecfg['cache'];state=json.loads((cache/'state.json').read_text())
        assert state['state']=='complete' and state['config']['source_manifest_sha256']==ref['manifest_sha256']
        candidates=pd.read_parquet(cache/'index.parquet',columns=['security','date','session_position','row_id','forward_return_5','forward_return_20'])
        src=np.memmap(cache/'images.bin',mode='r',dtype='uint8',shape=(state['rows'],1,64,60))
    else:
        bars=pd.read_parquet(base/'monthly_bars.parquet'); candidates=candidate_windows(bars,12)
        months=pd.DatetimeIndex(pd.read_parquet(base/'month_calendar.parquet').date)
        candidates['target_date']=(candidates.date.dt.to_period('M')+1).map(pd.Series(months,index=months.to_period('M')))
        candidates=candidates[candidates.target_date.notna()]
        groups={k:g.sort_values('month_number').reset_index(drop=True)[['open','high','low','close','volume','ma200']].to_numpy(float) for k,g in bars.groupby('security')}
    sets=[];jobs=[]
    for seed in [101,202,303]:
        out=root/f'{kind}_sample_{seed}';out.mkdir(parents=True,exist_ok=False)
        cfg=dict(basecfg);cfg['sample_seed']=seed;cfg['seeds']=[seed]
        cfg['purpose']='Independent fixed 100000 training draw; unchanged validation and comparison; sampling and initialization both vary'
        cfg.pop('reference_sample',None);cfg.pop('prepared_sample_dir',None)
        c=cfg['train']
        if kind=='daily':
            train=sample_partition(candidates,c['start'],c['end'],c['stocks'],seed,count=c['dates'])
        else:train=sample_months(candidates,c['start'],c['end'],total=100000,seed=seed)
        train['partition']='train'
        sample=pd.concat([train,fixed],ignore_index=True);sample['sample_id']=np.arange(len(sample));sample=sample[original.columns]
        assert len(train)==100000 and not train.duplicated(['security','date']).any()
        pd.testing.assert_frame_equal(sample[sample.partition.ne('train')].drop(columns='sample_id').reset_index(drop=True),fixed.drop(columns='sample_id').reset_index(drop=True))
        if kind=='daily':verify_splits({p:sample[sample.partition.eq(p)] for p in ['train','validation','comparison']},cfg['max_history'],cfg['max_horizon'])
        else:verify_monthly_splits(sample)
        sample.to_parquet(out/'sample.parquet',index=False)
        images=np.lib.format.open_memmap(out/'images.npy',mode='w+',dtype='uint8',shape=(len(sample),*oldimages.shape[1:]))
        if kind=='daily':
            for start in range(0,len(train),1024):images[start:min(start+1024,len(train))]=src[train.row_id.to_numpy()[start:start+1024]]
        else:
            for i,s in enumerate(train.itertuples()):images[i]=render_monthly(groups[s.security][int(s.window_end)-11:int(s.window_end)+1])
        for start in range(0,len(fixed),1024):images[len(train)+start:len(train)+start+1024]=oldimages[fixed.sample_id.to_numpy()[start:start+1024]]
        images.flush()
        assert np.array_equal(images[len(train):],oldimages[fixed.sample_id.to_numpy()])
        sample.assign(year=sample.date.dt.year).groupby(['partition','year']).agg(rows=('security','size'),stocks=('security','nunique'),dates=('date','nunique')).to_csv(out/'coverage.csv')
        keys=set(zip(train.security,train.date));sets.append(keys)
        audit={'training_candidates':len(train),'known_training_labels':int(train[f"forward_return_{5 if kind=='daily' else 1}"].notna().sum()),'training_dates':int(train.date.nunique()),'sample_seed':seed,'model_seed':seed,'evaluation_rows_identical':True,'evaluation_images_identical':True,'split_check_passed':True,'sample_sha256':hashlib.sha256((out/'sample.parquet').read_bytes()).hexdigest()}
        atomic_json(out/'sample_audit.json',audit)
        if kind=='daily':
            meta=json.loads((base/'sample_metadata.json').read_text());meta.update(sample_seed=seed,purpose=cfg['purpose'])
            atomic_json(out/'sample_metadata.json',meta)
            atomic_json(out/'config.json',cfg)
            run=root/f'daily_fit_{seed}';runcfg=dict(cfg,prepared_sample_dir=str(out));config=root/f'daily_fit_{seed}.json';atomic_json(config,runcfg)
            cmd=[sys.executable,str(ROOT/'research/development_lab/run_budget.py'),'--config',str(config),'--run-dir',str(run)]
        else:
            for name in ['month_calendar.parquet','monthly_bars.parquet']:os.link(base/name,out/name)
            meta=json.loads((base/'metadata.json').read_text());meta['sample_seed']=seed;atomic_json(out/'metadata.json',meta)
            atomic_json(out/'config.json',cfg);run=out
            cmd=[sys.executable,str(ROOT/'research/monthly_lab/train.py'),'--run-dir',str(out)]
        jobs.append({'seed':seed,'run_dir':str(run),'command':cmd})
        print('PREPARED',kind,seed,json.dumps(audit),flush=True)
    overlaps=[{'a':[101,202,303][i],'b':[101,202,303][j],'shared_rows':len(sets[i]&sets[j])} for i in range(3) for j in range(i+1,3)]
    assert all(r['shared_rows']<100000 for r in overlaps)
    atomic_json(root/f'{kind}_audit.json',{'jobs':jobs,'overlaps':overlaps,'all_six_checks_per_sample_passed':True})
    return jobs

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=list(BASES),required=True);p.add_argument('--root',required=True);p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args()
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    status=root/f'{a.kind}_queue.json'
    try:
        atomic_json(status,{'state':'preparing','pid':os.getpid()})
        jobs=prepare(a.kind,root)
        atomic_json(status,{'state':'waiting_for_existing_run','wait_pid':a.wait_pid,'jobs':jobs,'pid':os.getpid()})
        while True:
            try:os.kill(a.wait_pid,0)
            except ProcessLookupError:break
            time.sleep(15)
        for job in jobs:
            atomic_json(status,{'state':'training','current_seed':job['seed'],'jobs':jobs,'pid':os.getpid()})
            with (root/f"{a.kind}_{job['seed']}.log").open('a') as log:subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        atomic_json(status,{'state':'complete','jobs':jobs})
    except BaseException as e:
        atomic_json(status,{'state':'failed','error':repr(e)});raise

if __name__=='__main__':main()
