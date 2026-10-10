"""Matched binary-vs-five-class experiment: sample 101, seed 101."""
from pathlib import Path
import sys,json,os,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
import torch
from research.multiclass_lab.train import train_and_score
from research.next_open_lab.run import BASIS
from pattern.research.development import evaluate,cohort_schedule
from pattern.research.full_cache import atomic_json
from pattern.research import tracking

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=101,choices=[101,202,303]);seed=parser.parse_args().seed
    out=ROOT/f'runs/next_open_five_class_sample101_seed{seed}_20260918';out.mkdir(exist_ok=False)
    try:
        source=ROOT/'runs/next_open_I20_R5_20260918/sample_101'
        cfg=json.loads((source/'config.json').read_text());cfg['experiments'][0]['name']='I20_R5_next_open_five_class';cfg['purpose']=f'Five-class A/B; sample 101, model seed {seed}';cfg['seeds']=[seed]
        atomic_json(out/'config.json',cfg)
        for name in ['sample.parquet','images.npy']:os.link(source/name,out/name)
        assert os.path.samefile(source/'sample.parquet',out/'sample.parquet') and os.path.samefile(source/'images.npy',out/'images.npy')
        atomic_json(out/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'research/multiclass_lab/train.py']})
        sample=pd.read_parquet(out/'sample.parquet');images=np.load(out/'images.npy',mmap_mode='r')
        ref=json.loads((ROOT/cfg['dataset_pointer']).read_text());calendar=pd.DatetimeIndex(pd.read_parquet(Path(ref['path'])/'candidate_sessions.parquet').date)
        schedule=cohort_schedule(calendar,cfg['comparison']['start'],cfg['comparison']['end'],5)
        assert torch.backends.mps.is_available();torch.set_num_threads(cfg['threads'])
        atomic_json(out/'status.json',{'state':'training','sample_seed':101,'model_seed':seed,'pid':os.getpid()})
        exp=cfg['experiments'][0];dest=out/exp['name']/f'seed_{seed}'
        scores=train_and_score(cfg,exp,seed,sample,images,20,dest,torch.device('mps'),schedule)
        metrics,tables=evaluate(scores,calendar,5,schedule,cfg['ntiles']);metrics['basis']=BASIS
        tables['cohorts']['entry_date']=calendar[calendar.get_indexer(tables['cohorts'].index)+1]
        atomic_json(dest/'metrics.json',metrics)
        for name,table in tables.items():table.to_parquet(dest/f'{name}.parquet',index=True)
        tracking.finish_metrics(cfg,dest,metrics,tables)
        baseline=source/'I20_R5_next_open_close/seed_101' if seed==101 else ROOT/f'runs/next_open_fixed_sample101_20260918/sample_{seed}/I20_R5_next_open_close/seed_{seed}'
        a=json.loads((baseline/'metrics.json').read_text())
        pd.DataFrame([{'target':'binary',**a},{'target':'five_class',**metrics}]).to_csv(out/'comparison.csv',index=False)
        atomic_json(out/'status.json',{'state':'complete','fits':1,'binary_baseline':str(baseline)})
    except BaseException as e:
        atomic_json(out/'status.json',{'state':'failed','error':repr(e)});raise

if __name__=='__main__':main()
