"""Independent locked workers and an atomic aggregate for corrected daily fits."""
from pathlib import Path
import sys,os,json,fcntl,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from research.next_open_lab.run import BASIS
from research.development_lab.run import train_and_score
from pattern.research.development import evaluate,cohort_schedule
from pattern.research.full_cache import atomic_json
from pattern.research import tracking
OUT=ROOT/'runs/next_open_I20_R5_20260918'

def worker(seed):
    run=OUT/f'sample_{seed}'
    with (run/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            cfg=json.loads((run/'config.json').read_text());exp=cfg['experiments'][0]
            assert torch.backends.mps.is_available()
            torch.set_num_threads(cfg['threads'])
            ref=json.loads((ROOT/cfg['dataset_pointer']).read_text())
            calendar=pd.DatetimeIndex(pd.read_parquet(Path(ref['path'])/'candidate_sessions.parquet').date)
            sample=pd.read_parquet(run/'sample.parquet');images=np.load(run/'images.npy',mmap_mode='r')
            schedule=cohort_schedule(calendar,cfg['comparison']['start'],cfg['comparison']['end'],5)
            atomic_json(run/'status.json',{'state':'training','seed':seed,'pid':os.getpid()})
            dest=run/exp['name']/f'seed_{seed}'
            scores=train_and_score(cfg,exp,seed,sample,images,20,dest,torch.device('mps'),schedule)
            metrics,tables=evaluate(scores,calendar,5,schedule,cfg['ntiles']);metrics['basis']=BASIS
            tables['cohorts']['entry_date']=calendar[calendar.get_indexer(tables['cohorts'].index)+1]
            atomic_json(dest/'metrics.json',metrics)
            for name,table in tables.items():table.to_parquet(dest/f'{name}.parquet',index=True)
            tracking.finish_metrics(cfg,dest,metrics,tables)
            atomic_json(run/'status.json',{'state':'complete','seed':seed})
        except BaseException as e:
            atomic_json(run/'status.json',{'state':'failed','seed':seed,'error':repr(e)});raise

def main():
    if len(sys.argv)>1:return worker(int(sys.argv[1]))
    with (OUT/'parallel.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children=[]
        for seed in [101,202,303]:
            with (OUT/f'seed_{seed}.log').open('a') as log:
                p=subprocess.Popen([sys.executable,'-u',__file__,str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((seed,p))
        atomic_json(OUT/'status.json',{'state':'training_parallel','workers':{str(s):p.pid for s,p in children},'basis':BASIS})
        codes={str(s):p.wait() for s,p in children}
        rows=[]
        for seed,_ in children:
            f=OUT/f'sample_{seed}/I20_R5_next_open_close/seed_{seed}/metrics.json'
            if codes[str(seed)]==0 and f.exists():rows.append({'seed':seed,**json.loads(f.read_text())})
        pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
        atomic_json(OUT/'status.json',{'state':'complete' if all(c==0 for c in codes.values()) else 'failed','returncodes':codes,'fits':len(rows),'basis':BASIS})

if __name__=='__main__':main()
