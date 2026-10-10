"""Hold sample 101 fixed; vary only training seeds, reusing its completed seed 101."""
from pathlib import Path
import sys,json,os,hashlib,subprocess,fcntl
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
from pattern.research.full_cache import atomic_json
from research.next_open_lab import parallel
OUT=ROOT/'runs/next_open_fixed_sample101_20260918'
SOURCE=ROOT/'runs/next_open_I20_R5_20260918/sample_101'

def main():
    if len(sys.argv)>1:
        parallel.OUT=OUT
        return parallel.worker(int(sys.argv[1]))
    OUT.mkdir(exist_ok=False)
    try:
        original=json.loads((SOURCE/'config.json').read_text())
        assert json.loads((SOURCE/'status.json').read_text())['state']=='complete'
        records={}
        for seed in [202,303]:
            run=OUT/f'sample_{seed}';run.mkdir()
            cfg=dict(original);cfg['seeds']=[seed]
            atomic_json(run/'config.json',cfg)
            for name in ['sample.parquet','images.npy']:os.link(SOURCE/name,run/name)
            assert os.path.samefile(run/'images.npy',SOURCE/'images.npy')
            assert os.path.samefile(run/'sample.parquet',SOURCE/'sample.parquet')
            records[str(seed)]={'sample_seed':101,'training_seed':seed,'identical_sample_and_images':True,'fresh_checkpoint':not (run/'I20_R5_next_open_close'/f'seed_{seed}'/'last.pt').exists()}
        atomic_json(OUT/'preflight.json',{'source':str(SOURCE),'sample_sha256':hashlib.sha256((SOURCE/'sample.parquet').read_bytes()).hexdigest(),'workers':records})
        atomic_json(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'research/next_open_lab/parallel.py',ROOT/'research/development_lab/run.py']})
        children=[]
        for seed in [202,303]:
            with (OUT/f'seed_{seed}.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',__file__,str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((seed,p))
        atomic_json(OUT/'status.json',{'state':'training_parallel','sample_seed':101,'reused_seed':101,'workers':{str(s):p.pid for s,p in children}})
        codes={str(s):p.wait() for s,p in children}
        folders={101:SOURCE/'I20_R5_next_open_close/seed_101',**{s:OUT/f'sample_{s}/I20_R5_next_open_close/seed_{s}' for s,p in children if codes[str(s)]==0}}
        rows=[{'sample_seed':101,'model_seed':s,**json.loads((p/'metrics.json').read_text())} for s,p in folders.items()]
        pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
        atomic_json(OUT/'status.json',{'state':'complete' if all(c==0 for c in codes.values()) else 'failed','returncodes':codes,'fits':len(rows),'sample_seed':101})
    except BaseException as e:
        atomic_json(OUT/'status.json',{'state':'failed','error':repr(e)});raise

if __name__=='__main__':main()
