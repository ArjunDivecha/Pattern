"""Exclusive Air assignments; resume local remainder; retrieve and aggregate results."""
from pathlib import Path
import sys,os,json,time,subprocess,tarfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from pattern.research.full_cache import atomic_json
from research.annual_ab import run as annual
OUT=annual.OUT;HOST='air@arjuns-macbook-air.taild3d657.ts.net';REMOTE='/Users/air/pattern-worker'
NAMES=[f'{y}_{m}_seed303' for y in [2014,2015,2016] for m in ['expanding','rolling5']]

def remaining():
 cfg=json.loads((OUT/'config.json').read_text())
 for job in cfg['jobs']:
  if job['seed']!=303 or job['name'] in NAMES:continue
  p=OUT/job['name']/'status.json'
  if p.exists() and json.loads(p.read_text()).get('state')=='complete':continue
  annual.fit(job,cfg)

def main():
 if len(sys.argv)>1 and sys.argv[1]=='remaining':return remaining()
 cfg=json.loads((OUT/'config.json').read_text());jobs=[j for j in cfg['jobs'] if j['name'] in NAMES]
 for j in jobs:assert not (OUT/j['name']/'last.pt').exists()
 atomic_json(OUT/'air_assignments.json',{'host':HOST,'jobs':NAMES,'state':'preparing','local_seed303_excludes':NAMES})
 with (OUT/'worker_303_resumed.log').open('a') as log:
  local=subprocess.Popen([sys.executable,'-u',__file__,'remaining'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 pack=ROOT/'runs/air_transfer_annual';pack.mkdir(exist_ok=True)
 src=np.memmap(annual.CACHE/'images.bin',mode='r',dtype='uint8',shape=(cfg['cache_rows'],1,64,60))
 for job in jobs:
  p=pack/job['name'];p.mkdir(exist_ok=True);s=pd.read_parquet(OUT/job['name']/'sample.parquet');s.to_parquet(p/'sample.parquet',index=False)
  images=np.lib.format.open_memmap(p/'images.npy',mode='w+',dtype='uint8',shape=(len(s),1,64,60))
  for n in range(0,len(s),1024):images[n:n+1024]=src[s.row_id.to_numpy()[n:n+1024]]
  images.flush()
 frozen=Path(cfg['dataset']['path']);pd.read_parquet(frozen/'candidate_sessions.parquet').to_parquet(pack/'candidate_sessions.parquet',index=False)
 remote_cfg=dict(cfg);remote_cfg['jobs']=jobs;remote_cfg['dataset']=dict(cfg['dataset'],path=REMOTE+'/runs/annual_expanding_rolling_50k_20260918')
 atomic_json(pack/'config.json',remote_cfg)
 # Independent worker changes only storage and tracking; fit math is the same.
 source=(ROOT/'research/annual_ab/run.py').read_text();a=source.index("    src=np.memmap",source.index('def fit'));b=source.index('    fraction=',a)
 source=source[:a]+"    images=np.load(p/'images.npy',mmap_mode='r')\n"+source[b:]
 start=source.index('def main():')
 source=source[:start]+'''def main():
    os.environ['WANDB_MODE']='offline'
    cfg=json.loads((OUT/'config.json').read_text())
    for job in cfg['jobs']:
        status=OUT/job['name']/'status.json'
        if status.exists() and json.loads(status.read_text()).get('state')=='complete':continue
        try:fit(job,cfg)
        except BaseException as e:atomic_json(status,{'state':'failed','error':repr(e)});raise
    atomic_json(OUT/'air_status.json',{'state':'complete'})
if __name__=='__main__':main()
'''
 (pack/'air_run.py').write_text(source)
 with tarfile.open(pack/'code.tar.gz','w:gz') as t:
  for folder in ['pattern','research']:
   for p in (ROOT/folder).rglob('*.py'):t.add(p,arcname=str(p.relative_to(ROOT)))
 subprocess.run(['ssh',HOST,f'mkdir -p {REMOTE}/runs/annual_expanding_rolling_50k_20260918'],check=True)
 subprocess.run(['scp',str(pack/'code.tar.gz'),HOST+':'+REMOTE+'/'],check=True)
 subprocess.run(['ssh',HOST,f'cd {REMOTE} && tar xzf code.tar.gz'],check=True)
 subprocess.run(['scp',str(pack/'air_run.py'),HOST+':'+REMOTE+'/research/annual_ab/air_run.py'],check=True)
 subprocess.run(['scp',str(pack/'config.json'),str(pack/'candidate_sessions.parquet'),HOST+':'+REMOTE+'/runs/annual_expanding_rolling_50k_20260918/'],check=True)
 for job in jobs:
  subprocess.run(['rsync','-az',str(pack/job['name'])+'/',HOST+':'+REMOTE+'/runs/annual_expanding_rolling_50k_20260918/'+job['name']+'/'],check=True)
 command=f'cd {REMOTE} && nohup caffeinate -i .venv/bin/python -u research/annual_ab/air_run.py > air_training.log 2>&1 < /dev/null &'
 subprocess.run(['ssh',HOST,command],check=True)
 atomic_json(OUT/'air_assignments.json',{'host':HOST,'jobs':NAMES,'state':'training','local_remainder_pid':local.pid,'tracking':'offline W&B on Air; local logs retrieved'})
 while True:
  for job in jobs:
   subprocess.run(['rsync','-az','--exclude=images.npy','--exclude=sample.parquet',HOST+':'+REMOTE+'/runs/annual_expanding_rolling_50k_20260918/'+job['name']+'/',str(OUT/job['name'])+'/'],check=True)
  states=[json.loads((OUT/j['name']/'status.json').read_text()).get('state') if (OUT/j['name']/'status.json').exists() else 'pending' for j in jobs]
  if 'failed' in states:raise RuntimeError('Air fit failed')
  if all(s=='complete' for s in states):break
  time.sleep(60)
 atomic_json(OUT/'air_assignments.json',{'host':HOST,'jobs':NAMES,'state':'complete'})
 while True:
  rows=[]
  for j in cfg['jobs']:
   p=OUT/j['name'];status=p/'status.json'
   if status.exists() and json.loads(status.read_text()).get('state')=='failed':raise RuntimeError('Local fit failed')
   if (p/'metrics.json').exists():rows.append({**j,**json.loads((p/'metrics.json').read_text())})
  pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
  if len(rows)==60:break
  time.sleep(60)
 atomic_json(OUT/'status.json',{'state':'complete','completed':60,'air_fits':6,'main_mac_fits':54})

if __name__=='__main__':main()
