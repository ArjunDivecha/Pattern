"""Second exclusive Air assignment, with checkpoint-resumed local remainders."""
from pathlib import Path
import sys,json,subprocess,time,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from pattern.research.full_cache import atomic_json
from research.annual_ab import run as annual
OUT=annual.OUT;HOST='air@arjuns-macbook-air.taild3d657.ts.net';REMOTE='/Users/air/pattern-worker'
NAMES=['2022_rolling5_seed101','2022_rolling5_seed202']
def main():
 cfg=json.loads((OUT/'config.json').read_text())
 if len(sys.argv)>1:
  seed=int(sys.argv[1])
  for job in cfg['jobs']:
   if job['seed']!=seed or job['name'] in NAMES:continue
   status=OUT/job['name']/'status.json'
   if status.exists() and json.loads(status.read_text()).get('state')=='complete':continue
   try:annual.fit(job,cfg)
   except BaseException as e:atomic_json(status,{'state':'failed','error':repr(e)});raise
  return
 jobs=[j for j in cfg['jobs'] if j['name'] in NAMES]
 for j in jobs:assert not (OUT/j['name']/'last.pt').exists()
 atomic_json(OUT/'air_additional_assignments.json',{'jobs':NAMES,'state':'preparing','old_air_worker_pid':69243})
 workers={}
 for seed in [101,202]:
  with (OUT/f'worker_{seed}_resumed.log').open('a') as log:p=subprocess.Popen([sys.executable,'-u',__file__,str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  workers[str(seed)]=p.pid
 pack=ROOT/'runs/air_transfer_additional';pack.mkdir(exist_ok=True)
 src=np.memmap(annual.CACHE/'images.bin',mode='r',dtype='uint8',shape=(cfg['cache_rows'],1,64,60))
 for job in jobs:
  p=pack/job['name'];p.mkdir(exist_ok=True);s=pd.read_parquet(OUT/job['name']/'sample.parquet');s.to_parquet(p/'sample.parquet',index=False)
  images=np.lib.format.open_memmap(p/'images.npy',mode='w+',dtype='uint8',shape=(len(s),1,64,60))
  for n in range(0,len(s),1024):images[n:n+1024]=src[s.row_id.to_numpy()[n:n+1024]]
  images.flush()
  subprocess.run(['rsync','-az',str(p)+'/',HOST+':'+REMOTE+'/runs/annual_expanding_rolling_50k_20260918/'+job['name']+'/'],check=True)
 remote_cfg=dict(cfg);remote_cfg['jobs']=jobs;remote_cfg['dataset']=dict(cfg['dataset'],path=REMOTE+'/runs/annual_expanding_rolling_50k_20260918')
 atomic_json(pack/'additional_config.json',remote_cfg)
 script='''from pathlib import Path
import sys,json,os,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.annual_ab.air_run import fit,OUT,atomic_json
os.environ['WANDB_MODE']='offline'
while True:
    try:os.kill(69243,0)
    except ProcessLookupError:break
    time.sleep(10)
cfg=json.loads((OUT/'additional_config.json').read_text())
for job in cfg['jobs']:
    status=OUT/job['name']/'status.json'
    if status.exists() and json.loads(status.read_text()).get('state')=='complete':continue
    try:fit(job,cfg)
    except BaseException as e:atomic_json(status,{'state':'failed','error':repr(e)});raise
atomic_json(OUT/'additional_air_status.json',{'state':'complete'})
'''
 (pack/'additional_run.py').write_text(script)
 subprocess.run(['scp',str(pack/'additional_config.json'),HOST+':'+REMOTE+'/runs/annual_expanding_rolling_50k_20260918/'],check=True)
 subprocess.run(['scp',str(pack/'additional_run.py'),HOST+':'+REMOTE+'/research/annual_ab/'],check=True)
 subprocess.run(['ssh',HOST,f'cd {REMOTE} && nohup caffeinate -i .venv/bin/python -u research/annual_ab/additional_run.py > additional_air.log 2>&1 < /dev/null &'],check=True)
 atomic_json(OUT/'air_additional_assignments.json',{'jobs':NAMES,'state':'queued_after_original_air_worker','local_workers':workers,'host':HOST})
 while True:
  for j in jobs:subprocess.run(['rsync','-az','--exclude=images.npy','--exclude=sample.parquet',HOST+':'+REMOTE+'/runs/annual_expanding_rolling_50k_20260918/'+j['name']+'/',str(OUT/j['name'])+'/'],check=True)
  states=[json.loads((OUT/j['name']/'status.json').read_text()).get('state') if (OUT/j['name']/'status.json').exists() else 'queued' for j in jobs]
  if 'failed' in states:raise RuntimeError('Additional Air fit failed')
  if all(s=='complete' for s in states):break
  time.sleep(60)
 atomic_json(OUT/'air_additional_assignments.json',{'jobs':NAMES,'state':'complete','local_workers':workers,'host':HOST})
if __name__=='__main__':main()
