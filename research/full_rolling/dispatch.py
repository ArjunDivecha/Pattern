from pathlib import Path
import sys,os,json,time,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pattern.research.full_cache import atomic_json
OUT=ROOT/'runs/full_rolling_clean_2003_2026_seed101';HOST='air@arjuns-macbook-air.taild3d657.ts.net';REMOTE='/Users/air/pattern-worker';REL='runs/'+OUT.name
while not (OUT/'config.json').exists():time.sleep(5)
cfg=json.loads((OUT/'config.json').read_text());pids={}
for worker in ['local0','local1']:
 with (OUT/f'{worker}.log').open('a') as f:p=subprocess.Popen(['caffeinate','-i',sys.executable,'-u','research/full_rolling/train.py',worker],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 pids[worker]=p.pid
atomic_json(OUT/'dispatch_status.json',{'state':'transferring','local_pids':pids})
def run(args):subprocess.run(args,check=True)
run(['ssh',HOST,f'mkdir -p {REMOTE}/{REL} {REMOTE}/research/full_rolling'])
run(['scp',str(ROOT/'research/full_rolling/train.py'),HOST+':'+REMOTE+'/research/full_rolling/train.py'])
for file in ['config.json','candidate_sessions.parquet']:run(['scp',str(OUT/file),HOST+':'+REMOTE+'/'+REL+'/'])
jobs=[j for j in cfg['jobs'] if j['worker']=='air']
for n,j in enumerate(jobs):
 run(['rsync','-az',str(OUT/j['name'])+'/',HOST+':'+REMOTE+'/'+REL+'/'+j['name']+'/'])
 if n==0:
  run(['ssh',HOST,f'cd {REMOTE} && WANDB_MODE=offline nohup caffeinate -i .venv/bin/python -u research/full_rolling/train.py air > {REL}/air.log 2>&1 < /dev/null &'])
 atomic_json(OUT/'dispatch_status.json',{'state':'transferring','transferred_air_jobs':n+1,'local_pids':pids})
atomic_json(OUT/'dispatch_status.json',{'state':'running','local_pids':pids,'air_jobs':[j['name'] for j in jobs]})
while True:
 for j in jobs:run(['rsync','-az','--exclude=images.npy','--exclude=sample.parquet',HOST+':'+REMOTE+'/'+REL+'/'+j['name']+'/',str(OUT/j['name'])+'/'])
 states=[]
 for j in cfg['jobs']:
  p=OUT/j['name']/'status.json';states.append(json.loads(p.read_text()).get('state') if p.exists() else 'queued')
 atomic_json(OUT/'status.json',{'state':'complete' if states.count('complete')==24 else 'running','completed':states.count('complete'),'failed':states.count('failed')})
 if states.count('complete')==24:break
 if 'failed' in states:raise RuntimeError('Training failure; inspect per-job status')
 time.sleep(60)
