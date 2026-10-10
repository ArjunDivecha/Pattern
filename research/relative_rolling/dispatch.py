"""Own fixed, disjoint Mac/Air queues; retrieve artifacts until complete. No app heartbeat."""
from pathlib import Path
import sys,subprocess,json,time,fcntl,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pattern.research.full_cache import atomic_json
OUT=ROOT/'runs/relative_rolling5_seed101_latest';HOST='air@arjuns-macbook-air.taild3d657.ts.net';REMOTE='/Users/air/pattern-worker';REL='runs/'+OUT.name

def run(args):subprocess.run(args,check=True,timeout=3600)
def ssh(cmd):run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,cmd])
def main():
 lock=(OUT/'dispatcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=json.loads((OUT/'config.json').read_text())
 if '--collect-only' in sys.argv:
  pids=json.loads((OUT/'recovery_20260921.json').read_text())['pids']
  airjobs=[j for j in cfg['jobs'] if j['worker']=='air']
 else:
  airyears={2003,2009,2011,2015,2017,2021};k=0
  for j in cfg['jobs']:
   if not j.get('deployment') and j['year'] in airyears:j['worker']='air'
   else:j['worker']=f'local{k%2}';k+=1
  atomic_json(OUT/'config.json',cfg)
  pids={}
  for worker in ['local0','local1']:
   with (OUT/f'{worker}.log').open('a') as log:
    proc=subprocess.Popen(['caffeinate','-i',sys.executable,'-u','research/relative_rolling/train.py',worker],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   pids[worker]=proc.pid
  atomic_json(OUT/'dispatch_status.json',dict(state='transferring',pids=pids,air_years=sorted(airyears)))
  ssh(f'mkdir -p {REMOTE}/{REL} {REMOTE}/research/relative_rolling')
  run(['scp',str(ROOT/'research/relative_rolling/train.py'),HOST+':'+REMOTE+'/research/relative_rolling/'])
  for name in ['config.json','candidate_sessions.parquet']:run(['scp',str(OUT/name),HOST+':'+REMOTE+'/'+REL+'/'])
  airjobs=[j for j in cfg['jobs'] if j['worker']=='air']
  for i,j in enumerate(airjobs):
   name=j['name'];dest=REMOTE+'/'+REL+'/'+name
   ssh(f'mkdir -p {dest}')
   # Original four packs exist on Air and are immutable. Others transfer once.
   if j['year'] in {2003,2009,2015,2021}:
    ssh(f'ln -sfn {REMOTE}/runs/full_rolling_clean_2003_2026_seed101/{name}/images.npy {dest}/images.npy')
   else:
    run(['rsync','-azL',str(OUT/name/'images.npy'),HOST+':'+dest+'/'])
   run(['scp',str(OUT/name/'sample.parquet'),HOST+':'+dest+'/'])
   # Readiness marker is always transferred last.
   run(['scp',str(OUT/name/'prepared.json'),HOST+':'+dest+'/'])
   if i==0:
    ssh(f'cd {REMOTE} && WANDB_MODE=offline nohup caffeinate -i .venv/bin/python -u research/relative_rolling/train.py air > {REL}/air.log 2>&1 < /dev/null &')
 while True:
  failures=[]
  for j in airjobs:
   try:run(['rsync','-az','--exclude=images.npy','--exclude=sample.parquet','--exclude=prepared.json',HOST+':'+REMOTE+'/'+REL+'/'+j['name']+'/',str(OUT/j['name'])+'/'])
   except Exception as exc:failures.append(str(exc))
  states={j['name']:(json.loads((OUT/j['name']/'status.json').read_text()).get('state') if (OUT/j['name']/'status.json').exists() else 'queued') for j in cfg['jobs']}
  complete=sum(s=='complete' for s in states.values())
  atomic_json(OUT/'dispatch_status.json',dict(state='complete' if complete==25 else 'running',completed=complete,pids=pids,jobs=states,transfer_errors=failures))
  if complete==25:break
  if 'failed' in states.values():raise RuntimeError('Fit failed; inspect individual status')
  time.sleep(60)
if __name__=='__main__':main()
