"""Resumable read-only Gateway daily spread/fee archive collector.
Four workers, globally at most one new request per 0.5 seconds. No orders.
Small-bar rules are not used to accelerate small bars: this only requests 1-day bars.
"""
from pathlib import Path
import argparse,asyncio,csv,datetime,fcntl,hashlib,json,math,time
from ib_async import Contract,Stock,util
from ibkr_connect import open_market_data
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/cost_aware_ab_2020_2026_1m'
CACHE=OUT/'broker_history'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)
class Pace:
 def __init__(self):self.lock=asyncio.Lock();self.next=0
 async def wait(self):
  async with self.lock:
   await asyncio.sleep(max(0,self.next-time.monotonic()));self.next=time.monotonic()+0.5
 async def cooldown(self):
  async with self.lock:self.next=max(self.next,time.monotonic()+600)
def valid_contracts(cs):
 return len(cs)==1 and cs[0] is not None and bool(getattr(cs[0],'conId',0))
async def main(limit,ib):
 CACHE.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader((OUT/'broker_requests.csv').open()))
 known={r['security']:r for r in json.loads((ROOT/'runs/latest_cross_section_execution/ibkr_snapshot.json').read_text()) if r.get('conid')}
 # Known identities first to establish throughput; no coverage-based portfolio exclusion.
 rows.sort(key=lambda r:(r['security'] not in known,r['security']))
 if limit:rows=rows[:limit]
 pace=Pace();ib.RequestTimeout=35
 state={'state':'collecting','started_utc':stamp(),'requested_securities':len(rows),'completed_this_pass':0,'cache_reused':0,'unresolved_cached':0,'resolved':0,'unresolved':0,'empty_or_failed_series':0,'transport':'TWS Gateway paper market-data, read-only','bar_size':'1 day','workers':4,'new_request_interval_seconds':0.5,'fee_units':'unverified','historical_availability':'not sourced'}
 errors=json.loads((OUT/'collection_errors.json').read_text()) if (OUT/'collection_errors.json').exists() else []
 def on_error(req,code,msg,contract):
  errors.append({'utc':stamp(),'request_id':req,'code':code,'message':msg,'symbol':getattr(contract,'symbol',None)})
  if 'pacing' in msg.lower():asyncio.create_task(pace.cooldown())
 ib.errorEvent+=on_error
 queue=asyncio.Queue()
 for row in rows:queue.put_nowait(row)
 def save():
  state['updated_utc']=stamp();state['remaining_this_pass']=queue.qsize();state['error_count']=len(errors)
  atomic(OUT/'collection_status.json',state);atomic(OUT/'collection_errors.json',errors)
 async def worker():
  while not queue.empty():
   row=await queue.get();key=hashlib.sha256(row['security'].encode()).hexdigest()[:20];path=CACHE/(key+'.json')
   if path.exists():
    old=json.loads(path.read_text())
    if old.get('state') in ['complete','unresolved_contract']:
     state['cache_reused' if old['state']=='complete' else 'unresolved_cached']+=1;queue.task_done();save();continue
   record={'security':row['security'],'symbol':row['symbol'],'requested_first_formation':row['first_formation'],'requested_last_exit':row['last_exit'],'retrieved_utc':stamp(),'state':'pending','series':{}}
   try:
    if row['security'] in known:
     k=known[row['security']];c=Contract(conId=k['conid'],symbol=k['symbol'],secType='STK',exchange='SMART',currency='USD')
     record['identity_source']='previous uniquely qualified snapshot; historical identity validation still required'
    else:
     await pace.wait();cs=await asyncio.wait_for(ib.qualifyContractsAsync(Stock(row['symbol'],'SMART','USD')),20)
     if not valid_contracts(cs):
      record['state']='unresolved_contract';state['unresolved']+=1;continue
     c=cs[0];record['identity_source']='current unique symbol qualification; historical identity validation still required'
    record['conid']=c.conId;record['primary_exchange']=c.primaryExchange;state['resolved']+=1
    start=datetime.datetime.fromisoformat(row['first_formation']);end=datetime.datetime.fromisoformat(row['last_exit'])+datetime.timedelta(days=1)
    duration=f'{max(1,math.ceil(((end-start).days+35)/365))} Y';endstr=end.strftime('%Y%m%d 23:59:59 UTC')
    for typ in ['BID_ASK','FEE_RATE']:
     data=[];attempts=[]
     for attempt in range(2):
      if attempt:await asyncio.sleep(20)
      await pace.wait();before=len(errors)
      bars=await ib.reqHistoricalDataAsync(c,endstr,duration,'1 day',typ,True,formatDate=1,timeout=30)
      errs=[e for e in errors[before:] if e.get('symbol')==c.symbol]
      attempts.append({'utc':stamp(),'returned':len(bars),'errors':errs})
      if bars:
       data=[util.dataclassAsDict(b) for b in bars];break
      if any('query returned no data' in e['message'].lower() for e in errs):break
     record['series'][typ]={'status':'returned' if data else 'missing_or_timeout','bars':data,'attempts':attempts}
     if not data:state['empty_or_failed_series']+=1
     atomic(path,record)
    record['state']='complete' if all(v['status']=='returned' for v in record['series'].values()) else 'incomplete'
   except Exception as e:record['state']='error';record['error']=repr(e)
   finally:
    record['finished_utc']=stamp();atomic(path,record);state['completed_this_pass']+=1;queue.task_done();save()
   if not ib.isConnected():raise RuntimeError('Gateway disconnected; collection stopped; cache retained')
 try:
  save();await asyncio.gather(*(worker() for _ in range(4)));state['state']='collection_pass_finished'
 except Exception as e:state['state']='stopped_error';state['error']=repr(e);raise
 finally:ib.disconnect();save()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);a=parser.parse_args()
 lock=(OUT/'collector.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Central connection helper uses the synchronous ib_async bridge; nest via util.run.
 ib=open_market_data('pattern',timeout=10,heal=False)
 util.run(main(a.limit,ib))
