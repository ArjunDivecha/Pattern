"""Read-only indicative broker snapshot of saved June30 relative top/bottom50.
No orders, instructions, alerts, or gateway restarts. Delayed data explicitly labelled.
"""
from pathlib import Path
import csv,json,datetime,math,asyncio
from ibkr_connect import open_market_data
from ib_async import Stock,util
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/latest_cross_section_execution'
def finite(x):
 try:return float(x) if math.isfinite(float(x)) else None
 except (TypeError,ValueError):return None
async def qualify(ib,row):
 try:
  contracts=await asyncio.wait_for(ib.qualifyContractsAsync(Stock(row['symbol'],'SMART','USD')),8)
  return row,contracts[0] if len(contracts)==1 else None,None
 except Exception as e:return row,None,type(e).__name__+': '+str(e)
rows=list(csv.DictReader((OUT/'basket.csv').open()));results=[];ib=open_market_data('pattern',timeout=10,heal=False);ib.RequestTimeout=10;ib.reqMarketDataType(4)
try:
 for a in range(0,len(rows),10):
  batch=util.run(asyncio.gather(*(qualify(ib,row) for row in rows[a:a+10])),timeout=12);tickers=[]
  for row,c,error in batch:
   if c is None:results.append({**row,'error':error or 'unresolved_contract','executable':False});continue
   tickers.append((row,c,ib.reqMktData(c,'236',snapshot=False)))
  ib.sleep(6)
  for row,c,t in tickers:
   bid,ask=finite(t.bid),finite(t.ask);valid=bid is not None and ask is not None and 0<bid<=ask;mid=(bid+ask)/2 if valid else None
   record={**row,'conid':c.conId,'primary_exchange':c.primaryExchange,'local_symbol':c.localSymbol,'quote_received_utc':str(t.time),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'market_data_type':t.marketDataType,'bid':bid,'ask':ask,'bid_size':finite(t.bidSize),'ask_size':finite(t.askSize),'last':finite(t.last),'shortable_shares':finite(t.shortableShares),'borrow_fee_pct':None,'spread_bps':(ask-bid)/mid*10000 if valid else None,'usable_two_sided':valid,'executable':False,'note':'Delayed/off-hours indicative snapshot, not execution estimate. Quote time is receipt time, not independently verified exchange time. Borrow fee unavailable.'}
   for capital in [1000000,5000000]:
    record[f'shares_per_position_{capital}_per_side']=math.ceil(capital/50/mid) if valid else None
    record[f'adv_participation_pct_{capital}_per_side']=capital/50/float(row['adv20'])*100
   results.append(record);ib.cancelMktData(c)
  (OUT/'ibkr_snapshot.json').write_text(json.dumps(results,indent=2));print('captured',len(results),flush=True)
finally:ib.disconnect()
fields=sorted(set().union(*(r.keys() for r in results)))
with (OUT/'ibkr_snapshot.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
(OUT/'status.json').write_text(json.dumps({'state':'indicative_snapshot_complete','rows':len(results),'signal_date':'2026-06-30','live_execution_cost_estimate':'blocked: no real-time quote entitlement; app connector reauthentication required; borrow fees unavailable','orders_placed':0},indent=2))
