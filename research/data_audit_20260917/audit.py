"""Read-only 30-case GDM price audit. Outputs evidence beside this script.

Inputs are resolved through GDM. No source tables are modified, no API calls.
Purposeful exception sample, not a random estimate of error prevalence.
"""
from pathlib import Path
import sys, json, hashlib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart')
from gdm import dataset

price_path = dataset('equities.research_vintage_prices', acknowledge='survivorship-incomplete')
master_path = dataset('equities.research_vintage_security_master')
event_root = dataset('equities.events')
manifest_path = dataset('equities.research_vintage_manifest')
sources = [price_path, master_path, manifest_path, event_root/'splits.parquet', event_root/'dividends.parquet']
before = {str(f): (f.stat().st_size, f.stat().st_mtime_ns) for f in sources}
p = pd.read_parquet(price_path).sort_values(['security','date']).reset_index(drop=True)
master = pd.read_parquet(master_path).set_index('security')
events = {}
for typ in ['splits','dividends']:
    e = pd.read_parquet(event_root/f'{typ}.parquet')
    e['event_date'] = pd.to_datetime(e['Ex-Date'], errors='coerce')
    events[typ] = e
g=p.groupby('security',sort=False)
p['previous_date']=g.date.shift()
p['gap_days']=(p.date-p.previous_date).dt.days
p['previous_close']=g.close.shift()
p['previous_tri']=g.total_return_index_gross_dividends.shift()
p['price_return']=p.close/p.previous_close-1
p['total_return']=p.total_return_index_gross_dividends/p.previous_tri-1
p['abs_return']=p.total_return.abs()
p['range_violation']=np.maximum(p[['open','close']].max(axis=1)-p.high,p.low-p[['open','close']].min(axis=1)).clip(lower=0)
p['range_violation_pct']=p.range_violation/p.close.abs().replace(0,np.nan)
cases=[]
def add(category,row,selection):
    cases.append({'case_id':f'C{len(cases)+1:02d}','category':category,'security':row.security,'date':row.date,'selection_rule':selection})
def choose(category,frame,sortby,ascending=False):
    for _,r in frame.sort_values(sortby,ascending=ascending).drop_duplicates('security').head(5).iterrows():
        add(category,r,f'Five distinct securities ordered by {sortby}; ascending={ascending}')
choose('Extreme return',p[p.previous_tri.gt(0)&p.total_return_index_gross_dividends.gt(0)&p.gap_days.le(7)],'abs_return')
choose('Nonpositive price',p[p.close.le(0)|p.total_return_index_gross_dividends.le(0)],'date')
choose('OHLC inconsistency',p[p.close.ge(5)&p.range_violation.gt(0)],'range_violation_pct')
choose('Long gap',p[p.gap_days.gt(180)],'gap_days')
# Positive controls: fixed before looking at the price behavior.
for ticker,date in [('AAPL','2020-08-31'),('TSLA','2020-08-31'),('NVDA','2024-06-10'),('AMZN','2022-06-06'),('GOOGL','2022-07-18')]:
    rows=p[(p.security==ticker+' US Equity')&(p.date==date)]
    if len(rows)!=1: raise ValueError(f'Missing split control {ticker} {date}')
    add('Split control',rows.iloc[0],'Fixed well-known split event; verify against stored event table and issuer source')
for ticker in ['AAPL','MSFT','JNJ','KO','XOM']:
    e=events['dividends'];e=e[(e.security==ticker+' US Equity')&e.event_date.between('2024-01-01','2024-12-31')&(e['Dividend Type']=='Regular Cash')].sort_values('event_date')
    if e.empty: raise ValueError(f'Missing dividend control {ticker}')
    date=e.iloc[-1].event_date
    rows=p[(p.security==ticker+' US Equity')&(p.date==date)]
    if len(rows)!=1: raise ValueError(f'Missing dividend price {ticker}')
    add('Dividend control',rows.iloc[0],'Last stored regular cash ex-date in 2024 for fixed large-cap ticker')
windows=[]; event_rows=[]
for c in cases:
    stock=p[p.security==c['security']]; loc=stock.index[stock.date.eq(c['date'])][0];r=p.loc[loc]
    w=p.loc[max(stock.index.min(),loc-3):min(stock.index.max(),loc+3)].copy();w.insert(0,'case_id',c['case_id']);windows.append(w)
    for field in ['close','previous_close','previous_date','gap_days','volume','price_return','total_return','range_violation_pct']:
        c[field]=r[field]
    if c['security'] in master.index:
        for field in ['name','id_bb_global','id_bb_unique','security_typ','metadata_is_point_in_time','retrieved_at']:
            c[field]=master.loc[c['security'],field]
    n=0
    for typ,e in events.items():
        matches=e[e.security.eq(c['security'])&(e.event_date-c['date']).abs().le(pd.Timedelta(days=7))].copy()
        if not matches.empty:
            matches.insert(0,'case_id',c['case_id']);matches.insert(1,'event_table',typ);event_rows.append(matches);n+=len(matches)
    c['nearby_event_rows']=n
    category=c['category']
    if category=='Nonpositive price':
        c['classification']='Confirmed unusable observation; source cause unresolved'
        c['finding']='Nonpositive close/TRI cannot support an ordinary positive-price stock return. Could be source placeholder or terminal security event; do not infer worthless delisting or replace with zero return.'
    elif category=='OHLC inconsistency':
        c['classification']='Confirmed internal inconsistency; source cause unresolved'
        c['finding']='Open/close lies outside supplied high-low interval. Adjustment mismatch, session definitions, or vendor record error remain alternatives.'
    elif category=='Long gap':
        c['classification']='Unresolved identity/continuity'
        c['finding']='Large calendar gap is verified; latest static FIGI/name does not prove both segments belong to one issuer. Do not bridge until continuity is established.'
    elif category=='Extreme return':
        c['classification']='Unresolved microprice observation'
        c['finding']='Positive-endpoint arithmetic produces the jump. Stored observations alone do not establish an executable trade or erroneous quote; do not clip or call it verified market performance.'
    elif category=='Split control':
        c['classification']='Corporate-action control; adjustment reconciliation required'
        c['finding']='Compare stored split event and issuer announcement with surrounding prices. Smooth ex-date prices can indicate back-adjustment; they are not contemporaneous nominal prices.'
    else:
        c['classification']='Dividend control; adjustment reconciliation required'
        c['finding']='Compare close and TRI returns against stored cash distribution. A cash amount and historical close may use different split-adjustment bases.'
    if category=='Dividend control':
        e=events['dividends'];match=e[e.security.eq(c['security'])&e.event_date.eq(c['date'])&(e['Dividend Type']=='Regular Cash')]
        if len(match)==1:
            amount=float(match.iloc[0]['Dividend Amount']);c['cash_dividend']=amount
            c['cash_adjusted_return']=(r.close+amount)/r.previous_close-1
            c['tri_minus_cash_return_bp']=(r.total_return-c['cash_adjusted_return'])*10000
df=pd.DataFrame(cases)
df.to_csv(ROOT/'cases.csv',index=False)
pd.concat(windows).to_csv(ROOT/'surrounding_prices.csv',index=False)
pd.concat(event_rows,ignore_index=True).to_csv(ROOT/'nearby_events.csv',index=False)
selected=master.loc[master.index.intersection(df.security)].reset_index();selected.to_csv(ROOT/'security_identity.csv',index=False)
after={str(f):(f.stat().st_size,f.stat().st_mtime_ns) for f in sources}
assert before==after,'Source file changed during audit'
assert len(df)==30 and df.case_id.nunique()==30
assert df.groupby('category').size().eq(5).all()
assert all(1<=len(w)<=7 and w.security.nunique()==1 for w in windows)
assert all(w.date.eq(c['date']).sum()==1 for w,c in zip(windows,cases))
meta={'source_files':before,'source_unchanged_stat_check':before==after,'case_count':len(df),'sampling':'Purposeful exception sample plus fixed corporate-action controls; not representative','api_calls':0,'price_rows':len(p),'price_securities':p.security.nunique(),'manifest_generated':json.loads(manifest_path.read_text())['generated']}
(ROOT/'audit_metadata.json').write_text(json.dumps(meta,indent=2,default=str))
print(df[['case_id','category','security','date','close','previous_close','total_return','nearby_event_rows']].to_string(index=False))
print('DIVIDEND_RECONCILIATION',df[df.category.eq('Dividend control')][['security','date','cash_dividend','price_return','total_return','cash_adjusted_return','tri_minus_cash_return_bp']].to_string(index=False))
print('30 cases selected; six groups of five; all windows contain the target and only its security; endpoints may truncate windows; source sizes/mtimes unchanged.')
