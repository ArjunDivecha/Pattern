"""Versioned non-destructive eligibility and next-open label overlay for cached charts."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pattern.research.full_cache import atomic_json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/research_universe/clean_I20_R5_v1'

def derive(g):
 g=g.sort_values('date').reset_index(drop=True)
 tri=g.total_return_index_gross_dividends
 daily=tri/tri.shift()-1;price=g.close/g.close.shift()-1
 g['historical_jump']=daily.abs().gt(1)|price.abs().gt(1)
 g['input_jump']=g.historical_jump.rolling(39,min_periods=1).max().astype(bool)
 g['below_price_floor']=g.close.lt(5)|~np.isfinite(g.close)
 # Labels require all six dates in the same uninterrupted calendar segment.
 pos=g.session_index
 contiguous=pos.shift(-5).eq(pos+5)&g.calendar_segment.shift(-5).eq(g.calendar_segment)
 good=g.return_endpoint_ok.astype(bool)&g.current_renderer_bar_ok.astype(bool)
 forward_good=good.rolling(6,min_periods=6).sum().shift(-5).eq(6)
 entry=g.open.shift(-1);entryclose=g.close.shift(-1)
 r=(entryclose/entry)*(tri.shift(-5)/tri.shift(-1))-1
 valid=contiguous&forward_good&entry.gt(0)&np.isfinite(r)
 g['next_open_return_5_raw']=r.where(valid)
 g['return_outlier']=g.next_open_return_5_raw.abs().gt(1)
 g['label_valid']=valid&~g.return_outlier
 g['formation_eligible']=~g.below_price_floor&~g.input_jump
 g['training_eligible']=g.formation_eligible&g.label_valid
 g['next_open_return_5_clean']=g.next_open_return_5_raw.where(g.label_valid)
 g['entry_date']=g.date.shift(-1).where(contiguous)
 g['exit_date']=g.date.shift(-5).where(contiguous)
 return g

def main():
 OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/'status.json',{'state':'building'})
 ref=json.loads((ROOT/'data/frozen/CURRENT.json').read_text());src=Path(ref['path'])
 indexes=[]
 for shard in ['frozen_full_I20_M20_v1','frozen_full_I20_M20_extension_v1']:
  i=pd.read_parquet(ROOT/'cache'/shard/'index.parquet',columns=['security','date','row_id']);i['cache_shard']=shard;indexes.append(i)
 idx=pd.concat(indexes,ignore_index=True);groups={k:v for k,v in idx.groupby('security',sort=False)};del idx,indexes
 filters=[('date','>=',pd.Timestamp('1996-01-01'))]
 p=pd.read_parquet(src/'prices.parquet',columns=['security','date','open','close','total_return_index_gross_dividends'],filters=filters)
 for filename,cols in [('quality.parquet',['current_renderer_bar_ok','return_endpoint_ok']),('calendar_quality.parquet',['session_index','calendar_segment'])]:
  q=pd.read_parquet(src/filename,columns=['security','date']+cols,filters=filters);p=p.merge(q,on=['security','date'],validate='one_to_one');del q
 writer=None;counts={};annual=[];extremes=[]
 cols=['security','date','close','below_price_floor','input_jump','formation_eligible','entry_date','exit_date','next_open_return_5_raw','return_outlier','label_valid','training_eligible','next_open_return_5_clean']
 for n,(key,g) in enumerate(p.groupby('security',sort=True)):
  if key not in groups:continue
  d=derive(g);x=groups.pop(key).merge(d[cols],on=['security','date'],validate='one_to_one')
  assert not (x.training_eligible & ~x.formation_eligible).any()
  assert x.loc[x.formation_eligible,'close'].ge(5).all()
  assert x.loc[x.label_valid,'next_open_return_5_clean'].abs().le(1).all()
  assert (x.loc[x.label_valid,'entry_date']>x.loc[x.label_valid,'date']).all()
  for c in ['below_price_floor','input_jump','formation_eligible','return_outlier','label_valid','training_eligible']:counts[c]=counts.get(c,0)+int(x[c].sum())
  counts['images']=counts.get('images',0)+len(x)
  y=x.assign(year=x.date.dt.year).groupby('year').agg(images=('date','size'),formation_eligible=('formation_eligible','sum'),training_eligible=('training_eligible','sum'),outliers=('return_outlier','sum')).reset_index();annual.append(y)
  extremes.append(x[x.return_outlier])
  table=pa.Table.from_pandas(x,preserve_index=False)
  if writer is None:writer=pq.ParquetWriter(OUT/'eligibility.tmp.parquet',table.schema)
  writer.write_table(table)
  if n%500==0:print(n,counts,flush=True)
 writer.close();(OUT/'eligibility.tmp.parquet').replace(OUT/'eligibility.parquet')
 pd.concat(annual).groupby('year').sum().to_csv(OUT/'annual_coverage.csv')
 pd.concat(extremes).to_parquet(OUT/'outlier_audit.parquet',index=False)
 atomic_json(OUT/'manifest.json',{'source':ref,'counts':counts,'price_floor':5,'price_basis':'historically adjusted; not verified nominal price','history':39,'historical_jump_abs_return_gt':1,'horizon':5,'label_abs_return_gt':1,'return_basis':'t+1 close/open multiplied by TRI(t+5)/TRI(t+1), minus1','evaluation_policy':'Rank only formation_eligible; keep raw outcomes and disclose flagged/missing holdings; no post-outcome universe deletion. Clean labels for training only.','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 atomic_json(OUT/'status.json',{'state':'complete',**counts})
 print(counts,flush=True)
if __name__=='__main__':main()
