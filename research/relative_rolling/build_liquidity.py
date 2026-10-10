"""Compute PIT trailing20 dollar volume after sorting frozen prices by security/date."""
from pathlib import Path
import pandas as pd,numpy as np,pyarrow.parquet as pq,json
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'runs/full_rolling_clean_2003_2026_seed101';OUT=ROOT/'runs/annual_rolling5_absolute_vs_relative/liquidity';OUT.mkdir(exist_ok=True)
keys=pd.concat([pd.read_parquet(BASE/f'{y}_rolling5_seed101/scores.parquet',columns=['security','date']) for y in range(2003,2027)],ignore_index=True);assert not keys.duplicated().any()
keygroups={k:g for k,g in keys.groupby('security')};source=ROOT/'data/frozen/pattern_daily_v1_20260918/prices.parquet'
px=pq.read_table(source,columns=['security','date','close','volume']).to_pandas(categories=['security']);px=px[px.security.isin(keygroups)].sort_values(['security','date']);parts=[]
for sec,g in px.groupby('security',observed=True,sort=False):
 assert not g.date.duplicated().any()
 value=(g.close*g.volume).where(np.isfinite(g.close)&g.close.gt(0)&np.isfinite(g.volume)&g.volume.ge(0));value=value.where(np.isfinite(value));out=g[['security','date']].copy();out['adv20']=value.rolling(20,min_periods=20).mean();out['valid_days']=value.rolling(20,min_periods=1).count();parts.append(out.merge(keygroups[sec],on=['security','date'],validate='one_to_one'))
result=pd.concat(parts,ignore_index=True);assert len(result)==len(keys) and not result.duplicated(['security','date']).any();result.to_parquet(OUT/'adv20.parquet',index=False)
(OUT/'definition.json').write_text(json.dumps({'source':str(source),'measure':'20 latest observations through formation close, average raw close times volume; 20 valid observations required','threshold_usd':1000000,'entry':'next session open','rows':len(result)},indent=2));print('rows',len(result),'eligible',int(result.adv20.ge(1e6).sum()),flush=True)
