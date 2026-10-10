"""Frozen daily data -> calendar-exact images and independent forward labels."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from pattern.imaging.renderer import render_window


def load_frozen(pointer, start, end, securities=None):
    ref=json.loads(Path(pointer).read_text()); root=Path(ref['path'])
    digest=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    if digest!=ref['manifest_sha256']: raise ValueError('Manifest checksum mismatch')
    filters=[('date','>=',pd.Timestamp(start)),('date','<=',pd.Timestamp(end))]
    if securities is not None: filters.append(('security','in',list(securities)))
    p=pd.read_parquet(root/'prices.parquet',filters=filters)
    for name,cols in [('quality.parquet',['security','date','current_renderer_bar_ok','return_endpoint_ok']),
                      ('calendar_quality.parquet',['security','date','session_index','calendar_segment'])]:
        p=p.merge(pd.read_parquet(root/name,columns=cols,filters=filters),on=['security','date'],validate='one_to_one')
    return p.sort_values(['security','date']),pd.read_parquet(root/'candidate_sessions.parquet').date,ref


def samples(panel,sessions,signal_dates,lookback,horizon,ma_length):
    if min(lookback,horizon,ma_length)<1: raise ValueError('Lengths must be positive')
    history=lookback+ma_length-1
    sessions=pd.DatetimeIndex(sessions); dates=set(pd.to_datetime(signal_dates))
    images=[]; rows=[]
    for security,g in panel.groupby('security',sort=True):
        g=g.set_index('date').sort_index()
        for date in g.index.intersection(pd.DatetimeIndex(sorted(dates))):
            pos=sessions.get_indexer([date])[0]
            if pos<history-1: continue
            past=g.reindex(sessions[pos-history+1:pos+1])
            if not past.current_renderer_bar_ok.fillna(False).all():continue
            if past.calendar_segment.nunique()!=1:continue
            image=render_window(past[['open','high','low','close','volume']].to_numpy(float),
                past.total_return_index_gross_dividends.to_numpy(float),window=lookback,width=3*lookback,ma_length=ma_length)
            if image is None:continue
            target=sessions[pos+horizon] if pos+horizon<len(sessions) else pd.NaT
            future=g.reindex(sessions[pos:pos+horizon+1])
            valid=(len(future)==horizon+1 and future.return_endpoint_ok.fillna(False).all()
                   and future.calendar_segment.nunique()==1)
            ret=float(future.total_return_index_gross_dividends.iloc[-1]/future.total_return_index_gross_dividends.iloc[0]-1) if valid else np.nan
            rows.append(dict(security=security,date=date,input_start=past.index[0],target_date=target,
                             forward_return=ret,label=float(ret>0) if valid else np.nan,outcome_known=valid))
            images.append(image)
    if not images:raise ValueError('No valid input windows')
    return np.stack(images),pd.DataFrame(rows)
