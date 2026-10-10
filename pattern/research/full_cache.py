"""Resumable full-universe cache. One image stream shared by both target horizons."""
from pathlib import Path
import json,hashlib,os,time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pattern.imaging.fast import load_renderer
from pattern.imaging.renderer import render_window


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str));tmp.replace(path)


def stock_index(g,calendar,history,horizons,first_pos,last_pos):
    g=g.sort_values('date').reset_index(drop=True);pos=g.session_index.to_numpy(dtype='int64',na_value=-1)
    segment=g.calendar_segment.to_numpy();good=g.current_renderer_bar_ok.to_numpy(bool)
    cs=np.r_[0,np.cumsum(~good)];ends=np.arange(history-1,len(g));starts=ends-history+1
    valid=(pos[ends]>=first_pos)&(pos[ends]<=last_pos)&(pos[starts]>=0)&(pos[ends]-pos[starts]==history-1)&(segment[ends]==segment[starts])&(cs[ends+1]-cs[starts]==0)
    ends=ends[valid]
    index=pd.DataFrame({'security':g.security.iloc[ends].to_numpy(),'date':g.date.iloc[ends].to_numpy(),'session_position':pos[ends].astype('int32')})
    tri=g.total_return_index_gross_dividends.to_numpy(float);endpoint=g.return_endpoint_ok.to_numpy(bool);bad=np.r_[0,np.cumsum(~endpoint)]
    for h in horizons:
        stop=ends+h;inside=stop<len(g);ok=np.zeros(len(ends),bool);v=np.flatnonzero(inside);e=ends[v];t=stop[v]
        ok[v]=(pos[t]-pos[e]==h)&(segment[t]==segment[e])&(bad[t+1]-bad[e]==0)
        ret=np.full(len(ends),np.nan);ret[ok]=tri[stop[ok]]/tri[ends[ok]]-1
        ret[~np.isfinite(ret)]=np.nan
        index[f'forward_return_{h}']=ret
    return g,ends,index


def build_cache(root,cache,calendar,first_pos,last_pos,window=20,ma=20,horizons=(5,20)):
    root=Path(root);cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);(cache/'parts').mkdir(exist_ok=True)
    cfg={'source_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'window':window,'ma':ma,'horizons':list(horizons),'first_pos':int(first_pos),'last_pos':int(last_pos),
         'source_code_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,str(Path(__file__).parents[1]/'imaging/fast_renderer.c')]}}
    state_path=cache/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'config':cfg,'last_security':None,'rows':0,'stocks_processed':0,'state':'building'}
    if state['config']!=cfg:raise ValueError('Cache configuration changed; use a new cache directory')
    if state['state']=='complete':
        if (cache/'images.bin').stat().st_size!=state['rows']*64*3*window:raise ValueError('Cache byte count mismatch')
        return state
    # Only development-period source rows; reserved prices not loaded.
    lower=pd.Timestamp('1996-01-01');upper=calendar[last_pos+max(horizons)]
    filters=[('date','>=',lower),('date','<=',upper)]
    columns=['security','date','open','high','low','close','volume','total_return_index_gross_dividends']
    print('Loading full development panel',lower,upper,flush=True)
    p=pd.read_parquet(root/'prices.parquet',columns=columns,filters=filters)
    for file,cols in [('quality.parquet',['security','date','current_renderer_bar_ok','return_endpoint_ok']),('calendar_quality.parquet',['security','date','session_index','calendar_segment'])]:
        q=pd.read_parquet(root/file,columns=cols,filters=filters);p=p.merge(q,on=['security','date'],validate='one_to_one');del q
    p=p.sort_values(['security','date']).reset_index(drop=True)
    render=load_renderer(cache/'compiled');image_file=cache/'images.bin';history=window+ma-1
    if not image_file.exists():image_file.touch()
    with open(image_file,'r+b') as file:
        file.truncate(state['rows']*64*3*window);file.seek(0,2)
        started=time.monotonic();initial=state['rows'];verified=0
        for security,g in p.groupby('security',sort=True):
            if state['last_security'] is not None and security<=state['last_security']:continue
            g,ends,index=stock_index(g,calendar,history,horizons,first_pos,last_pos)
            ohlcv=g[['open','high','low','close','volume']].to_numpy(float);tri=g.total_return_index_gross_dividends.to_numpy(float)
            counts=[]
            for k in range(0,len(ends),1024):
                block=ends[k:k+1024];images=render(ohlcv,tri,block,window,ma)
                if len(block) and verified<100:
                    end=int(block[0]);expected=render_window(ohlcv[end-history+1:end+1],tri[end-history+1:end+1],window=window,width=window*3,ma_length=ma)
                    if not np.array_equal(images[0],expected):raise AssertionError(f'Render parity failure: {security} {end}')
                    verified+=1
                images.tofile(file);counts.extend(np.count_nonzero(images,axis=(1,2,3)).tolist())
            index['white_pixels']=np.array(counts,dtype='int16');index['row_id']=np.arange(state['rows'],state['rows']+len(index),dtype='int64')
            part=cache/'parts'/f"{state['stocks_processed']:05d}.parquet"
            if len(index):
                index.to_parquet(part.with_suffix('.tmp.parquet'),index=False);part.with_suffix('.tmp.parquet').replace(part)
            file.flush();os.fsync(file.fileno())
            state.update(rows=state['rows']+len(index),last_security=security,stocks_processed=state['stocks_processed']+1,updated_utc=pd.Timestamp.now(tz='UTC').isoformat(),current_build_reference_parity_checks=verified)
            atomic_json(state_path,state)
            if state['stocks_processed']%100==0:print('CACHE',state['stocks_processed'],'stocks',state['rows'],'images',round((state['rows']-initial)/max(time.monotonic()-started,1)),'images/sec',flush=True)
    writer=None
    for part in sorted((cache/'parts').glob('*.parquet')):
        table=pq.read_table(part)
        if writer is None:writer=pq.ParquetWriter(cache/'index.tmp.parquet',table.schema)
        writer.write_table(table)
    if writer is None:raise ValueError('Empty cache')
    writer.close();(cache/'index.tmp.parquet').replace(cache/'index.parquet')
    assert pq.ParquetFile(cache/'index.parquet').metadata.num_rows==state['rows']
    assert image_file.stat().st_size==state['rows']*64*3*window
    index=pd.read_parquet(cache/'index.parquet',columns=['row_id','date'])
    assert np.array_equal(index.row_id,np.arange(state['rows']))
    assert index.date.max()<pd.Timestamp('2023-01-01')
    state['state']='complete';state['completed_utc']=pd.Timestamp.now(tz='UTC').isoformat();atomic_json(state_path,state)
    return state
