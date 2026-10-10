import json
import numpy as np
import pandas as pd
from pattern.research.full_cache import build_cache,stock_index
from pattern.data.frozen import samples


def panel():
    cal=pd.bdate_range('1996-01-01',periods=120)
    p=pd.DataFrame({'security':'A','date':cal,'session_index':np.arange(len(cal)),'calendar_segment':0,
        'current_renderer_bar_ok':True,'return_endpoint_ok':True,'open':10.,'high':11.,'low':9.,'close':10.,'volume':100.,'total_return_index_gross_dividends':100*1.001**np.arange(len(cal))})
    return p,cal


def test_fast_index_matches_calendar_reference():
    p,c=panel();p=p.drop(index=[55]);p.loc[p.date>c[55],'calendar_segment']=1
    g,ends,index=stock_index(p,c,39,[5,20],38,90)
    for h in [5,20]:
        _,ref=samples(p,c,index.date,20,h,20)
        assert np.array_equal(index.date.values,ref.date.values)
        assert np.allclose(index[f'forward_return_{h}'],ref.forward_return,equal_nan=True)


def test_disk_cache_round_trip_and_resume(tmp_path):
    p,c=panel();root=tmp_path/'frozen';root.mkdir();cache=tmp_path/'cache'
    (root/'manifest.json').write_text(json.dumps({'fixture':True}))
    p.to_parquet(root/'prices.parquet',index=False)
    p[['security','date','current_renderer_bar_ok','return_endpoint_ok']].to_parquet(root/'quality.parquet',index=False)
    p[['security','date','session_index','calendar_segment']].to_parquet(root/'calendar_quality.parquet',index=False)
    state=build_cache(root,cache,c,38,90);again=build_cache(root,cache,c,38,90)
    assert state['state']=='complete' and again['rows']==53
    idx=pd.read_parquet(cache/'index.parquet');mm=np.memmap(cache/'images.bin',dtype='uint8',mode='r',shape=(len(idx),1,64,60))
    expected,ref=samples(p,c,idx.date,20,5,20)
    assert np.array_equal(mm,expected)
    assert np.array_equal(idx.white_pixels,np.count_nonzero(mm,axis=(1,2,3)))
