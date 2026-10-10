"""Verify a frozen snapshot without querying GDM or using external services."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
import pyarrow.parquet as pq

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(root):
    root=Path(root); m=json.loads((root/'manifest.json').read_text()); checks=0
    for name,info in m['files'].items():
        assert sha(root/name)==info['sha256'],name
    checks+=1
    p=pd.read_parquet(root/'prices.parquet',columns=['security','date'])
    assert len(p)==m['rows'] and p.security.nunique()==m['securities']
    assert not p.isna().any().any() and not p.duplicated().any(); checks+=1
    p=p.sort_values(['security','date']).reset_index(drop=True)
    for name in ['quality.parquet','calendar_quality.parquet']:
        q=pd.read_parquet(root/name,columns=['security','date'])
        pd.testing.assert_frame_equal(p,q,check_dtype=False); checks+=1
    for name in ['quality.parquet','calendar_quality.parquet']:
        assert not any(('20' in c or '39' in c) for c in pq.read_schema(root/name).names)
    checks+=1
    q=pd.read_parquet(root/'quality.parquet',columns=['return_pair_ok','screened_total_return','unresolved_identity_boundary','current_renderer_bar_ok','ohlc_nonpositive','ohlc_inconsistent'])
    assert q.loc[~q.return_pair_ok,'screened_total_return'].isna().all()
    assert not q.loc[q.ohlc_nonpositive|q.ohlc_inconsistent,'current_renderer_bar_ok'].any(); checks+=1
    cal=pd.read_parquet(root/'candidate_sessions.parquet').date
    assert cal.is_monotonic_increasing and not cal.duplicated().any(); checks+=1
    assert m['lookback_sessions'] is None and m['forecast_horizon_sessions'] is None
    assert m['source_rows_dropped']==0; checks+=1
    return {'passed_checks':checks,'rows_verified':len(p),'all_file_hashes_match':True,
            'scope':'Frozen artifact integrity and screening behavior; not independent vendor price certification'}

if __name__=='__main__':print(json.dumps(verify(sys.argv[1]),indent=2))
