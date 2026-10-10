"""Freeze existing GDM data and practical screening layers; no network access."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys
import pyarrow.parquet as pq
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'research/data_audit_20260917'
DEST=ROOT/'data/frozen/pattern_daily_v1_20260918'

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    if DEST.exists(): raise FileExistsError('Frozen version already exists; never overwrite')
    sys.path.insert(0,str(ROOT.parent/'Global Data Mart'))
    from gdm import dataset
    price=Path(dataset('equities.research_vintage_prices',acknowledge='survivorship-incomplete'))
    master=Path(dataset('equities.research_vintage_security_master'))
    staging=DEST.with_name(DEST.name+'.building')
    staging.mkdir(parents=True,exist_ok=False)
    provenance={}
    def copy(src,name):
        src=Path(src); before=digest(src)
        shutil.copyfile(src,staging/name)
        assert digest(staging/name)==before==digest(src)
        provenance[name]={'source':str(src),'source_sha256':before}
    copy(price,'prices.parquet')
    copy(master,'security_master.parquet')
    quality=AUDIT/'quality_layer_20260918T064856Z/quality.parquet'
    calendar=AUDIT/'calendar_layer_20260918T070935Z/calendar_quality.parquet'
    for src,name,exclude in [(quality,'quality.parquet',lambda c:'observations_ok' in c or 'observation_intervals_ok' in c),
                               (calendar,'calendar_quality.parquet',lambda c:c.startswith(('past_','forward_','target_')))]:
        before=digest(src)
        cols=[c for c in pq.read_schema(src).names if not exclude(c)]
        pq.write_table(pq.read_table(src,columns=cols),staging/name,compression='zstd')
        assert digest(src)==before
        provenance[name]={'source':str(src),'source_sha256':before,'retained_columns':cols}
    cal=AUDIT/'calendar_layer_20260918T070935Z'
    for filename in ['candidate_sessions.parquet','calendar_evidence.csv','excluded_weekdays_review.csv']:
        copy(cal/filename,filename)
    events=AUDIT/'adjustment_audit_20260918T065713Z'
    for filename in ['cash_events.csv','split_events.csv','split_conflicts.csv','review_summary.json']:
        copy(events/filename,'adjustment_'+filename)
    shutil.copyfile(Path(__file__),staging/'freeze_build.py')
    shutil.copyfile(Path(__file__).with_name('verify.py'),staging/'verify.py')
    shutil.copyfile(Path(__file__).with_name('CONTRACT.md'),staging/'README.md')
    keys=pq.read_table(staging/'prices.parquet',columns=['security','date']).to_pandas()
    assert not keys.isna().any().any() and not keys.duplicated().any()
    keys=keys.sort_values(['security','date']).reset_index(drop=True)
    for name in ['quality.parquet','calendar_quality.parquet']:
        other=pq.read_table(staging/name,columns=['security','date']).to_pandas()
        pd.testing.assert_frame_equal(keys,other,check_dtype=False)
    manifest={'dataset_id':DEST.name,'status':'FROZEN_RESEARCH_BASELINE','version':1,
        'created_utc':datetime.now(timezone.utc).isoformat(),'rows':len(keys),
        'securities':int(keys.security.nunique()),'first_date':str(keys.date.min().date()),
        'last_date':str(keys.date.max().date()),'source_rows_dropped':0,'vendor_calls':0,
        'lookback_sessions':None,'forecast_horizon_sessions':None,
        'horizons':'Chosen independently per experiment; no precomputed labels or fixed window filters',
        'calendar_status':'Empirical existing-data calendar; frozen approximation, not exchange-certified',
        'coverage_status':'GDM research coverage gate remains uncleared; not claimed survivorship-complete',
        'sources':provenance,'files':{}}
    for f in sorted(staging.iterdir()):
        manifest['files'][f.name]={'sha256':digest(f),'bytes':f.stat().st_size}
        if f.suffix=='.parquet':
            manifest['files'][f.name]['schema']=str(pq.read_schema(f))
            manifest['files'][f.name]['rows']=pq.ParquetFile(f).metadata.num_rows
    (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    from verify import verify
    result=verify(staging)
    (staging/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    staging.rename(DEST)
    for f in DEST.iterdir(): f.chmod(0o444)
    (DEST.parent/'CURRENT.json').write_text(json.dumps({'dataset_id':DEST.name,'path':str(DEST),'manifest_sha256':digest(DEST/'manifest.json')},indent=2)+'\n')
    print(json.dumps({'path':str(DEST),'manifest':manifest,'verification':result},indent=2),flush=True)

if __name__=='__main__':main()
