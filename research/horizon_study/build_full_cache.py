"""Build the common full-universe I20 cache from the frozen daily database."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
from pattern.research.walkforward import annual_schedule
from pattern.research.full_cache import build_cache
if __name__=='__main__':
    ref=json.loads((ROOT/'data/frozen/CURRENT.json').read_text());root=Path(ref['path'])
    cal=pd.DatetimeIndex(pd.read_parquet(root/'candidate_sessions.parquet').date)
    folds=annual_schedule(cal)
    result=build_cache(root,ROOT/'cache/frozen_full_I20_M20_v1',cal,cal.get_loc(folds[0]['train'][0]),cal.get_loc(folds[-1]['test'][-1]))
    print(json.dumps(result,indent=2),flush=True)
