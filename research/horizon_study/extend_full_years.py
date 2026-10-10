"""Score full 2010/2011 test years using unchanged saved smoke-test checkpoints.
Gross-only research extension; no retraining, new stock selection or holdout access.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from datetime import datetime,timezone
import json,hashlib,shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from pattern.data.frozen import load_frozen,samples
from pattern.config import ModelConfig,ImageConfig
from pattern.models.cnn import ChartCNN
from pattern.train.loop import predict
from pattern.research.gross_signal import gross_diagnostics


def main():
    torch.set_num_threads(4)
    original=ROOT/'runs/walkforward_smoke_20260918T073832Z'
    out=ROOT/'runs'/('gross_full_years_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
    names=json.loads((original/'securities.json').read_text())
    panel,cal,ref=load_frozen(ROOT/'data/frozen/CURRENT.json','2009-01-01','2012-01-31',names)
    cal=pd.DatetimeIndex(cal)
    snapshots=out/'source';snapshots.mkdir()
    for p in [Path(__file__),ROOT/'pattern/research/gross_signal.py',ROOT/'pattern/research/tiles.py',ROOT/'pattern/data/frozen.py',ROOT/'pattern/imaging/renderer.py',ROOT/'pattern/models/cnn.py',ROOT/'pattern/models/blocks.py',ROOT/'pattern/config.py',ROOT/'pattern/train/loop.py']:
        shutil.copyfile(p,snapshots/p.name)
    manifest={'dataset':ref,'source_run':str(original),'purpose':'full-year gross-signal extension, unchanged models and securities','years':[2010,2011],
              'lookback':10,'horizon':5,'ma':10,'stocks_selected':len(names),'checkpoints':{},'costs_evaluated':False,
              'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshots.iterdir()}}
    all_metrics=[];all_scores=[]
    for year in [2010,2011]:
        dest=out/str(year);dest.mkdir();cfg=json.loads((original/str(year)/'config.json').read_text());checkpoint=original/str(year)/'model.pt'
        manifest['checkpoints'][str(year)]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        shutil.copyfile(checkpoint,dest/'model.pt');shutil.copyfile(original/str(year)/'config.json',dest/'training_config.json')
        model=ChartCNN(ModelConfig(blocks=2,channels=[8,16]),ImageConfig(window=10,width=30));model.load_state_dict(torch.load(checkpoint,weights_only=True))
        parts=[]
        for quarter in [1,2,3,4]:
            dates=cal[(cal.year==year)&(cal.quarter==quarter)]
            images,index=samples(panel,cal,dates,10,5,10)
            x=(images.astype('float32')/255-cfg['pixel_mean_train_only'])/cfg['pixel_std_train_only']
            loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.zeros(len(x),dtype=torch.long)),batch_size=64)
            probs,_=predict(model,loader,device_str='cpu');index['score']=probs[:,1].numpy();index['fit_year']=year
            assert np.isfinite(index.score).all();parts.append(index)
            print(year,'quarter',quarter,'scores',len(index),flush=True)
        scores=pd.concat(parts,ignore_index=True).sort_values(['security','date']).reset_index(drop=True)
        assert not scores.duplicated(['security','date']).any()
        scores.to_parquet(dest/'scores.parquet',index=False)
        pd.testing.assert_frame_equal(scores,pd.read_parquet(dest/'scores.parquet'))
        old=pd.read_parquet(original/str(year)/'test_scores.parquet')
        matched=old.merge(scores,on=['security','date'],suffixes=('_old','_new'),validate='one_to_one')
        assert len(matched)==len(old) and np.allclose(matched.score_old,matched.score_new,atol=1e-6,rtol=1e-5)
        assert np.allclose(matched.forward_return_old,matched.forward_return_new,equal_nan=True)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['checkpoints'][str(year)]
        metrics,tables=gross_diagnostics(scores,cal,5)
        for name,table in tables.items():table.to_csv(dest/f'{name}.csv')
        (dest/'metrics.json').write_text(json.dumps(metrics,indent=2));all_metrics.append({'year':year,**metrics});all_scores.append(scores)
        print(year,'metrics',json.dumps(metrics),flush=True)
    combined=pd.concat(all_scores,ignore_index=True);combined.to_parquet(out/'combined_scores.parquet',index=False)
    pd.DataFrame(all_metrics).to_csv(out/'annual_gross_metrics.csv',index=False)
    manifest['checks']={'unique_stock_dates':not combined.duplicated(['security','date']).any(),'original_checkpoint_hashes_unchanged':True,
      'prior_30_session_predictions_reproduced':True,'saved_scores_read_back_equal':True,'contributions_reconcile':True,'reserved_years_accessed':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('OUTPUT',out,flush=True)

if __name__=='__main__':main()
