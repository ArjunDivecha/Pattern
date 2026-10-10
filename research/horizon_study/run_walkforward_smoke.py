"""Two annual fits using the common study schedule, small deterministic samples.
Only development data loaded; no reserved-year performance inspected.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import json,hashlib,shutil
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from pattern.data.frozen import load_frozen,samples
from pattern.research.walkforward import annual_schedule,describe
from pattern.research.tiles import diagnostics,tile_report
from pattern.research.report import write_report
from pattern.config import ImageConfig,ModelConfig,TrainConfig
from pattern.models.cnn import ChartCNN
from pattern.train.loop import train_model,predict


def main():
    torch.set_num_threads(4)
    ref=json.loads((ROOT/'data/frozen/CURRENT.json').read_text());root=Path(ref['path'])
    cal=pd.DatetimeIndex(pd.read_parquet(root/'candidate_sessions.parquet').date)
    cfg=json.loads((ROOT/'research/horizon_study/walkforward_config.json').read_text())
    schedule=annual_schedule(cal,**{k:cfg[k] for k in ['train_start','first_test','last_test','reserve_start','max_history','max_horizon','validation_sessions']})
    describe(schedule).to_csv(ROOT/'research/horizon_study/annual_schedule.csv',index=False)
    out=ROOT/'runs'/('walkforward_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
    shutil.copyfile(ROOT/'research/horizon_study/walkforward_config.json',out/'study_config.json')
    source=out/'source';source.mkdir()
    for p in [Path(__file__),ROOT/'pattern/research/walkforward.py',ROOT/'pattern/data/frozen.py',ROOT/'pattern/research/tiles.py',ROOT/'pattern/research/report.py',ROOT/'pattern/train/loop.py',ROOT/'pattern/imaging/renderer.py',ROOT/'pattern/models/cnn.py',ROOT/'pattern/models/blocks.py',ROOT/'pattern/config.py']:
        shutil.copyfile(p,source/p.name)
    (out/'source_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},indent=2))
    first=cal[cal.searchsorted(pd.Timestamp(cfg['train_start']))]
    q=pd.read_parquet(root/'quality.parquet',columns=['security','current_renderer_bar_ok'],filters=[('date','==',first)])
    names=sorted(q.loc[q.current_renderer_bar_ok,'security'],key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:100]
    (out/'securities.json').write_text(json.dumps(names))
    folds=schedule[:2]
    panel,_,_=load_frozen(ROOT/'data/frozen/CURRENT.json',first,cal[cal.searchsorted(folds[-1]['test'][-1])+5],names)
    all_scores=[];annual=[];verification=[]
    for fold in folds:
        year=fold['year'];dest=out/str(year);dest.mkdir();torch.manual_seed(17);rng=np.random.default_rng(17)
        # Spread training/validation sample dates throughout their full allowed histories.
        dates={part:fold[part][np.unique(np.linspace(0,len(fold[part])-1,cap).astype(int))] for part,cap in [('train',40),('validation',10)]}
        dates['test']=fold['test'][:30]
        arrays={};idx={}
        for part in dates:
            arrays[part],idx[part]=samples(panel,cal,dates[part],10,5,10)
            idx[part].to_parquet(dest/f'{part}_index.parquet',index=False)
        assert idx['train'].target_date.max()<idx['validation'].input_start.min()
        assert idx['validation'].target_date.max()<idx['test'].input_start.min()
        assert idx['test'].target_date.max()<pd.Timestamp(cfg['reserve_start'])
        labels=idx['train'].label.to_numpy();groups=[np.flatnonzero(labels==i) for i in [0,1]];n=min(map(len,groups));assert n>0
        chosen=np.concatenate([rng.choice(g,n,replace=False) for g in groups]);rng.shuffle(chosen)
        train=arrays['train'][chosen].astype('float32')/255;mean=float(train.mean());std=float(train.std());assert std>0
        loaders={}
        for part in dates:
            ids=chosen if part=='train' else np.flatnonzero(idx[part].label.notna()) if part=='validation' else np.arange(len(idx[part]))
            x=(arrays[part][ids].astype('float32')/255-mean)/std
            y=idx[part].label.iloc[ids].fillna(0).to_numpy('int64')
            loaders[part]=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=64,shuffle=part=='train')
        model=ChartCNN(ModelConfig(blocks=2,channels=[8,16]),ImageConfig(window=10,width=30))
        model,_=train_model(model,loaders['train'],loaders['validation'],TrainConfig(max_epochs=2,lr=1e-4,device='cpu'),dest,seed=17)
        probs,_=predict(model,loaders['test'],device_str='cpu')
        scores=idx['test'].copy();scores['score']=probs[:,1].numpy();scores['fit_year']=year
        assert np.isfinite(scores.score).all();scores.to_parquet(dest/'test_scores.parquet',index=False)
        restored=ChartCNN(ModelConfig(blocks=2,channels=[8,16]),ImageConfig(window=10,width=30));restored.load_state_dict(torch.load(dest/'model.pt',weights_only=True))
        again,_=predict(restored,loaders['test'],device_str='cpu');assert torch.allclose(probs,again)
        daily,summary,assigned,spread,ic,cohorts,metrics=diagnostics(scores,cal,5)
        for name,frame in [('deciles_daily',daily),('deciles_summary',summary),('decile_assignments',assigned),('daily_spreads',spread),('rank_ic',ic),('nonoverlapping_cohorts',cohorts)]:frame.to_csv(dest/f'{name}.csv')
        for tiles in [5,20]:
            a,b,_=tile_report(scores,tiles);a.to_csv(dest/f'tiles_{tiles}_daily.csv');b.to_csv(dest/f'tiles_{tiles}_summary.csv')
        (dest/'metrics.json').write_text(json.dumps(metrics,indent=2))
        (dest/'config.json').write_text(json.dumps({'dataset':ref,'lookback':10,'horizon':5,'ma':10,'effective_history':19,'seed':17,'epochs':2,'channels':[8,16],'stocks_selected':len(names),'pixel_mean_train_only':mean,'pixel_std_train_only':std,'fit_year':year,'test_scope':'first 30 sessions only; smoke test'},indent=2))
        write_report(dest)
        annual.append({'year':year,**metrics});all_scores.append(scores)
        verification.append({'year':year,'split_intervals_disjoint':True,'checkpoint_predictions_match':True,'test_scores':len(scores),'reserved_outcomes_accessed':False})
        print('Completed',year,len(scores),flush=True)
    combined=pd.concat(all_scores,ignore_index=True);assert not combined.duplicated(['security','date']).any()
    combined.to_parquet(out/'combined_oos_scores.parquet',index=False)
    pd.testing.assert_frame_equal(combined,pd.read_parquet(out/'combined_oos_scores.parquet'))
    pd.DataFrame(annual).to_csv(out/'annual_metrics.csv',index=False)
    daily,summary,_=tile_report(combined);daily.to_csv(out/'combined_deciles_daily.csv');summary.to_csv(out/'combined_deciles_summary.csv')
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    (out/'README.md').write_text('# Annual walk-forward smoke test\n\nTwo independent annual fits (2010 and 2011), using the common full-grid purge schedule. Daily test signals cover only the first 30 sessions of each year. Training and validation dates are subsampled for an operational test. No reserved-year outcomes were loaded. This is not full-year or full-universe investment evidence.\n\nSee annual_metrics.csv and combined_deciles_summary.csv; each year has a checkpoint, train/validation/test index, decile report and unresolved-outcome counts. Each model uses training-only normalization and balancing. All checkpoint predictions match after reload.\n')
    print('OUTPUT',out,flush=True)

if __name__=='__main__':main()
