"""Exercise frozen data -> independent L/H/MA images -> CNN -> out-of-sample tiles.
Run from repository root with .venv/bin/python research/horizon_study/run_pilot.py.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import argparse,json,hashlib,subprocess,shutil
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from pattern.data.frozen import load_frozen,samples
from pattern.config import ImageConfig,ModelConfig,TrainConfig
from pattern.models.cnn import ChartCNN
from pattern.train.loop import train_model,predict
from pattern.research.tiles import diagnostics,tile_report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lookback',type=int,default=10);ap.add_argument('--horizon',type=int,default=5);ap.add_argument('--ma',type=int,default=10);args=ap.parse_args()
    if args.lookback+args.ma-1>40 or args.horizon>=30:raise ValueError('Pilot date schedule supports history<=40 and H<30; full-grid runner requires wider splits')
    torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17)
    out=ROOT/'runs'/('frozen_pilot_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(parents=True)
    code=out/'source';code.mkdir()
    for path in ['pattern/data/frozen.py','pattern/imaging/renderer.py','pattern/models/cnn.py','pattern/models/blocks.py','pattern/train/loop.py','pattern/config.py','pattern/research/tiles.py','pattern/research/report.py','research/horizon_study/run_pilot.py','research/horizon_study/SPEC.md']:
        dest=code/path.replace('/','__');shutil.copyfile(ROOT/path,dest)
    (out/'source_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.iterdir()},indent=2))
    (out/'versions.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__},indent=2))
    pointer=ROOT/'data/frozen/CURRENT.json';ref=json.loads(pointer.read_text());frozen=Path(ref['path'])
    sessions=pd.read_parquet(frozen/'candidate_sessions.parquet').date
    year=sessions[sessions.dt.year==2020].reset_index(drop=True)
    first=year.iloc[0]
    q=pd.read_parquet(frozen/'quality.parquet',columns=['security','current_renderer_bar_ok'],filters=[('date','==',first)])
    names=sorted(q.loc[q.current_renderer_bar_ok,'security'],key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:120]
    (out/'securities.json').write_text(json.dumps(names,indent=2))
    panel,sessions,ref=load_frozen(pointer,first,year.iloc[-1],names)
    schedule={'train':year.iloc[40:90],'validation':year.iloc[130:145],'test':year.iloc[180:210]}
    arrays={};indexes={}
    for part,dates in schedule.items():
        x,index=samples(panel,sessions,dates,args.lookback,args.horizon,args.ma)
        index.to_parquet(out/f'{part}_index.parquet',index=False)
        arrays[part]=x;indexes[part]=index
        print(part,len(index),'unknown',int(index.label.isna().sum()),flush=True)
    assert indexes['train'].target_date.max()<indexes['validation'].input_start.min()
    assert indexes['validation'].target_date.max()<indexes['test'].input_start.min()
    # Unknown supervised outcomes excluded only from train/validation. Test scores retained.
    train_known=indexes['train'].label.notna().to_numpy()
    train_ids=np.flatnonzero(train_known);rng=np.random.default_rng(17)
    y=indexes['train'].label.to_numpy()
    groups=[train_ids[y[train_ids]==v] for v in [0,1]];size=min(map(len,groups))
    if size==0:raise ValueError("Training sample must contain both classes")
    chosen=np.concatenate([rng.choice(g,size,replace=False) for g in groups]);rng.shuffle(chosen)
    raw_train=arrays['train'][chosen].astype(np.float32)/255
    mean=float(raw_train.mean());std=float(raw_train.std());assert std>0
    loaders={}
    for part in schedule:
        ids=chosen if part=='train' else (np.flatnonzero(indexes[part].label.notna()) if part=='validation' else np.arange(len(indexes[part])))
        x=(arrays[part][ids].astype(np.float32)/255-mean)/std
        label=indexes[part].label.iloc[ids].fillna(0).to_numpy(np.int64)
        loaders[part]=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(label)),batch_size=64,shuffle=part=='train',num_workers=0)
    cfg={'dataset':ref,'lookback':args.lookback,'horizon':args.horizon,'ma':args.ma,'effective_history':args.lookback+args.ma-1,
         'seed':17,'epochs':2,'channels':[8,16],'blocks':2,'pixel_mean_train_only':mean,'pixel_std_train_only':std,
         'balanced_train_samples':len(chosen),'purpose':'operational pilot, not evidence of predictive value',
         'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    image=ImageConfig(window=args.lookback,width=3*args.lookback)
    model=ChartCNN(ModelConfig(blocks=2,channels=[8,16]),image)
    model,history=train_model(model,loaders['train'],loaders['validation'],TrainConfig(max_epochs=2,device='cpu',lr=1e-4,num_workers=0),out,seed=17)
    probs,_=predict(model,loaders['test'],device_str='cpu')
    scores=indexes['test'].copy();scores['score']=probs[:,1].numpy()
    assert np.isfinite(scores.score).all()
    scores.to_parquet(out/'test_scores.parquet',index=False)
    daily,summary,assigned,spreads,ic,cohorts,metrics=diagnostics(scores,sessions,args.horizon)
    for name,frame in [('deciles_daily',daily),('deciles_summary',summary),('decile_assignments',assigned),('daily_spreads',spreads),('rank_ic',ic),('nonoverlapping_cohorts',cohorts)]:frame.to_csv(out/f'{name}.csv',index=True)
    for n in [5,20]:
        a,b,_=tile_report(scores,n);a.to_csv(out/f'tiles_{n}_daily.csv',index=False);b.to_csv(out/f'tiles_{n}_summary.csv')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    # Exercise saved results and checkpoint, rather than only live tensors.
    back=pd.read_parquet(out/'test_scores.parquet');pd.testing.assert_frame_equal(scores,back)
    restored=ChartCNN(ModelConfig(blocks=2,channels=[8,16]),image)
    restored.load_state_dict(torch.load(out/'model.pt',weights_only=True))
    check,_=predict(restored,loaders['test'],device_str='cpu');assert torch.allclose(probs,check)
    from pattern.research.report import write_report
    write_report(out)
    print('OUTPUT',out,flush=True);print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
