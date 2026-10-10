"""Resumable full-data annual I20/R5 and I20/R20 ensembles on the local GPU."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import argparse,json,hashlib,shutil,os,traceback
import numpy as np
import pandas as pd
import torch
from pattern.research.walkforward import annual_schedule,describe
from pattern.research.full_cache import build_cache,atomic_json
from pattern.research.full_train import fit,score
from pattern.research.gross_signal import gross_diagnostics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);args=parser.parse_args()
    out=Path(args.run_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'RUNNING.lock'
    # No two writers may train or publish to one run directory.
    if lock.exists():
        pid=int(lock.read_text())
        try:os.kill(pid,0)
        except ProcessLookupError:lock.unlink()
        else:raise RuntimeError(f'Run already active, PID {pid}')
    lock.write_text(str(os.getpid()))
    try:
        cfg=json.loads((ROOT/'research/horizon_study/full_run_config.json').read_text())
        ref=json.loads((ROOT/cfg['dataset_pointer']).read_text());frozen=Path(ref['path']);cache=ROOT/cfg['cache']
        fingerprint=hashlib.sha256(json.dumps({'config':cfg,'dataset':ref},sort_keys=True).encode()).hexdigest()
        if (out/'config.json').exists():
            if json.loads((out/'config.json').read_text())['fingerprint']!=fingerprint:raise ValueError('Run configuration changed')
        else:atomic_json(out/'config.json',{'fingerprint':fingerprint,'config':cfg,'dataset':ref})
        source_paths=[ROOT/'pattern/research/full_train.py',ROOT/'pattern/research/full_cache.py',ROOT/'pattern/research/gross_signal.py',ROOT/'pattern/research/tiles.py',ROOT/'pattern/research/walkforward.py',ROOT/'pattern/imaging/fast_renderer.c',ROOT/'pattern/imaging/fast.py',ROOT/'pattern/models/cnn.py',ROOT/'pattern/models/blocks.py',ROOT/'pattern/config.py',Path(__file__)]
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
        if (out/'source_hashes.json').exists():
            if json.loads((out/'source_hashes.json').read_text())!=hashes:raise ValueError('Source changed; start a new run or explicitly migrate')
        else:
            (out/'source').mkdir(exist_ok=True)
            for p in source_paths:shutil.copyfile(p,out/'source'/str(p.relative_to(ROOT)).replace('/','__'))
            atomic_json(out/'source_hashes.json',hashes)
        if cfg['device']=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; launch outside sandbox. No silent CPU fallback.')
        torch.set_num_threads(4);device=torch.device(cfg['device'])
        atomic_json(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'device':str(device),'pid':os.getpid()})
        cal=pd.DatetimeIndex(pd.read_parquet(frozen/'candidate_sessions.parquet').date)
        folds=annual_schedule(cal,train_start=cfg['train_start'],first_test=cfg['first_test'],last_test=cfg['last_test'],reserve_start=cfg['reserve_start'],max_history=cfg['common_split_max_history'],max_horizon=cfg['common_split_max_horizon'])
        describe(folds).to_csv(out/'annual_schedule.csv',index=False)
        atomic_json(out/'status.json',{'state':'building_cache','updated_utc':pd.Timestamp.now(tz='UTC').isoformat()})
        state=build_cache(frozen,cache,cal,cal.get_loc(folds[0]['train'][0]),cal.get_loc(folds[-1]['test'][-1]),window=20,ma=20,horizons=tuple(cfg['horizons']))
        images=np.memmap(cache/'images.bin',dtype='uint8',mode='r',shape=(state['rows'],1,64,60));index=pd.read_parquet(cache/'index.parquet')
        if not np.array_equal(index.row_id,np.arange(len(index))):raise ValueError('Cache row IDs not contiguous')
        for fold in folds:
            test_ids=np.flatnonzero(index.date.between(fold['test'][0],fold['test'][-1]).to_numpy())
            if not len(test_ids):raise ValueError('Empty test fold')
            for horizon in cfg['horizons']:
                dest=out/f"I20_R{horizon}"/str(fold['year']);dest.mkdir(parents=True,exist_ok=True)
                if (dest/'COMPLETE.json').exists():continue
                returns=index[f'forward_return_{horizon}'].to_numpy()
                train_ids=np.flatnonzero(index.date.between(fold['train'][0],fold['train'][-1]).to_numpy()&np.isfinite(returns))
                val_ids=np.flatnonzero(index.date.between(fold['validation'][0],fold['validation'][-1]).to_numpy()&np.isfinite(returns))
                pos=index.session_position.to_numpy()
                assert pos[train_ids].max()+horizon<pos[val_ids].min()-38
                assert pos[val_ids].max()+horizon<pos[test_ids].min()-38
                assert cal[pos[test_ids].max()+horizon]<pd.Timestamp(cfg['reserve_start'])
                atomic_json(dest/'sample_counts.json',{'train_known':len(train_ids),'validation_known':len(val_ids),'test_all_inputs':len(test_ids),'test_unknown_outcomes':int(np.isnan(returns[test_ids]).sum()),'test_stocks':int(index.iloc[test_ids].security.nunique()),'test_dates':int(index.iloc[test_ids].date.nunique())})
                probs=[]
                for seed in cfg['seeds']:
                    seed_dir=dest/f'seed_{seed}';seed_dir.mkdir(exist_ok=True);prediction=seed_dir/'scores.npy'
                    atomic_json(out/'status.json',{'state':'training','year':fold['year'],'horizon':horizon,'seed':seed,'progress_file':str(seed_dir/'progress.json'),'updated_utc':pd.Timestamp.now(tz='UTC').isoformat()})
                    if prediction.exists():
                        prob=np.load(prediction);assert prob.shape==(len(test_ids),)
                    else:
                        model,mean,std=fit(images,index,returns,train_ids,val_ids,seed_dir,seed,device,cfg['batch_size'],cfg['max_epochs'],cfg['patience'],cfg['learning_rate'])
                        atomic_json(out/'status.json',{'state':'scoring','year':fold['year'],'horizon':horizon,'seed':seed})
                        prob=score(model,images,test_ids,mean,std,device,cfg['batch_size'])
                        # A loaded best checkpoint must reproduce a real held-out batch.
                        from pattern.models.cnn import ChartCNN
                        from pattern.config import ModelConfig,ImageConfig
                        loaded=ChartCNN(ModelConfig(),ImageConfig()).to(device);loaded.load_state_dict(torch.load(seed_dir/'best.pt',map_location='cpu',weights_only=True))
                        check=score(loaded,images,test_ids[:256],mean,std,device,cfg['batch_size'])
                        if not np.allclose(check,prob[:256],atol=1e-6,rtol=1e-5):raise AssertionError('Checkpoint prediction mismatch')
                        with open(prediction.with_suffix('.tmp'),'wb') as f:np.save(f,prob)
                        prediction.with_suffix('.tmp').replace(prediction)
                        del model,loaded
                        if device.type=='mps':torch.mps.empty_cache()
                    if not np.isfinite(prob).all():raise ValueError('Invalid scores')
                    probs.append(prob)
                scores=index.iloc[test_ids][['security','date']].copy();scores['forward_return']=returns[test_ids];scores['score']=np.mean(probs,axis=0);scores['fit_year']=fold['year']
                scores.to_parquet(dest/'ensemble_scores.parquet',index=False)
                metrics,tables=gross_diagnostics(scores,cal,horizon)
                for name,table in tables.items():table.to_parquet(dest/f'{name}.parquet',index=True)
                atomic_json(dest/'metrics.json',metrics)
                atomic_json(dest/'COMPLETE.json',{'seeds':cfg['seeds'],'rows':len(scores),'completed_utc':pd.Timestamp.now(tz='UTC').isoformat()})
                print('COMPLETE YEAR',fold['year'],'HORIZON',horizon,json.dumps(metrics),flush=True)
        atomic_json(out/'status.json',{'state':'complete','completed_utc':pd.Timestamp.now(tz='UTC').isoformat()})
    except BaseException as error:
        atomic_json(out/'status.json',{'state':'failed','error':repr(error),'traceback':traceback.format_exc(),'updated_utc':pd.Timestamp.now(tz='UTC').isoformat()});raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
