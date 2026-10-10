"""Build fixed development samples and run small controlled CNN experiments."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import os
import shutil
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from pattern.config import ImageConfig, ModelConfig
from pattern.models.cnn import ChartCNN
from pattern.research.development import sample_partition, verify_splits, cohort_schedule, evaluate
from pattern.research.full_cache import atomic_json
from pattern.research.checkpoint import atomic_save, save_epoch, restore_epoch, cpu_tree
from pattern.research import tracking


def prepare(cfg, out):
    ref = json.loads((ROOT / cfg['dataset_pointer']).read_text())
    frozen = Path(ref['path'])
    if hashlib.sha256((frozen/'manifest.json').read_bytes()).hexdigest() != ref['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    calendar = pd.DatetimeIndex(pd.read_parquet(frozen/'candidate_sessions.parquet').date)
    cache = ROOT / cfg['cache']
    state = json.loads((cache/'state.json').read_text())
    if state['state'] != 'complete' or state['config']['source_manifest_sha256'] != ref['manifest_sha256']:
        raise ValueError('Wrong or incomplete source cache')
    window = state['config']['window']
    if state['config']['ma'] + window - 1 > cfg['max_history']:
        raise ValueError('Input history exceeds split protection')
    horizons = sorted({e['horizon'] for e in cfg['experiments']})
    if max(horizons) > cfg['max_horizon'] or not set(horizons) <= set(state['config']['horizons']):
        raise ValueError('Unsupported horizon')
    if cfg.get('prepared_sample_dir'):
        prepared=ROOT/cfg['prepared_sample_dir']
        old=json.loads((prepared/'config.json').read_text())
        for key in ['dataset_pointer','cache','sample_seed','train','validation','comparison','max_history','max_horizon']:
            if old[key]!=cfg[key]:raise ValueError(f'Prepared sample mismatch: {key}')
        if not set(horizons)<={e['horizon'] for e in old['experiments']}:raise ValueError('Missing prepared labels')
        for name in ['sample.parquet','coverage.csv','sample_metadata.json']:
            shutil.copyfile(prepared/name,out/name)
        os.link(prepared/'images.npy',out/'images.npy')
        manifest=pd.read_parquet(out/'sample.parquet')
        images=np.load(out/'images.npy',mmap_mode='r')
        if images.shape!=(len(manifest),1,64,window*3):raise ValueError('Prepared image dimensions differ')
        verify_splits({p:manifest[manifest.partition.eq(p)] for p in ['train','validation','comparison']},cfg['max_history'],cfg['max_horizon'])
        meta=json.loads((out/'sample_metadata.json').read_text())
        if meta['dataset']!=ref:raise ValueError('Prepared dataset identity differs')
        meta['purpose']=cfg.get('purpose','development comparison');meta['reused_from']=str(prepared)
        atomic_json(out/'sample_metadata.json',meta)
        c=cfg['comparison']
        return manifest,images,calendar,{h:cohort_schedule(calendar,c['start'],c['end'],h) for h in horizons},window
    index = pd.read_parquet(cache/'index.parquet', columns=['security','date','session_position','row_id']+[f'forward_return_{h}' for h in horizons])
    parts = {}
    for name in ['train','validation']:
        p = cfg[name]
        parts[name] = sample_partition(index, p['start'], p['end'], p['stocks'], cfg['sample_seed'], count=p['dates'])
    p = cfg['comparison']
    schedules = {h:cohort_schedule(calendar, p['start'], p['end'], h) for h in horizons}
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(x) for x in schedules.values()])))
    parts['comparison'] = sample_partition(index, p['start'], p['end'], p['stocks'], cfg['sample_seed'], dates=all_dates)
    verify_splits(parts, cfg['max_history'], cfg['max_horizon'])
    if calendar.get_loc(parts['comparison'].date.max()) + cfg['max_horizon'] >= calendar.searchsorted('2023-01-01'):
        raise ValueError('Possible reserved-outcome access')
    manifest = pd.concat([v.assign(partition=k) for k,v in parts.items()], ignore_index=True)
    manifest['sample_id'] = np.arange(len(manifest))
    manifest.to_parquet(out/'sample.parquet', index=False)
    manifest.assign(year=manifest.date.dt.year).groupby(['partition','year']).agg(rows=('security','size'),stocks=('security','nunique'),dates=('date','nunique')).to_csv(out/'coverage.csv')
    # A compact independent image pack; the source cache and frozen data remain read-only.
    src = np.memmap(cache/'images.bin', dtype='uint8', mode='r', shape=(state['rows'],1,64,3*window))
    dest = np.lib.format.open_memmap(out/'images.npy', mode='w+', dtype='uint8', shape=(len(manifest),1,64,3*window))
    for start in range(0,len(manifest),1024):
        dest[start:start+1024] = src[manifest.row_id.to_numpy()[start:start+1024]]
    dest.flush()
    atomic_json(out/'sample_metadata.json', {'dataset':ref,'source_cache':str(cache),'source_cache_config':state['config'],
        'image_window':window, 'ma_bars':state['config']['ma'],'bar_frequency':'daily','rows':len(manifest),
        'sample_seed':cfg['sample_seed'],'selection_uses_future_returns':False,
        'normalization':'training images only, recomputed after channel ablation',
        'target':'absolute forward return direction; binary cross entropy via two-class logits',
        'class_balance':'natural sampled distribution, no undersampling',
        'purpose':cfg.get('purpose','development comparisons; 2023 onward untouched')})
    return manifest, dest, calendar, schedules, window


def batch_x(images, ids, volume):
    x = images[ids].astype('float32')
    if not volume:
        x[:,:,52:,:] = 0
    return x


def normalize(x, mean, std):
    # NumPy scalar promotion must not turn MPS input batches into float64.
    return (x.astype(np.float32, copy=False)-np.float32(mean))/np.float32(std)


def train_and_score(cfg, exp, seed, sample, images, window, dest, device, schedule):
    dest.mkdir(parents=True,exist_ok=True)
    returns = sample[f"forward_return_{exp['horizon']}"].to_numpy()
    train = np.flatnonzero(sample.partition.eq('train') & np.isfinite(returns))
    val = np.flatnonzero(sample.partition.eq('validation') & np.isfinite(returns))
    comparison = np.flatnonzero(sample.partition.eq('comparison') & sample.date.isin(schedule))
    if not len(train) or not len(val) or not len(comparison):
        raise ValueError('Empty partition')
    # Binary renderer pixels allow exact training-only moments without large float copies.
    count = 0
    pixels = 0
    for start in range(0,len(train),1024):
        x = batch_x(images, train[start:start+1024], exp['volume'])
        count += int(np.count_nonzero(x)); pixels += x.size
    fraction = count/pixels
    mean, std = float(255*fraction), float(255*np.sqrt(fraction*(1-fraction)))
    if std <= 0:
        raise ValueError('Constant inputs')
    atomic_json(dest/'normalization.json', {'mean':mean,'std':std,'train_rows':len(train),'validation_rows':len(val),
        'comparison_rows':len(comparison),'train_up_fraction':float((returns[train]>0).mean())})
    torch.manual_seed(seed)
    model_config = ModelConfig(**exp.get('model',{}))
    model = ChartCNN(model_config, ImageConfig(window=window,width=3*window)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=exp['learning_rate'])
    signature=hashlib.sha256(json.dumps({'experiment':exp,'seed':seed,'window':window,
        'batch_size':cfg['batch_size'],'patience':cfg['patience'],'normalization':[mean,std],
        'sample_sha256':hashlib.sha256((dest.parents[1]/'sample.parquet').read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()
    best, stale, history = float('inf'), 0, []
    next_epoch=1;best_model=None
    if (dest/'last.pt').exists():
        state=restore_epoch(dest/'last.pt',model,opt,signature,device)
        best,stale,history=state['best'],state['stale'],state['history']
        next_epoch=state['epoch']+1;best_model=state['best_model']
        atomic_save(dest/'best.pt',best_model)
    wb=tracking.start(cfg,exp,seed,dest)
    def inputs(ids):
        return torch.from_numpy(normalize(batch_x(images,ids,exp['volume']),mean,std)).to(device)
    for epoch in range(next_epoch,cfg['max_epochs']+1):
        if stale>=cfg['patience']:break
        started = time.monotonic()
        torch.manual_seed(seed+epoch*10000)
        row = {'epoch':epoch}
        for phase, ids in [('train',train),('validation',val)]:
            model.train(phase=='train')
            order = ids.copy()
            if phase=='train':np.random.default_rng(seed+epoch).shuffle(order)
            total = 0.;correct=0
            with torch.set_grad_enabled(phase=='train'):
                for start in range(0,len(order),cfg['batch_size']):
                    chosen = order[start:start+cfg['batch_size']]
                    y = torch.from_numpy((returns[chosen]>0).astype('int64')).to(device)
                    logits = model(inputs(chosen))
                    loss = torch.nn.functional.cross_entropy(logits,y)
                    if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():
                        raise RuntimeError('Non-finite model output')
                    if phase=='train':
                        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                    total += float(loss.item())*len(chosen)
                    correct += int((logits.argmax(1)==y).sum().item())
            row[phase+'_loss'] = total/len(order)
            row[phase+'_accuracy'] = correct/len(order)
        row['seconds'] = time.monotonic()-started
        history.append(row)
        if row['validation_loss'] < best:
            best = row['validation_loss'];stale = 0
            best_model=cpu_tree(model.state_dict())
            atomic_save(dest/'best.pt',best_model)
        else:stale += 1
        state=save_epoch(dest/'last.pt',model,opt,epoch,best,stale,history,best_model,signature,device)
        if stale==0:atomic_save(dest/'best_checkpoint.pt',state)
        tracking.log_epoch(wb,row,exp['learning_rate'],best,stale)
        pd.DataFrame(history).to_csv(dest/'epochs.csv',index=False)
        atomic_json(dest/'progress.json',{'state':'training','epoch':epoch,'max_epochs':cfg['max_epochs'],
            'validation_loss':row['validation_loss'],'best_validation_loss':best,'stale_epochs':stale,
            'updated_utc':pd.Timestamp.now(tz='UTC').isoformat()})
        print(exp['name'],seed,row,flush=True)
        if stale>=cfg['patience']:break
    atomic_json(dest/'fit_summary.json',{'epochs':len(history),'best_epoch':min(history,key=lambda x:x['validation_loss'])['epoch'],
        'best_validation_loss':best,'stop_reason':'patience' if stale>=cfg['patience'] else 'epoch_cap',
        'training_validation_seconds':sum(x['seconds'] for x in history)})
    model.load_state_dict(torch.load(dest/'best.pt',map_location='cpu',weights_only=True));model.eval()
    def predict(ids):
        result = []
        with torch.no_grad():
            for start in range(0,len(ids),cfg['batch_size']):
                result.append(torch.softmax(model(inputs(ids[start:start+cfg['batch_size']])),1)[:,1].cpu().numpy())
        result = np.concatenate(result)
        if not np.isfinite(result).all():raise RuntimeError('Invalid scores')
        return result
    probs = predict(comparison)
    loaded = ChartCNN(model_config,ImageConfig(window=window,width=3*window)).to(device)
    loaded.load_state_dict(torch.load(dest/'best.pt',map_location='cpu',weights_only=True));loaded.eval()
    with torch.no_grad():
        check = torch.softmax(loaded(inputs(comparison[:256])),1)[:,1].cpu().numpy()
    np.testing.assert_allclose(check,probs[:256],rtol=1e-5,atol=1e-6)
    result = sample.iloc[comparison][['security','date','sample_id']].copy()
    result['forward_return'] = returns[comparison];result['score'] = probs
    result.to_parquet(dest/'scores.parquet',index=False)
    atomic_json(dest/'progress.json',{'state':'complete','epochs':len(history),'checkpoint_reload_verified':True})
    del model, loaded, opt
    if device.type=='mps':torch.mps.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.run_dir).resolve()
    if args.resume:
        old=json.loads((out/'config.json').read_text())
        if {k:v for k,v in old.items() if k!='max_epochs'}!={k:v for k,v in cfg.items() if k!='max_epochs'} or cfg['max_epochs']<old['max_epochs']:
            raise ValueError('Resume only permits increasing max_epochs')
    else:out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'config.json',cfg)
    files = [Path(__file__),ROOT/'pattern/research/development.py',ROOT/'pattern/models/cnn.py',ROOT/'pattern/models/blocks.py',ROOT/'pattern/config.py',ROOT/'pattern/research/checkpoint.py',ROOT/'pattern/research/tracking.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if args.resume and json.loads((out/'source_hashes.json').read_text())!=hashes:raise ValueError('Resume source differs')
    atomic_json(out/'source_hashes.json',hashes)
    try:
        device = torch.device(cfg['device'])
        if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent CPU fallback')
        torch.set_num_threads(cfg['threads'])
        atomic_json(out/'status.json',{'state':'preparing'})
        if args.resume:
            sample=pd.read_parquet(out/'sample.parquet');images=np.load(out/'images.npy',mmap_mode='r')
            ref=json.loads((ROOT/cfg['dataset_pointer']).read_text());calendar=pd.DatetimeIndex(pd.read_parquet(Path(ref['path'])/'candidate_sessions.parquet').date)
            window=images.shape[-1]//3;c=cfg['comparison']
            schedules={e['horizon']:cohort_schedule(calendar,c['start'],c['end'],e['horizon']) for e in cfg['experiments']}
        else:sample,images,calendar,schedules,window = prepare(cfg,out)
        if cfg.get('reference_sample'):
            pd.testing.assert_frame_equal(sample,pd.read_parquet(ROOT/cfg['reference_sample']))
            atomic_json(out/'sample_equivalence.json',{'reference':cfg['reference_sample'],'identical':True,'rows':len(sample)})
        results = []
        for exp in cfg['experiments']:
            for seed in cfg['seeds']:
                atomic_json(out/'status.json',{'state':'training','experiment':exp['name'],'seed':seed})
                dest = out/exp['name']/f'seed_{seed}'
                scores = train_and_score(cfg,exp,seed,sample,images,window,dest,device,schedules[exp['horizon']])
                metrics,tables = evaluate(scores,calendar,exp['horizon'],schedules[exp['horizon']],cfg['ntiles'])
                atomic_json(dest/'metrics.json',metrics)
                tracking.finish_metrics(cfg,dest,metrics,tables)
                for name,table in tables.items():table.to_parquet(dest/f'{name}.parquet',index=True)
                results.append({'experiment':exp['name'],'seed':seed,**metrics})
                pd.DataFrame(results).to_csv(out/'comparison.csv',index=False)
                print('COMPLETE',exp['name'],seed,json.dumps(metrics),flush=True)
        atomic_json(out/'status.json',{'state':'complete','fits':len(results),'purpose':cfg.get('purpose','development comparison')})
    except BaseException as exc:
        if cfg.get('wandb',{}).get('enabled'):
            import wandb
            if wandb.run is not None:wandb.finish(exit_code=1)
        atomic_json(out/'status.json',{'state':'failed','error':repr(exc)})
        raise

if __name__=='__main__':main()
