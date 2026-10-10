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
from pattern.models.cnn import ChartCNN as BinaryCNN
from pattern.research.development import sample_partition, verify_splits, cohort_schedule, evaluate
from pattern.research.full_cache import atomic_json
from pattern.research.checkpoint import atomic_save, save_epoch, restore_epoch, cpu_tree
from pattern.research import tracking


from research.development_lab.run import batch_x, normalize

class ChartCNN(BinaryCNN):
    def __init__(self, model_cfg, image_cfg):
        super().__init__(model_cfg, image_cfg)
        # Preserve identical seed-101 convolutional initialization to binary A.
        self.fc=torch.nn.Linear(self.fc.in_features,5)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

def boundaries(returns):
    edges=np.quantile(returns[np.isfinite(returns)],[.2,.4,.6,.8])
    if not np.isfinite(edges).all() or not (np.diff(edges)>0).all():raise ValueError('Non-distinct class edges')
    return edges

def labels(returns,edges):
    return np.searchsorted(edges,returns,side='right').astype('int64')

def train_and_score(cfg, exp, seed, sample, images, window, dest, device, schedule):
    dest.mkdir(parents=True,exist_ok=True)
    returns = sample[f"forward_return_{exp['horizon']}"].to_numpy()
    train = np.flatnonzero(sample.partition.eq('train') & np.isfinite(returns))
    val = np.flatnonzero(sample.partition.eq('validation') & np.isfinite(returns))
    comparison = np.flatnonzero(sample.partition.eq('comparison') & sample.date.isin(schedule))
    if not len(train) or not len(val) or not len(comparison):
        raise ValueError('Empty partition')
    edges=boundaries(returns[train]);targets=labels(returns,edges)
    atomic_json(dest/'class_definition.json',{'boundaries':edges.tolist(),'source':'finite training returns only','class_counts':np.bincount(targets[train],minlength=5).tolist(),'ranking':'sum probability times ordered class 1..5'})
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
    if wb is not None:wb.config.update({'target':'five training-quantile absolute-return classes','class_boundaries':edges.tolist(),'score':'expected ordered class 1..5'},allow_val_change=True)
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
                    y = torch.from_numpy(targets[chosen]).to(device)
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
                result.append(torch.softmax(model(inputs(ids[start:start+cfg['batch_size']])),1).cpu().numpy())
        result = np.concatenate(result)
        if not np.isfinite(result).all():raise RuntimeError('Invalid scores')
        return result
    probs = predict(comparison)
    loaded = ChartCNN(model_config,ImageConfig(window=window,width=3*window)).to(device)
    loaded.load_state_dict(torch.load(dest/'best.pt',map_location='cpu',weights_only=True));loaded.eval()
    with torch.no_grad():
        check = torch.softmax(loaded(inputs(comparison[:256])),1).cpu().numpy()
    np.testing.assert_allclose(check,probs[:256],rtol=1e-5,atol=1e-6)
    result = sample.iloc[comparison][['security','date','sample_id']].copy()
    result['forward_return'] = returns[comparison];result['score'] = probs @ np.arange(1,6)
    for k in range(5):result[f'prob_class_{k+1}']=probs[:,k]
    result.to_parquet(dest/'scores.parquet',index=False)
    atomic_json(dest/'progress.json',{'state':'complete','epochs':len(history),'checkpoint_reload_verified':True})
    del model, loaded, opt
    if device.type=='mps':torch.mps.empty_cache()
    return result

