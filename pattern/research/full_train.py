"""Disk-backed full-CNN fitting with epoch checkpoints and progress heartbeats."""
from pathlib import Path
import math,time
import numpy as np
import pandas as pd
import torch
from pattern.config import ModelConfig,ImageConfig
from pattern.models.cnn import ChartCNN
from pattern.research.full_cache import atomic_json


def save_atomic(path,state):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


def batches(images,ids,labels,mean,std,batch_size,shuffle_seed=None):
    order=ids.copy()
    if shuffle_seed is not None:np.random.default_rng(shuffle_seed).shuffle(order)
    for start in range(0,len(order),batch_size):
        take=order[start:start+batch_size]
        x=(images[take].astype('float32')-mean)/std
        yield torch.from_numpy(x),torch.from_numpy(labels[take].astype('int64'))


def balanced_ids(train_ids,returns,seed):
    groups=[train_ids[returns[train_ids]>0],train_ids[returns[train_ids]<=0]]
    n=min(map(len,groups))
    if n==0:raise ValueError('Both training classes required')
    rng=np.random.default_rng(seed);ids=np.concatenate([rng.choice(g,n,replace=False) for g in groups]);rng.shuffle(ids)
    return ids


def fit(images,index,returns,train_ids,val_ids,dest,seed,device,batch_size=256,max_epochs=30,patience=3,lr=1e-5):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    selected=balanced_ids(train_ids,returns,seed)
    fraction=float(index.white_pixels.to_numpy()[selected].mean()/(64*60))
    mean=255*fraction;std=255*math.sqrt(fraction*(1-fraction))
    if not std>0:raise ValueError('Zero pixel variance')
    labels=np.nan_to_num(returns,nan=0)>0
    torch.manual_seed(seed);model=ChartCNN(ModelConfig(),ImageConfig()).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    next_epoch=1;best=float('inf');stale=0;history=[]
    if (dest/'last.pt').exists():
        state=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        next_epoch=state['epoch']+1;best=state['best'];stale=state['stale'];history=state['history']
    atomic_json(dest/'normalization.json',{'mean':mean,'std':std,'train_selected':len(selected),'train_available':len(train_ids),'validation':len(val_ids),'seed':seed})
    for epoch in range(next_epoch,max_epochs+1):
        if stale>=patience:break
        torch.manual_seed(seed+10000*epoch);started=time.monotonic();record={'epoch':epoch}
        for phase,ids in [('train',selected),('validation',val_ids)]:
            model.train(phase=='train');loss_sum=0.;correct=0;seen=0
            with torch.set_grad_enabled(phase=='train'):
                for step,(x,y) in enumerate(batches(images,ids,labels,mean,std,batch_size,seed+epoch if phase=='train' else None)):
                    x=x.to(device);y=y.to(device);logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y)
                    if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():raise RuntimeError('Non-finite full-data batch')
                    if phase=='train':optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                    loss_sum+=float(loss.item())*len(y);correct+=int((logits.argmax(1)==y).sum().item());seen+=len(y)
                    if step%50==0:
                        atomic_json(dest/'progress.json',{'state':'running','phase':phase,'epoch':epoch,'processed':seen,'total':len(ids),'loss':loss_sum/seen,'elapsed_seconds':time.monotonic()-started,'updated_utc':pd.Timestamp.now(tz='UTC').isoformat()})
            if seen==0:raise ValueError('Empty fit partition')
            record[phase+'_loss']=loss_sum/seen;record[phase+'_accuracy']=correct/seen
        value=record['validation_loss'];history.append(record)
        if value<best:
            best=value;stale=0;save_atomic(dest/'best.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:stale+=1
        save_atomic(dest/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'stale':stale,'history':history})
        pd.DataFrame(history).to_csv(dest/'epochs.csv',index=False)
        print('EPOCH',str(dest),record,flush=True)
    model.load_state_dict(torch.load(dest/'best.pt',map_location='cpu',weights_only=True));model.to(device).eval()
    atomic_json(dest/'progress.json',{'state':'fit_complete','epochs':len(history),'best_validation_loss':best,'updated_utc':pd.Timestamp.now(tz='UTC').isoformat()})
    return model,mean,std


def score(model,images,ids,mean,std,device,batch_size=256):
    model.eval();out=np.empty(len(ids),dtype='float32');labels=np.zeros(len(images),dtype='int8');offset=0
    with torch.no_grad():
        for x,y in batches(images,ids,labels,mean,std,batch_size):
            probs=torch.softmax(model(x.to(device)),1)[:,1].cpu().numpy()
            if not np.isfinite(probs).all():raise RuntimeError('Non-finite probabilities')
            out[offset:offset+len(probs)]=probs;offset+=len(probs)
    return out
