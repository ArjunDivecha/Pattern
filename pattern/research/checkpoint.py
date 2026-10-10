"""Atomic epoch-boundary checkpoints for the development trainer."""
from pathlib import Path
import os
import torch


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def save_epoch(path, model, optimizer, epoch, best, stale, history, best_model, signature, device):
    rng={'cpu':torch.get_rng_state()}
    if device.type=='mps':rng['mps']=torch.mps.get_rng_state()
    if device.type=='cuda':rng['cuda']=torch.cuda.get_rng_state_all()
    state={'format_version':1,'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),
           'epoch':epoch,'best':best,'stale':stale,'history':history,'best_model':cpu_tree(best_model),
           'signature':signature,'rng':cpu_tree(rng),
           'sampling_rng':'NumPy default_rng(seed+epoch); torch.manual_seed(seed+epoch*10000) at each epoch boundary'}
    atomic_save(path,state)
    return state


def restore_epoch(path, model, optimizer, signature, device):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('format_version')!=1 or state['signature']!=signature:
        raise ValueError('Checkpoint model, data or training configuration mismatch')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng']['cpu'])
    if device.type=='mps' and 'mps' in state['rng']:torch.mps.set_rng_state(state['rng']['mps'])
    if device.type=='cuda' and 'cuda' in state['rng']:torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state
