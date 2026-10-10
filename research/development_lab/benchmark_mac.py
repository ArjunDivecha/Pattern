"""Hardware benchmark on fixed, time-stratified real chart observations.
No research performance claims; does not restart any study or select models on returns.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import argparse,contextlib,hashlib,json,gc,time,traceback
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from pattern.config import ModelConfig,ImageConfig
from pattern.models.cnn import ChartCNN


def sample_rows(index,seed=71):
    eligible=index.loc[index.date.dt.year.between(1997,2022)&index.forward_return_5.notna()]
    dates=pd.DatetimeIndex(eligible.date.unique()).sort_values()
    rng=np.random.default_rng(seed);groups={q:dates[dates.to_period('Q')==q] for q in dates.to_period('Q').unique()}
    extras=set(rng.choice(list(groups),200-len(groups),replace=False))
    selected=[]
    for quarter,available in groups.items():selected.extend(rng.choice(available,1+int(quarter in extras),replace=False))
    selected=pd.DatetimeIndex(selected)
    rows=[]
    for date,g in eligible[eligible.date.isin(selected)].groupby('date',sort=True):
        if len(g)<500:raise ValueError('Not enough stocks on sampled date')
        rows.append(g.iloc[rng.choice(len(g),500,replace=False)])
    result=pd.concat(rows).sort_values(['date','security']).reset_index(drop=True)
    assert len(result)==100000 and result.date.nunique()==200
    return result


def model(seed=41):
    torch.manual_seed(seed)
    return ChartCNN(ModelConfig(),ImageConfig()).to('mps')


class Data:
    def __init__(self,images,rows):
        self.disk=images;self.ids=rows.row_id.to_numpy();self.raw=np.array(images[self.ids],copy=True)
        self.labels=(rows.forward_return_5.to_numpy()>0).astype('int64')
        self.mean=float(self.raw.mean());self.std=float(self.raw.std())
        self.normalized=(self.raw.astype('float32')-self.mean)/self.std
        self.u8=torch.from_numpy(self.raw).to('mps');self.f32=torch.from_numpy(self.normalized).to('mps');self.y=torch.from_numpy(self.labels).to('mps')
    def batch(self,kind,ids):
        if kind=='disk':
            x=torch.from_numpy((self.disk[self.ids[ids]].astype('float32')-self.mean)/self.std).to('mps')
            y=torch.from_numpy(self.labels[ids]).to('mps')
        elif kind=='ram':
            x=torch.from_numpy(np.ascontiguousarray(self.normalized[ids])).to('mps');y=torch.from_numpy(self.labels[ids]).to('mps')
        else:
            take=torch.from_numpy(np.asarray(ids,dtype='int64')).to('mps');y=self.y[take]
            x=self.f32[take] if kind=='gpu_f32' else (self.u8[take].float()-self.mean)/self.std
        return x,y


def run_case(data,case,count=16384,repeats=2):
    torch.set_num_threads(case['threads']);torch.mps.empty_cache();m=model();opt=torch.optim.Adam(m.parameters(),lr=1e-5)
    batch=case['batch'];kind=case['storage'];mode=case['sync'];precision=case.get('precision','float32')
    dtype={'bfloat16':torch.bfloat16,'float16':torch.float16}.get(precision)
    # Finite checks are preserved; optimized mode synchronizes them at block boundaries.
    order=np.random.default_rng(11).permutation(len(data.ids))
    def execute(ids):
        total=torch.zeros((),device='mps');finite=torch.ones((),dtype=torch.bool,device='mps');correct=torch.zeros((),device='mps');seen=0
        for step,start in enumerate(range(0,len(ids),batch)):
            chosen=ids[start:start+batch];x,y=data.batch(kind,chosen)
            ctx=torch.autocast('mps',dtype=dtype) if dtype is not None else contextlib.nullcontext()
            with ctx:logits=m(x);loss=torch.nn.functional.cross_entropy(logits,y)
            if mode=='per_batch':
                if not torch.isfinite(loss).item() or not torch.isfinite(logits).all().item():raise RuntimeError('Non-finite batch')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if mode=='per_batch':
                total+=float(loss.item())*len(y);correct+=int((logits.argmax(1)==y).sum().item())
            else:
                finite=finite&torch.isfinite(loss)&torch.isfinite(logits).all()
                total+=loss.detach().float()*len(y);correct+=(logits.detach().argmax(1)==y).sum()
                if (step+1)%32==0:
                    if not finite.item():raise RuntimeError('Non-finite block')
            seen+=len(y)
        if not finite.item() or not torch.isfinite(total).item():raise RuntimeError('Non-finite accumulated loss')
        return float(total.item()/seen)
    execute(order[:batch*5]);torch.mps.synchronize()
    timings=[];losses=[]
    for repeat in range(repeats):
        torch.manual_seed(100+repeat);torch.mps.synchronize();start=time.perf_counter()
        loss=execute(order[:count]);torch.mps.synchronize();elapsed=time.perf_counter()-start
        timings.append(elapsed);losses.append(loss)
    m.eval();x,y=data.batch(kind,order[:256])
    with torch.no_grad():pred=m(x);assert torch.isfinite(pred).all().item()
    memory={'mps_allocated_gb':torch.mps.current_allocated_memory()/1e9,'mps_driver_gb':torch.mps.driver_allocated_memory()/1e9}
    del m,opt;gc.collect();torch.mps.empty_cache()
    return {**case,'observations_per_repeat':count,'seconds':timings,'median_images_per_second':count/float(np.median(timings)),
            'estimated_100k_epoch_seconds':100000*float(np.median(timings))/count,'losses':losses,'finite':True,**memory}


def parity(data):
    ids=np.arange(256);base_x,base_y=data.batch('ram',ids);errors={}
    reference=model().eval()
    with torch.no_grad():ref=reference(base_x).cpu()
    for kind in ['disk','ram','gpu_u8','gpu_f32']:
        x,y=data.batch(kind,ids);diff=float((x-base_x).abs().max().item())
        with torch.no_grad():logit_diff=float((reference(x).cpu()-ref).abs().max())
        assert torch.equal(y,base_y) and diff<1e-6 and logit_diff<1e-5
        errors[kind]={'max_input_error':diff,'max_logit_error':logit_diff}
    # Identical FP32 updates under two data paths (same seed, batch, dropout stream).
    states=[]
    for kind in ['ram','gpu_u8']:
        m=model();opt=torch.optim.Adam(m.parameters(),lr=1e-5);torch.manual_seed(101)
        for k in range(3):
            x,y=data.batch(kind,np.arange(k*128,(k+1)*128));opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward();opt.step()
        states.append({k:v.detach().cpu() for k,v in m.state_dict().items()})
        del m,opt
    max_error=max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0] if states[0][k].is_floating_point())
    # Different MPS dispatch paths can introduce small floating-point differences.
    assert max_error<1e-4, max_error
    errors['three_step_weight_max_error']=max_error
    del reference;gc.collect();torch.mps.empty_cache();return errors


def main():
    if not torch.backends.mps.is_available():raise RuntimeError('Run outside sandbox with MPS available')
    old=ROOT/'runs/full_I20_R5_R20_v1_20260918/RUNNING.lock'
    if old.exists():raise RuntimeError('Old run must remain stopped')
    out=ROOT/'runs'/('mac_development_benchmark_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
    cache=ROOT/'cache/frozen_full_I20_M20_v1';state=json.loads((cache/'state.json').read_text());assert state['state']=='complete'
    index=pd.read_parquet(cache/'index.parquet',columns=['security','date','row_id','forward_return_5']);rows=sample_rows(index);del index
    rows.to_parquet(out/'sample.parquet',index=False)
    rows.groupby(rows.date.dt.year).agg(stocks=('security','nunique'),dates=('date','nunique'),observations=('row_id','size')).to_csv(out/'sample_by_year.csv')
    images=np.memmap(cache/'images.bin',dtype='uint8',mode='r',shape=(state['rows'],1,64,60));data=Data(images,rows)
    (out/'metadata.json').write_text(json.dumps({'purpose':'hardware benchmark, not model selection or alpha evaluation','sample_seed':71,'source_cache':str(cache),'rows':len(rows),'dates':rows.date.nunique(),'stock_count':rows.security.nunique(),'years':[int(rows.date.dt.year.min()),int(rows.date.dt.year.max())],'raw_sample_mb':data.raw.nbytes/1e6,'float_sample_mb':data.normalized.nbytes/1e6,'normalization_mean':data.mean,'normalization_std':data.std,'torch':torch.__version__,'python':sys.version,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    (out/'parity.json').write_text(json.dumps(parity(data),indent=2));print('PARITY PASS',out,flush=True)
    cases=[{'name':'disk_sync_256','storage':'disk','sync':'per_batch','batch':256,'threads':4},
           {'name':'ram_sync_256','storage':'ram','sync':'per_batch','batch':256,'threads':4},
           {'name':'ram_block_256','storage':'ram','sync':'block','batch':256,'threads':4}]
    for storage in ['gpu_u8','gpu_f32']:
        for batch in [128,256,512,1024]:cases.append({'name':f'{storage}_{batch}','storage':storage,'sync':'block','batch':batch,'threads':4})
    results=[]
    for case in cases:
        try:r=run_case(data,case)
        except Exception as e:r={**case,'error':repr(e),'traceback':traceback.format_exc()};torch.mps.empty_cache()
        results.append(r);(out/'screening.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)
    valid=[r for r in results if r.get('finite')];best=max(valid,key=lambda r:r['median_images_per_second'])
    base={k:best[k] for k in ['name','storage','sync','batch','threads']}
    for threads in [2,8]:
        c={**base,'threads':threads,'name':base['name']+f'_threads{threads}'};r=run_case(data,c);results.append(r);print(json.dumps(r),flush=True)
    for precision in ['bfloat16','float16']:
        c={**base,'name':base['name']+'_'+precision,'precision':precision}
        try:r=run_case(data,c)
        except Exception as e:r={**c,'error':repr(e)};torch.mps.empty_cache()
        results.append(r);print(json.dumps(r),flush=True)
    (out/'screening.json').write_text(json.dumps(results,indent=2))
    # Select only FP32 here: AMP numerical quality needs a separate learning comparison.
    best=max([r for r in results if r.get('finite') and r.get('precision','float32')=='float32'],key=lambda r:r['median_images_per_second'])
    final=[]
    for c in [cases[0],{k:best[k] for k in ['name','storage','sync','batch','threads']}]:
        r=run_case(data,c,count=100000,repeats=2);final.append(r);print('FULL_EPOCH',json.dumps(r),flush=True)
    (out/'full_epochs.json').write_text(json.dumps(final,indent=2))
    recommendation={'storage':best['storage'],'batch':best['batch'],'threads':best['threads'],'sync':best['sync'],'precision':'float32',
        'measured_full_epoch_speedup':final[1]['median_images_per_second']/final[0]['median_images_per_second'],'full_epoch_seconds':float(np.median(final[1]['seconds'])),
        'scope':'I20/MA20 full CNN, 100k training observations; validation and preprocessing excluded. Batch size changes optimization dynamics.'}
    (out/'recommendation.json').write_text(json.dumps(recommendation,indent=2))
    assert not old.exists();print('OUTPUT',out,flush=True)

if __name__=='__main__':main()
