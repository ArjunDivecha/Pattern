import numpy as np
import pandas as pd
import torch
from pattern.research.full_train import fit,score,balanced_ids


def test_full_cnn_disk_batches_checkpoint_resume(tmp_path):
    torch.set_num_threads(2)
    rng=np.random.default_rng(7);shape=(64,1,64,60)
    mm=np.memmap(tmp_path/'images.bin',dtype='uint8',mode='w+',shape=shape)
    mm[:]=(rng.random(shape)>.8)*255;mm.flush()
    index=pd.DataFrame({'white_pixels':np.count_nonzero(mm,axis=(1,2,3))})
    returns=np.where(np.arange(64)%2,.02,-.01);train=np.arange(48);val=np.arange(48,64)
    args=(mm,index,returns,train,val)
    fit(*args,tmp_path/'resume',0,torch.device('cpu'),batch_size=16,max_epochs=1)
    resumed,mean,std=fit(*args,tmp_path/'resume',0,torch.device('cpu'),batch_size=16,max_epochs=2)
    direct,_,_=fit(*args,tmp_path/'direct',0,torch.device('cpu'),batch_size=16,max_epochs=2)
    a=score(resumed,mm,val,mean,std,torch.device('cpu'),16)
    b=score(direct,mm,val,mean,std,torch.device('cpu'),16)
    assert np.isfinite(a).all() and np.allclose(a,b,atol=1e-6)
    assert pd.read_csv(tmp_path/'resume/epochs.csv').epoch.tolist()==[1,2]
    selected=balanced_ids(train,returns,0)
    assert (returns[selected]>0).sum()==(returns[selected]<=0).sum()
