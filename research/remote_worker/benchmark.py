import time,json
import numpy as np,pandas as pd,torch
from pattern.config import ModelConfig,ImageConfig
from pattern.models.cnn import ChartCNN
assert torch.backends.mps.is_available(),'MPS unavailable'
torch.set_num_threads(3);torch.manual_seed(303)
s=pd.read_parquet('sample.parquet');ret=s.forward_return_5.to_numpy();ids=np.flatnonzero(np.isfinite(ret));images=np.load('images.npy',mmap_mode='r');fraction=np.count_nonzero(images[ids])/images[ids].size;mean=np.float32(255*fraction);std=np.float32(255*np.sqrt(fraction*(1-fraction)))
m=ChartCNN(ModelConfig(),ImageConfig(window=20,width=60)).to('mps');o=torch.optim.Adam(m.parameters(),lr=1e-5)
rows=[]
for epoch in range(1,4):
 torch.manual_seed(303+epoch*10000);order=ids.copy();np.random.default_rng(303+epoch).shuffle(order);t=time.monotonic();total=0
 for n in range(0,len(order),256):
  k=order[n:n+256];x=torch.from_numpy((images[k].astype('float32')-mean)/std).to('mps');y=torch.from_numpy((ret[k]>0).astype('int64')).to('mps');loss=torch.nn.functional.cross_entropy(m(x),y);assert torch.isfinite(loss);o.zero_grad(set_to_none=True);loss.backward();o.step();total+=loss.item()*len(k)
 torch.mps.synchronize();row={'epoch':epoch,'seconds':time.monotonic()-t,'loss':total/len(ids),'training_rows':len(ids)};rows.append(row);print(row,flush=True)
open('benchmark.json','w').write(json.dumps({'device':'mps','torch':torch.__version__,'epochs':rows},indent=2))
