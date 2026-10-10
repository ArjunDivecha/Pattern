import numpy as np
import torch
from research.multiclass_lab.train import boundaries,labels,ChartCNN,BinaryCNN
from pattern.config import ModelConfig,ImageConfig

def test_quantiles_and_order():
    x=np.arange(100,dtype=float)
    e=boundaries(x)
    np.testing.assert_array_equal(np.bincount(labels(x,e)),[20]*5)
    np.testing.assert_array_equal(labels(np.array([-1.,1000.]),e),[0,4])

def test_same_backbone_five_logits_and_gradient():
    torch.manual_seed(101);a=BinaryCNN(ModelConfig(),ImageConfig(window=20,width=60))
    torch.manual_seed(101);b=ChartCNN(ModelConfig(),ImageConfig(window=20,width=60))
    for k,v in a.conv_blocks.state_dict().items():torch.testing.assert_close(v,b.conv_blocks.state_dict()[k],rtol=0,atol=0)
    out=b(torch.zeros(2,1,64,60));assert out.shape==(2,5)
    loss=torch.nn.functional.cross_entropy(out,torch.tensor([0,4]));loss.backward()
    assert torch.isfinite(loss) and b.fc.weight.grad is not None
    score=torch.softmax(out,1)@torch.arange(1,6,dtype=out.dtype)
    assert ((score>=1)&(score<=5)).all()
