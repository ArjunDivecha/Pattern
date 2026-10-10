import numpy as np
import pytest
from pattern.imaging.fast import load_renderer
from pattern.imaging.renderer import render_window

@pytest.mark.parametrize('window,ma',[(5,3),(20,20),(60,20),(120,120)])
def test_compiled_renderer_pixel_exact(tmp_path,window,ma):
    rng=np.random.default_rng(18);n=window+ma+80
    close=np.exp(rng.normal(0,.04,n).cumsum())*10
    op=close*np.exp(rng.normal(0,.01,n));hi=np.maximum(op,close)*1.03;lo=np.minimum(op,close)*.97
    volume=rng.integers(0,100000,n).astype(float);ohlcv=np.column_stack([op,hi,lo,close,volume]);tri=close*3.4
    ends=np.arange(window+ma-2,n,3);got=load_renderer(tmp_path)(ohlcv,tri,ends,window,ma)
    for image,end in zip(got,ends):
        expected=render_window(ohlcv[end-window-ma+2:end+1],tri[end-window-ma+2:end+1],window=window,width=3*window,ma_length=ma)
        assert np.array_equal(image,expected)

def test_flat_and_zero_volume(tmp_path):
    o=np.ones((50,5));o[:,4]=0;t=np.ones(50);g=load_renderer(tmp_path)(o,t,[49])
    assert np.array_equal(g[0],render_window(o,t))
