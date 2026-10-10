"""Calendar target, missing outcome, and n-tile accounting regression tests."""
import numpy as np
import pandas as pd
import pytest
from pattern.data.frozen import samples
from pattern.research.tiles import tile_report,diagnostics
from pattern.imaging.renderer import render_window


def panel():
    dates=pd.bdate_range('2020-01-01',periods=400)
    p=pd.DataFrame({'security':'A','date':dates,'open':10.,'high':11.,'low':9.,'close':10.,'volume':100.,
        'total_return_index_gross_dividends':100*np.power(1.01,np.arange(400)),
        'current_renderer_bar_ok':True,'return_endpoint_ok':True,'calendar_segment':0})
    return p,dates


@pytest.mark.parametrize('lookback,horizon,ma',[(5,10,3),(20,5,10),(60,1,5),(120,60,120)])
def test_independent_lengths_and_arithmetic_return(lookback,horizon,ma):
    p,d=panel();x,i=samples(p,d,[d[250]],lookback,horizon,ma)
    assert x.shape==(1,1,64,3*lookback)
    assert i.input_start.iloc[0]==d[250-lookback-ma+2]
    assert i.target_date.iloc[0]==d[250+horizon]
    assert i.forward_return.iloc[0]==pytest.approx(1.01**horizon-1)


def test_missing_future_preserves_input_and_target():
    p,d=panel();p=p[p.date!=d[73]]
    x,i=samples(p,d,[d[70]],5,10,3)
    assert len(x)==1 and pd.isna(i.forward_return.iloc[0])
    assert i.target_date.iloc[0]==d[80]


def test_missing_past_never_compresses():
    p,d=panel();p=p[p.date!=d[68]]
    with pytest.raises(ValueError,match='No valid'):samples(p,d,[d[70]],5,10,3)


def test_future_changes_do_not_change_image():
    p,d=panel();a,i=samples(p,d,[d[70]],5,10,3)
    p.loc[p.date>d[70],'total_return_index_gross_dividends']*=2
    b,j=samples(p,d,[d[70]],5,10,3)
    assert np.array_equal(a,b) and i.forward_return.iloc[0]!=j.forward_return.iloc[0]


def test_unknown_winner_not_removed_before_ranking():
    s=pd.DataFrame({'security':[str(i) for i in range(20)],'date':pd.Timestamp('2020-01-01'),'score':np.arange(20),'forward_return':np.arange(20)/100})
    s.loc[19,'forward_return']=np.nan
    daily,summary,assigned=tile_report(s)
    assert assigned.loc[assigned.security=='19','tile'].iloc[0]==10
    assert assigned.loc[assigned.tile==10].shape[0]==2
    assert pd.isna(daily.loc[daily.tile==10,'forward_return'].iloc[0])
    assert daily.stocks.sum()==20


def test_nonoverlap_and_simple_spread():
    d=pd.bdate_range('2020-01-01',periods=10)
    frames=[]
    for k,date in enumerate(d):
        frames.append(pd.DataFrame({'security':[str(i) for i in range(20)],'date':date,'score':np.arange(20),'forward_return':np.arange(20)/100+k/1000}))
    _,_,_,s,ic,c,m=diagnostics(pd.concat(frames),d,5)
    assert len(c)==2
    assert np.allclose(s.spread,.18)
    assert np.allclose(c.net_spread_cost_sensitivity,.176)
    assert np.allclose(ic.rank_ic,1)


def test_ma_default_is_backward_compatible():
    p,d=panel();a=p[['open','high','low','close','volume']].to_numpy();tri=p.total_return_index_gross_dividends.to_numpy()
    assert np.array_equal(render_window(a,tri),render_window(a,tri,ma_length=20))
