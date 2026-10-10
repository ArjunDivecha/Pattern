import numpy as np
import pandas as pd
import pytest
from pattern.research.monthly import aggregate_stock,candidate_windows,render_monthly,month_calendar,monthly_metrics,verify_monthly_splits,sample_months


def daily_fixture():
    cal=pd.bdate_range('1995-01-01','1998-12-31')
    g=pd.DataFrame({'security':'A','date':cal,'close':np.arange(len(cal))+10.,'volume':np.arange(len(cal))+1.,
                    'total_return_index_gross_dividends':np.arange(len(cal))+100.,'calendar_segment':1,'return_endpoint_ok':True})
    return cal,g


def test_close_aggregation_and_exact_daily_ma():
    cal,g=daily_fixture();m=aggregate_stock(g,cal)
    last=g[g.date.dt.to_period('M')=='1998-12']
    row=m.iloc[-1]
    assert row.open==last.close.iloc[0]
    assert row.high==last.close.max() and row.low==last.close.min() and row.close==last.close.iloc[-1]
    assert row.volume==last.volume.sum()
    assert row.ma200==pytest.approx(g.close.iloc[-200:].mean())
    assert row.ma_start==g.date.iloc[-200]
    assert not np.isfinite(m.iloc[0].ma200)


def test_future_changes_do_not_change_inputs():
    cal,g=daily_fixture();a=aggregate_stock(g,cal)
    g.loc[g.date>'1997-12-31','close']*=100
    b=aggregate_stock(g,cal)
    pd.testing.assert_frame_equal(a[a.date<='1997-12-31'],b[b.date<='1997-12-31'])


def test_missing_daily_close_and_identity_gap_reset_ma():
    cal,g=daily_fixture();missing=g[(g.date.dt.year==1997)&(g.date.dt.month==12)].index[5]
    g=g.drop(missing)
    m=aggregate_stock(g,cal)
    row=m[m.month.eq('1997-12')].iloc[0]
    assert not row.bar_ok and not row.return_month_ok and np.isnan(row.ma200)
    g.loc[g.date>='1998-07-01','calendar_segment']=2
    m=aggregate_stock(g,cal)
    assert np.isnan(m[m.month.eq('1998-08')].ma200.iloc[0])


def test_unknown_next_month_retained_in_candidates():
    cal,g=daily_fixture()
    g.loc[g.date.eq(pd.Timestamp('1998-12-15')),'return_endpoint_ok']=False
    m=aggregate_stock(g,cal);c=candidate_windows(m)
    row=c[c.month.eq('1998-11')].iloc[0]
    assert np.isnan(row.forward_return_1)
    earlier=c[c.month.eq('1998-10')].iloc[0]
    expected=m[m.month.eq('1998-11')].tri_close.iloc[0]/m[m.month.eq('1998-10')].tri_close.iloc[0]-1
    assert earlier.forward_return_1==pytest.approx(expected)
    assert row.input_start<pd.Timestamp('1997-12-01')


def test_renderer_includes_ma_in_scale():
    bars=np.tile([10.,10.,10.,10.,0.,5.],(12,1))
    img=render_monthly(bars)
    assert img.shape==(1,64,36) and img.dtype==np.uint8
    assert (img[0,0,:]==255).all()
    assert img[0,50,1]==255 and img[0,50,34]==255
    assert img[0,51:,:].sum()==0
    with pytest.raises(ValueError):render_monthly(bars[:11])


def test_monthly_sharpe_uses_twelve_and_nonoverlapping_months():
    dates=pd.date_range('2018-01-31',periods=5,freq='ME');scores=[]
    for k,d in enumerate(dates[:-1]):
        for i in range(20):scores.append({'date':d,'security':str(i),'score':i,'forward_return':i*.001*(k+1)})
    m,t=monthly_metrics(pd.DataFrame(scores),dates,dates[:-1])
    x=np.arange(1,5)*.018
    assert m['observed_cohort_sharpe']==pytest.approx(x.mean()/x.std(ddof=1)*np.sqrt(12))
    assert (t['cohorts'].end_date.to_numpy()==dates[1:].to_numpy()).all()
    assert m['annualization_periods']==12


def test_ma_history_is_included_in_split_guard():
    s=pd.DataFrame({'partition':['train','validation','comparison'],
      'input_start':pd.to_datetime(['1996-01-01','2012-06-01','2016-06-01']),
      'target_date':pd.to_datetime(['2012-01-31','2016-01-31','2022-12-30'])})
    verify_monthly_splits(s)
    s.loc[1,'input_start']=pd.Timestamp('2011-12-01')
    with pytest.raises(ValueError,match='overlap'):verify_monthly_splits(s)
