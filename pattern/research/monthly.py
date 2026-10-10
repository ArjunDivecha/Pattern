"""Monthly close-derived bars, daily MA200, and 12-calendar-month charts.

The monthly price geometry uses stored adjusted CLOSE, never daily intraday
highs/lows or TRI-adjusted synthetic closes. Forward labels use the frozen TRI.
"""
import numpy as np
import pandas as pd
from pattern.imaging.renderer import _bresenham
from pattern.research.development import evaluate


def month_calendar(sessions):
    dates=pd.Series(pd.DatetimeIndex(sessions))
    return pd.DatetimeIndex(dates.groupby(dates.dt.to_period('M')).max().to_numpy())


def aggregate_stock(g,sessions,ma_days=200):
    cal=pd.DatetimeIndex(sessions)
    g=g.sort_values('date').copy()
    g=g.loc[g.date.isin(cal)].reset_index(drop=True)
    if not len(g):return pd.DataFrame()
    if g.date.duplicated().any():raise ValueError('Duplicate stock dates')
    pos=cal.get_indexer(g.date)
    close=g.close.to_numpy(float);vol=g.volume.to_numpy(float)
    good=np.isfinite(close)&(close>0)
    segment=g.calendar_segment.to_numpy(dtype='float64',na_value=np.nan)
    ma=pd.Series(np.where(good,close,np.nan)).rolling(ma_days,min_periods=ma_days).mean().to_numpy(copy=True)
    starts=np.arange(len(g))-ma_days+1;ok=starts>=0
    end=np.flatnonzero(ok);start=starts[ok]
    ok[end]&=(pos[end]-pos[start]==ma_days-1)&(segment[end]==segment[start])
    ma[~ok]=np.nan
    ma_start=np.full(len(g),np.datetime64('NaT'),dtype='datetime64[ns]')
    end=np.flatnonzero(ok&np.isfinite(ma));ma_start[end]=g.date.to_numpy()[starts[end]]
    g['ma200']=ma;g['ma_start']=ma_start;g['month']=g.date.dt.to_period('M')
    expected=pd.DataFrame({'date':cal,'month':cal.to_period('M')}).groupby('month').date.agg(['min','max','size'])
    g['price_volume_good']=np.isfinite(g.close)&g.close.gt(0)&np.isfinite(g.volume)&g.volume.ge(0)
    g['segment_known']=g.calendar_segment.notna()
    m=g.groupby('month',sort=True).agg(
        first_observed=('date','min'),last_observed=('date','max'),daily_count=('date','size'),
        open=('close','first'),high=('close','max'),low=('close','min'),close=('close','last'),volume=('volume','sum'),
        ma200=('ma200','last'),ma_start=('ma_start','last'),tri_close=('total_return_index_gross_dividends','last'),
        segment=('calendar_segment','last'),segment_count=('calendar_segment','nunique'),
        segment_known=('segment_known','all'),all_bars_good=('price_volume_good','all'),
        endpoint_ok=('return_endpoint_ok','last'),return_month_ok=('return_endpoint_ok','all'))
    # groupby.last skips NaNs: use literal final daily observations for MA and endpoints.
    last=g.drop_duplicates('month',keep='last').set_index('month')
    m[['ma200','ma_start','tri_close','close']]=last[['ma200','ma_start','total_return_index_gross_dividends','close']].rename(columns={'total_return_index_gross_dividends':'tri_close'})
    m=m.join(expected.rename(columns={'min':'first_session','max':'date','size':'expected_daily_count'}))
    complete=m.daily_count.eq(m.expected_daily_count)&m.first_observed.eq(m.first_session)&m.last_observed.eq(m.date)
    same=m.segment_known&m.segment_count.eq(1)
    m['bar_ok']=complete&same&m.all_bars_good;m['calendar_complete']=complete
    m['endpoint_ok']=complete&m.endpoint_ok
    m['return_month_ok']=complete&same&m.return_month_ok
    m['security']=g.security.iloc[0];m['month_number']=m.index.asi8;m['month']=m.index.astype(str)
    return m.reset_index(drop=True)


def candidate_windows(monthly,lookback=12):
    result=[]
    for security,g in monthly.groupby('security',sort=True):
        g=g.sort_values('month_number').copy().reset_index(drop=True)
        good=g.bar_ok & np.isfinite(g.ma200) & g.ma_start.notna()
        eligible=(good.astype(int).rolling(lookback,min_periods=lookback).sum()==lookback)
        eligible&=g.month_number-g.month_number.shift(lookback-1)==lookback-1
        eligible&=g.segment==g.segment.shift(lookback-1)
        g['input_start']=g.ma_start.shift(lookback-1)
        following=g.shift(-1)
        known=(following.month_number-g.month_number==1)&following.return_month_ok.eq(True)&g.endpoint_ok&following.segment.eq(g.segment)
        ret=following.tri_close/g.tri_close-1
        g['forward_return_1']=ret.where(known & np.isfinite(ret))
        g['target_date']=(g.date.dt.to_period('M')+1).map(lambda p:p.end_time.normalize())
        # Actual trading endpoints are overwritten from the monthly calendar by the caller.
        g['momentum_12_1']=g.close.shift(1)/g.close.shift(12)-1
        g['price_to_ma200']=g.close/g.ma200-1
        g['window_end']=np.arange(len(g))
        result.append(g.loc[eligible])
    if not result:raise ValueError('No monthly candidates')
    return pd.concat(result,ignore_index=True)


def sample_months(candidates,start,end,total=None,stocks=None,seed=71):
    eligible=candidates[candidates.date.between(start,end)]
    groups=list(eligible.groupby('date',sort=True))
    expected=pd.period_range(pd.Timestamp(start).to_period('M'),pd.Timestamp(end).to_period('M'),freq='M')
    if not pd.PeriodIndex([d.to_period('M') for d,_ in groups]).equals(expected):raise ValueError('Missing sample month')
    rng=np.random.default_rng(seed);rows=[]
    for i,(date,g) in enumerate(groups):
        n=stocks if total is None else total//len(groups)+int(i<total%len(groups))
        if len(g)<n:raise ValueError(f'{date}: need {n} input-eligible stocks, have {len(g)}')
        g=g.sort_values('security');rows.append(g.iloc[rng.choice(len(g),n,replace=False)])
    return pd.concat(rows).sort_values(['date','security']).reset_index(drop=True)


def verify_monthly_splits(sample):
    for left,right in [('train','validation'),('validation','comparison')]:
        a=sample[sample.partition.eq(left)];b=sample[sample.partition.eq(right)]
        if a.target_date.max()>=b.input_start.min():raise ValueError(f'Information interval overlap {left}/{right}')
    if sample.target_date.max()>=pd.Timestamp('2023-01-01'):raise ValueError('Reserved label accessed')


def render_monthly(bars):
    """12 monthly bars, precomputed MA200; OHLC and MA share the vertical scale."""
    a=np.asarray(bars,dtype=float)
    if a.shape!=(12,6) or not np.isfinite(a).all():raise ValueError('Expected 12 finite [O,H,L,C,V,MA200] rows')
    if (a[:,[0,1,2,3,5]]<=0).any() or (a[:,4]<0).any():raise ValueError('Invalid price or volume')
    price=a[:,[0,1,2,3,5]];lo=price.min();hi=price.max()
    if hi==lo:hi=lo+1e-8
    img=np.zeros((1,64,36),dtype='uint8')
    def row(p):return int(round((hi-p)/(hi-lo)*50))
    vmax=max(a[:,4].max(),1.)
    prev=None
    for t,(o,h,l,c,v,ma) in enumerate(a):
        x=t*3;img[0,row(h):row(l)+1,x+1]=255
        img[0,row(o),x]=255;img[0,row(c),x+2]=255
        r=row(ma);img[0,r,x+1]=255
        if prev is not None:
            for y,z in _bresenham(prev,x-2,r,x+1):img[0,y,z]=255
        prev=r;vh=int(round(v/vmax*12))
        if vh:img[0,64-vh:,x+1]=255
    return img


def monthly_metrics(scores,calendar,dates,ntiles=10):
    calendar=pd.DatetimeIndex(calendar)
    if not (np.diff(calendar.to_period('M').asi8)==1).all():raise ValueError('Incomplete monthly calendar')
    metrics,tables=evaluate(scores,calendar,1,dates,ntiles)
    factor=np.sqrt(12/252)
    for k in ['observed_cohort_sharpe','strict_cohort_sharpe']:
        if metrics[k] is not None:metrics[k]*=factor
        tables['by_year'][k]=pd.to_numeric(tables['by_year'][k],errors='coerce')*factor
    metrics.update(horizon_unit='calendar_month',annualization_periods=12,
        basis='Gross next-calendar-month TRI, $1 long/$1 short, zero cash rate, no costs. Observed metrics reweight known outcomes after ranking. Strict Sharpe requires complete extreme-leg outcomes. sqrt(12) annualization.')
    return metrics,tables
