"""Common calendar split schedule with strict full-interval separation."""
import numpy as np
import pandas as pd


def annual_schedule(sessions,train_start='1996-01-01',first_test=2010,last_test=2022,
                    reserve_start='2023-01-01',max_history=239,max_horizon=60,validation_sessions=252):
    cal=pd.DatetimeIndex(sessions)
    if not cal.is_monotonic_increasing or cal.has_duplicates:raise ValueError('Calendar must be sorted and unique')
    if min(max_history,max_horizon,validation_sessions)<1:raise ValueError('Lengths must be positive')
    if last_test>=pd.Timestamp(reserve_start).year:raise ValueError('Reserved years cannot be development test years')
    first=cal.searchsorted(pd.Timestamp(train_start))+max_history-1
    reserve=cal.searchsorted(pd.Timestamp(reserve_start))
    if reserve>=len(cal):raise ValueError('Calendar must include reserved boundary')
    gap=max_history-1+max_horizon
    result=[]
    for year in range(first_test,last_test+1):
        test=np.flatnonzero((cal.year==year)&(np.arange(len(cal))+max_horizon<reserve))
        if not len(test):raise ValueError('No test sessions')
        val_stop=test[0]-gap
        val_start=val_stop-validation_sessions
        train_stop=val_start-gap
        if train_stop<=first:raise ValueError('Insufficient training history for requested purge')
        result.append({'year':year,'train':cal[first:train_stop],'validation':cal[val_start:val_stop],'test':cal[test]})
    return result


def describe(schedule):
    return pd.DataFrame([{'test_year':fold['year'],**{f'{part}_{metric}':value for part in ['train','validation','test'] for metric,value in [('start',str(fold[part][0].date())),('end',str(fold[part][-1].date())),('sessions',len(fold[part]))]}} for fold in schedule])
