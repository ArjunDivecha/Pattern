import pandas as pd
import pytest
from pattern.research.walkforward import annual_schedule


def calendar():return pd.bdate_range('1990-01-01','2026-06-30')


def test_all_grid_intervals_disjoint_and_expanding():
    cal=calendar();folds=annual_schedule(cal)
    assert len(folds)==13
    for f in folds:
        for lookback in [5,10,20,40,60,120]:
            for horizon in [1,5,10,20,40,60]:
                history=2*lookback-1
                for a,b in [('train','validation'),('validation','test')]:
                    assert cal.get_loc(f[a][-1])+horizon<cal.get_loc(f[b][0])-history+1
        assert len(f['validation'])==252
        assert (f['test'].year==f['year']).all()
    assert all(a['train'][0]==b['train'][0] and len(a['train'])<len(b['train']) for a,b in zip(folds,folds[1:]))


def test_no_development_label_enters_reserve():
    cal=calendar();folds=annual_schedule(cal)
    for f in folds:
        assert cal[cal.get_loc(f['test'][-1])+60]<pd.Timestamp('2023-01-01')
    assert folds[-1]['test'][-1].month<=10


def test_reserve_and_short_history_fail_loudly():
    with pytest.raises(ValueError,match='Reserved'):annual_schedule(calendar(),last_test=2023)
    with pytest.raises(ValueError,match='Insufficient'):annual_schedule(calendar(),train_start='2009-01-01')
