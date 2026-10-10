"""Meaningful boundary checks for the quality sidecar; no network access."""
import pandas as pd
import pytest
from build_quality_layer import quality_layer, window_ok


def panel():
    return pd.DataFrame({'security':['A']*5,'date':pd.date_range('2020-01-01',periods=5),
                         'open':[10.]*5,'high':[12.]*5,'low':[9.]*5,'close':[11.]*5,
                         'volume':[100.]*5,'total_return_index_gross_dividends':[20.,21.,22.,23.,24.]})


def test_invalid_bar_blocks_both_adjacent_returns_and_windows():
    p=panel();p.loc[2,'close']=100
    q=quality_layer(p)
    assert q.loc[2,'ohlc_inconsistent']
    assert q.loc[[2,3],'screened_total_return'].isna().all()
    assert not window_ok(q,3).any()


def test_long_gap_boundary_not_bridged_but_new_segment_can_resume():
    p=panel();p.loc[3:,'date']=pd.date_range('2022-01-01',periods=2)
    q=quality_layer(p)
    assert q.segment_id.tolist()==[0,0,0,1,1]
    assert pd.isna(q.loc[3,'screened_total_return'])
    assert pd.notna(q.loc[4,'screened_total_return'])
    assert not window_ok(q,3).iloc[4]
    assert not window_ok(q,2,forward=True).iloc[1]


def test_extreme_positive_move_not_clipped():
    p=panel();p.loc[0,['open','high','low','close','total_return_index_gross_dividends']]=.000001
    q=quality_layer(p)
    assert q.loc[1,'screened_total_return']>1000
    assert q.loc[1,'extreme_return_gt_100pct']


def test_zero_not_terminal_loss_assumption():
    p=panel();p.loc[2,'total_return_index_gross_dividends']=0
    q=quality_layer(p)
    assert q.loc[[2,3],'screened_total_return'].isna().all()


def test_no_return_between_securities():
    p=panel();p.loc[3:,'security']='B'
    q=quality_layer(p)
    assert pd.isna(q.loc[3,'screened_total_return'])


def test_duplicate_keys_fail_closed():
    p=panel()
    with pytest.raises(ValueError,match='Duplicate'):
        quality_layer(pd.concat([p,p.iloc[:1]]))


def test_zero_volume_informational_missing_volume_blocks_renderer():
    p=panel();p.loc[1,'volume']=0;p.loc[2,'volume']=float('nan')
    q=quality_layer(p)
    assert q.loc[1,'image_bar_ok'] and q.loc[1,'volume_zero']
    assert not q.loc[2,'current_renderer_bar_ok']


def test_forward_window_endpoints_and_end_of_series():
    q=quality_layer(panel())
    assert window_ok(q,2,forward=True).tolist()==[True,True,True,False,False]
    assert window_ok(q,3).tolist()==[False,False,True,True,True]
