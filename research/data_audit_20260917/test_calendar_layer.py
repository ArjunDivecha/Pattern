import pandas as pd
from build_calendar_layer import calendar_flags


def fixture():
    sessions = pd.bdate_range('2020-01-01',periods=65)
    q = pd.DataFrame({'security':'A','date':sessions,'unresolved_identity_boundary':False,
                      'current_renderer_bar_ok':True,'return_endpoint_ok':True})
    return q,sessions


def test_fixed_horizon_and_edges():
    q,s = fixture(); r = calendar_flags(q,s)
    assert r.loc[0,'target_20_session_date']==s[20]
    assert r.past_20_sessions_complete.sum()==46
    assert r.past_39_sessions_complete.sum()==27
    assert r.forward_20_sessions_complete.sum()==45
    assert r.target_beyond_calendar.sum()==20


def test_missing_session_not_compressed_or_filled():
    q,s = fixture(); q=q.drop(index=10); r=calendar_flags(q,s)
    assert len(r)==64
    assert r.loc[10,'missing_sessions_before']==1
    assert not r.loc[0,'forward_20_sessions_complete']
    assert r.loc[0,'target_20_session_date']==s[20]
    assert not r.loc[28,'past_20_sessions_complete']
    assert r.loc[29,'past_20_sessions_complete']


def test_invalid_price_blocks_but_does_not_remove():
    q,s=fixture(); q.loc[10,['current_renderer_bar_ok','return_endpoint_ok']]=False
    r=calendar_flags(q,s)
    assert len(r)==65 and not r.loc[0,'forward_20_sessions_complete']
    assert not r.loc[29,'past_20_sessions_complete']
    assert r.loc[30,'past_20_sessions_complete']


def test_identity_boundary_and_securities():
    q,s=fixture(); q.loc[10,'unresolved_identity_boundary']=True
    other=q.assign(security='B',unresolved_identity_boundary=False)
    r=calendar_flags(pd.concat([q,other]),s)
    assert not r.loc[0,'forward_20_sessions_complete']
    assert r.loc[65,'forward_20_sessions_complete']


def test_off_calendar_retained_and_blocks():
    q,s=fixture(); q.loc[3,'date']=pd.Timestamp('2020-01-04')
    r=calendar_flags(q,s)
    assert len(r)==65 and r.off_candidate_calendar.sum()==1
    assert not r.loc[r.off_candidate_calendar,'past_20_sessions_complete'].any()


def test_short_history_cannot_make_window():
    q,s=fixture(); r=calendar_flags(q.iloc[:3],s)
    assert not r.past_20_sessions_complete.any()
    assert not r.past_39_sessions_complete.any()
    assert not r.forward_20_sessions_complete.any()
