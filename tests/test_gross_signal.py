import numpy as np
import pandas as pd
import pytest
from pattern.research.gross_signal import gross_diagnostics,hac_mean_t


def scores():
    dates=pd.bdate_range('2010-01-01',periods=40)
    return pd.concat([pd.DataFrame({'date':d,'security':[f'S{i}' for i in range(20)],'score':np.arange(20),'forward_return':np.arange(20)*.01}) for d in dates],ignore_index=True),dates


def test_paired_legs_and_contribution_reconcile():
    s,d=scores();m,t=gross_diagnostics(s,d)
    assert m['top_minus_bottom']==pytest.approx(.18)
    assert m['top_return_paired']-m['bottom_return_paired']==pytest.approx(m['top_minus_bottom'])
    assert t['stock_contributions'].mean_spread_contribution.sum()==pytest.approx(.18)
    assert not any('cost' in k or 'net_' in k for k in m)
    assert t['seen_vs_extension'].paired_dates.tolist()==[30,10]


def test_unknown_extreme_blocks_cohort_not_ranking():
    s,d=scores();s.loc[(s.date==d[0])&(s.security=='S19'),'forward_return']=np.nan
    m,t=gross_diagnostics(s,d)
    assert m['paired_extreme_dates']==39 and m['unresolved_cohorts']==1
    assert m['gross_cohort_sharpe'] is None
    assert len(t['assignments'])==800
    assert m['top_return_paired']-m['bottom_return_paired']==pytest.approx(m['top_minus_bottom'])


def test_hac_does_not_compress_missing_dates():
    d=pd.bdate_range('2020-01-01',periods=80)
    s=pd.Series(np.random.default_rng(2).normal(size=80),index=d);s.iloc[30:35]=np.nan
    assert np.isfinite(hac_mean_t(s,d,5))
    assert hac_mean_t(s.iloc[:10],d,5) is None
