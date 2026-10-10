import numpy as np
from audit_adjustments import split_class,cash_errors

def test_split_adjusted_and_unadjusted_disambiguation():
    assert split_class(np.array([1.,4.,1.,.1,2.]),np.array([4.,4.,.1,.1,4.])).tolist()==['split_adjusted_relative_to_TRI','unadjusted_relative_to_TRI','split_adjusted_relative_to_TRI','unadjusted_relative_to_TRI','unresolved']

def test_cash_dividend_identity():
    raw,norm,no=cash_errors(99.,100.,1.,1.,1.)
    assert abs(raw)<1e-8 and abs(norm)<1e-8 and abs(no-100)<1e-8

def test_future_split_normalizes_cash():
    raw,norm,no=cash_errors(24.75,25.,1.,1.,4.)
    assert abs(norm)<1e-8 and abs(raw)>100 and abs(no-100)<1e-8

def test_cash_already_adjusted_price_differs_from_price_only():
    raw,norm,no=cash_errors(100.,100.,1.,1.,1.)
    assert no==0 and abs(raw)>99 and abs(norm)>99

def test_special_distribution_backward_scale_identity():
    from review_adjustments import predicted_scale_after
    assert abs(predicted_scale_after(.98,2.,98.)-1.)<1e-12
    assert np.isnan(predicted_scale_after(1.,200.,100.))
