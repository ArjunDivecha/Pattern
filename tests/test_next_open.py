import numpy as np
from research.next_open_lab.run import open_return

def test_entry_overnight_is_excluded():
    # Previous close 100, entry open 120, entry close 126; later TRI gain 10%.
    assert np.isclose(open_return(120,126,126,138.6),.155)

def test_entry_day_dividend_is_not_received():
    # Entry-day TRI contains the overnight ex-dividend adjustment; cancel it.
    assert np.isclose(open_return(90,90,100,100),0)

def test_subsequent_total_returns_are_included():
    assert np.isclose(open_return(100,100,200,210),.05)

def test_invalid_open_is_unknown():
    assert np.isnan(open_return(0,100,100,105))
