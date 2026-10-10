import numpy as np
import pandas as pd
from research.clean_universe.build import derive

def fixture():
 n=55
 return pd.DataFrame({'date':pd.bdate_range('2020-01-01',periods=n),'open':np.full(n,10.),'close':np.full(n,10.),'total_return_index_gross_dividends':np.full(n,10.),'session_index':np.arange(n),'calendar_segment':np.zeros(n),'current_renderer_bar_ok':True,'return_endpoint_ok':True})
def test_forward_jump_never_changes_formation():
 g=fixture();g.loc[42:,'total_return_index_gross_dividends']=1000
 d=derive(g);assert d.loc[40,'formation_eligible'];assert d.loc[40,'return_outlier'];assert not d.loc[40,'training_eligible'];assert d.loc[40,'next_open_return_5_raw']==99
 assert not d.loc[42,'formation_eligible']
def test_penny_floor():
 g=fixture();g.loc[40,'close']=.005
 assert not derive(g).loc[40,'formation_eligible']
def test_next_open_timing():
 g=fixture();g.loc[41,'open']=5
 assert derive(g).loc[40,'next_open_return_5_raw']==1
 assert derive(g).tail(5).next_open_return_5_raw.isna().all()
def test_gap_unknown():
 g=fixture();g.loc[43:,'session_index']+=1
 assert not derive(g).loc[40,'label_valid']
def test_large_loss_retained():
 g=fixture();g.loc[42:,'total_return_index_gross_dividends']=1
 d=derive(g);assert d.loc[40,'label_valid'];assert np.isclose(d.loc[40,'next_open_return_5_raw'],-.9)
