import numpy as np
import pandas as pd
import pytest
from pattern.research.development import sample_partition, verify_splits, cohort_schedule, evaluate


def fixture_scores():
    cal = pd.bdate_range('2020-01-01',periods=31)
    dates = cohort_schedule(cal,cal[0],cal[-1],5)
    rows = []
    for k,date in enumerate(dates):
        for stock in range(20):
            rows.append({'date':date,'security':f'S{stock:02d}','score':float(stock),
                         'forward_return':stock*.001*(k+1)})
    return cal,dates,pd.DataFrame(rows)


def test_sampling_does_not_select_on_future_returns():
    dates = pd.bdate_range('2000-01-01',periods=100)
    index = pd.DataFrame([{'date':d,'security':str(s),'row_id':i*10+s,'forward_return_5':0.01} for i,d in enumerate(dates) for s in range(10)])
    a = sample_partition(index,dates[0],dates[-1],5,71,count=10)
    index['forward_return_5'] = np.nan
    b = sample_partition(index,dates[0],dates[-1],5,71,count=10)
    assert a.row_id.tolist()==b.row_id.tolist()
    assert len(a)==50 and a.date.nunique()==10
    assert a.date.min()<dates[10] and a.date.max()>=dates[-10]


def test_split_protection():
    parts = {k:pd.DataFrame({'session_position':v,'date':pd.to_datetime(['2000-01-01','2000-02-01'])}) for k,v in [('train',[0,100]),('validation',[400,500]),('comparison',[800,900])]}
    verify_splits(parts,239,60)
    parts['validation'].loc[0,'session_position']=398
    with pytest.raises(ValueError,match='Overlapping'):verify_splits(parts,239,60)


def test_known_metrics_and_regular_schedule():
    cal,dates,scores = fixture_scores()
    m,t = evaluate(scores,cal,5,dates)
    expected = pd.Series([.018*(k+1) for k in range(len(dates))])
    assert m['rank_ic']==pytest.approx(1)
    assert m['spread']==pytest.approx(expected.mean())
    assert m['strict_cohort_sharpe']==pytest.approx(expected.mean()/expected.std()*np.sqrt(252/5))
    assert t['tile_profile'].mean_return.is_monotonic_increasing
    assert (t['cohorts'].end_date.iloc[:-1].to_numpy()==dates[1:].to_numpy()).all()


def test_missing_outcome_keeps_assignment_and_discloses_bias():
    cal,dates,scores = fixture_scores()
    scores.loc[(scores.date==dates[0])&(scores.security=='S19'),'forward_return']=np.nan
    m,t = evaluate(scores,cal,5,dates)
    assert m['strict_cohort_sharpe'] is None
    assert m['observed_cohort_sharpe'] is not None
    assert m['unknown_outcomes']==1
    assert len(t['assignments'])==len(scores)
    assert t['assignments'].query("security=='S19'").tile.eq(10).all()
    assert t['tile_profile'].loc[10,'unknown']==1


def test_gaps_and_overlaps_cannot_produce_sharpe():
    cal,dates,scores=fixture_scores()
    with pytest.raises(ValueError,match='Missing'):
        evaluate(scores[scores.date!=dates[2]],cal,5,dates)
    with pytest.raises(ValueError,match='regularly spaced'):
        evaluate(scores[scores.date!=dates[2]],cal,5,dates.delete(2))
    with pytest.raises(ValueError,match='regularly spaced'):
        evaluate(scores,cal,10,dates)


def test_ties_and_duplicate_guard():
    cal,dates,scores=fixture_scores();scores['score']=1.
    m,t=evaluate(scores,cal,5,dates)
    assert np.isnan(m['rank_ic'])
    with pytest.raises(ValueError,match='Duplicate'):
        evaluate(pd.concat([scores,scores.iloc[:1]]),cal,5,dates)


def test_normalization_stays_float32_with_numpy_scalars():
    from research.development_lab.run import normalize, batch_x
    images=np.full((2,1,64,60),255,dtype='uint8')
    x=batch_x(images,np.array([0,1]),False)
    assert (x[:,:,52:,:]==0).all() and (x[:,:,:51,:]==255).all()
    normalized=normalize(x,np.float64(28.5),np.float64(80.3))
    assert normalized.dtype==np.float32
    assert np.isfinite(normalized).all()


@pytest.mark.parametrize('device_name',['cpu','mps'])
def test_epoch_resume_matches_uninterrupted_training(tmp_path,device_name):
    import shutil
    import torch
    from research.development_lab.run import train_and_score
    if device_name=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS requires host execution')
    torch.set_num_threads(2)
    rng=np.random.default_rng(71)
    images=(rng.random((32,1,64,60))>.8).astype('uint8')*255
    sample=pd.DataFrame({'security':[f'S{i}' for i in range(32)],'date':pd.Timestamp('2020-01-02'),
        'partition':['train']*16+['validation']*8+['comparison']*8,'sample_id':np.arange(32),
        'forward_return_5':np.tile([-.01,.02],16)})
    cfg={'batch_size':8,'max_epochs':3,'patience':10}
    exp={'name':'test','horizon':5,'volume':True,'learning_rate':1e-4,'model':{'channels':[2,4,8]}}
    full=tmp_path/'full';split=tmp_path/'split';full.mkdir();split.mkdir()
    sample.to_parquet(full/'sample.parquet',index=False);shutil.copyfile(full/'sample.parquet',split/'sample.parquet')
    schedule=pd.DatetimeIndex(['2020-01-02']);device=torch.device(device_name)
    train_and_score(cfg,exp,0,sample,images,20,full/'test/seed_0',device,schedule)
    train_and_score({**cfg,'max_epochs':2},exp,0,sample,images,20,split/'test/seed_0',device,schedule)
    train_and_score(cfg,exp,0,sample,images,20,split/'test/seed_0',device,schedule)
    a=torch.load(full/'test/seed_0/last.pt',weights_only=True)
    b=torch.load(split/'test/seed_0/last.pt',weights_only=True)
    assert a['epoch']==b['epoch']==3 and a['stale']==b['stale']
    assert a['optimizer']['state'] and a['rng']['cpu'].dtype==torch.uint8
    for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
    for k in a['optimizer']['state']:
        for field in a['optimizer']['state'][k]:
            torch.testing.assert_close(a['optimizer']['state'][k][field],b['optimizer']['state'][k][field],rtol=0,atol=0)
    from pattern.research.checkpoint import restore_epoch
    model=torch.nn.Linear(1,1);optimizer=torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError,match='mismatch'):
        restore_epoch(split/'test/seed_0/last.pt',model,optimizer,'different-data',device)
