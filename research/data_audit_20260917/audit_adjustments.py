"""Local-only empirical adjustment audit; no source edits or network calls.

Price/total-return identities are tested against stored corporate actions.
Volume continuity is descriptive, not an independent proof of share units.
"""
from pathlib import Path
from datetime import datetime, timezone
import sys, json
import numpy as np
import pandas as pd
from build_quality_layer import hash_file

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart')
from gdm import dataset


def split_class(implied, factor, tolerance=.005):
    a=np.abs(np.log(implied))
    b=np.abs(np.log(implied/factor))
    return np.select([ (a<=tolerance)&(b>tolerance), (b<=tolerance)&(a>tolerance)],
                     ['split_adjusted_relative_to_TRI','unadjusted_relative_to_TRI'],default='unresolved')


def cash_errors(close, prev, tri_ratio, cash, future_factor):
    tr=tri_ratio-1
    return ((tr-((close+cash)/prev-1))*10000,
            (tr-((close+cash/future_factor)/prev-1))*10000,
            (tr-(close/prev-1))*10000)


def main():
    paths={k:dataset(k,acknowledge='survivorship-incomplete') for k in [
        'equities.research_vintage_prices','equities.events','equities.research_vintage_security_master']}
    files=[paths['equities.research_vintage_prices'],paths['equities.events']/'splits.parquet',paths['equities.events']/'dividends.parquet',paths['equities.research_vintage_security_master']]
    hashes={str(f):hash_file(f) for f in files}
    p=pd.read_parquet(files[0]).sort_values(['security','date']).reset_index(drop=True)
    s=pd.read_parquet(files[1]);d=pd.read_parquet(files[2]);master=pd.read_parquet(files[3])
    for frame in [s,d]:
        frame['date']=pd.to_datetime(frame['Ex-Date'],errors='coerce')
        frame['amount']=pd.to_numeric(frame['Dividend Amount'],errors='coerce')
    # Never interpret split ratios or stock distributions as cash dividends.
    split_all=s[s['Dividend Type'].eq('Stock Split')&s.amount.gt(0)&np.isfinite(s.amount)].copy()
    split_all=split_all.drop_duplicates(['security','date','amount'])
    conflicts=split_all.duplicated(['security','date'],keep=False)
    split_conflicts=split_all[conflicts].copy()
    split_all=split_all[~conflicts].copy()
    group=p.groupby('security',sort=False)
    for field in ['date','close','total_return_index_gross_dividends']:
        p['prev_'+field]=group[field].shift()
    p['gap_days']=(p.date-p.prev_date).dt.days
    vals=p[['open','high','low','close']]
    p['bar_valid']=np.isfinite(vals).all(axis=1)&vals.gt(0).all(axis=1)&p.high.ge(p.low)&p.open.between(p.low,p.high)&p.close.between(p.low,p.high)
    p['pair_valid']=p.bar_valid&group.bar_valid.shift(fill_value=False)&p.total_return_index_gross_dividends.gt(0)&p.prev_total_return_index_gross_dividends.gt(0)&p.gap_days.between(1,7)
    p['tri_ratio']=p.total_return_index_gross_dividends/p.prev_total_return_index_gross_dividends
    columns=['security','date','open','high','low','close','volume','retrieved_at','prev_date','prev_close','gap_days','tri_ratio','pair_valid']
    print('Prices loaded, computing event joins',flush=True)
    sp=split_all[split_all.date.le('2026-06-30')].merge(p[columns],on=['security','date'],how='left',validate='one_to_one',indicator=True)
    sp['factor']=sp.amount
    # Exclude other same-day distribution types from simple split identity.
    other=d[~d['Dividend Type'].eq('Stock Split')][['security','date']].drop_duplicates().assign(other_action=True)
    sp=sp.merge(other,on=['security','date'],how='left',validate='one_to_one')
    sp['testable']=sp.pair_valid.eq(True)&sp.factor.apply(lambda x:abs(np.log(x))>=.1)&~sp.other_action.eq(True)&np.isfinite(sp.tri_ratio)
    sp['implied_share_multiplier']=sp.tri_ratio/(sp.close/sp.prev_close)
    sp['classification']='not_testable'
    sp.loc[sp.testable,'classification']=split_class(sp.loc[sp.testable,'implied_share_multiplier'],sp.loc[sp.testable,'factor'])
    # Cash hypotheses: quoted amount versus amount normalized for later splits.
    cash=d[d['Dividend Type'].isin(['Regular Cash','Special Cash'])&d.amount.gt(0)&np.isfinite(d.amount)&d.date.le('2026-06-30')].copy()
    # Equal amounts on one ex-date can be distinct distributions. Never dedup
    # by amount: retain full records and flag multiple same-type events.
    cash=cash.drop_duplicates()
    ambiguous=cash.groupby(['security','date','Dividend Type']).size().rename('n').reset_index()
    ambiguous=ambiguous[ambiguous.n.gt(1)][['security','date']].drop_duplicates().assign(ambiguous_cash_events=True)
    cash=cash.groupby(['security','date'],as_index=False).agg(cash=('amount','sum'),types=('Dividend Type',lambda x:' + '.join(sorted(set(x)))),event_rows=('amount','size'))
    cash=cash.merge(ambiguous,on=['security','date'],how='left',validate='one_to_one')
    noncash=d[~d['Dividend Type'].isin(['Regular Cash','Special Cash'])][['security','date']].drop_duplicates().assign(other_action=True)
    cash=cash.merge(noncash,on=['security','date'],how='left',validate='one_to_one')
    cash=cash.merge(p[columns],on=['security','date'],how='left',validate='one_to_one',indicator=True)
    cash['future_split_factor']=1.
    cash['future_factor_conflict']=False
    split_groups={key:frame.sort_values('date') for key,frame in split_all.groupby('security')}
    conflict_groups={key:frame for key,frame in split_conflicts.groupby('security')}
    for sec,rows in cash.groupby('security',sort=False):
        events=split_groups.get(sec)
        if events is not None:
            dates=events.date.to_numpy(dtype='datetime64[ns]');logs=np.r_[0.,np.cumsum(np.log(events.amount.to_numpy()))]
            left=np.searchsorted(dates,rows.date.to_numpy(dtype='datetime64[ns]'),side='right')
            right=np.searchsorted(dates,rows.retrieved_at.to_numpy(dtype='datetime64[ns]'),side='right')
            right=np.maximum(left,right)
            cash.loc[rows.index,'future_split_factor']=np.exp(logs[right]-logs[left])
        if sec in conflict_groups:
            for date in conflict_groups[sec].date.unique():
                mask=rows.date.lt(date)&rows.retrieved_at.ge(date)
                cash.loc[rows.index[mask],'future_factor_conflict']=True
    cash['raw_cash_error_bp'],cash['split_normalized_cash_error_bp'],cash['no_cash_error_bp']=cash_errors(cash.close,cash.prev_close,cash.tri_ratio,cash.cash,cash.future_split_factor)
    cash['raw_cash_yield_bp']=cash.cash/cash.prev_close*10000
    cash['normalized_cash_yield_bp']=cash.raw_cash_yield_bp/cash.future_split_factor
    cash['testable']=cash.pair_valid.eq(True)&~cash.other_action.eq(True)&~cash.ambiguous_cash_events.eq(True)&~cash.future_factor_conflict&np.isfinite(cash.tri_ratio)
    cash['raw_match']=cash.raw_cash_error_bp.abs().le(1)&cash.testable
    cash['normalized_match']=cash.split_normalized_cash_error_bp.abs().le(1)&cash.testable
    cash['cash_match']=cash.raw_match|cash.normalized_match
    cash['material_cash']=cash.normalized_cash_yield_bp.ge(5)&cash.raw_cash_yield_bp.ge(5)
    cash['no_cash_match']=cash.no_cash_error_bp.abs().le(1)&cash.testable
    cash['classification']=np.select([~cash.testable,cash.cash_match&~cash.no_cash_match,cash.no_cash_match&~cash.cash_match,cash.cash_match&cash.no_cash_match],['not_testable','cash_identity_supported','no_cash_difference_detected','indistinguishable_at_1bp'],default='unresolved')
    # Split-volume test: medians of 10 positive-volume observations each side,
    # excluding event day. Natural activity changes confound individual events.
    volume=[]
    stock_indices=group.indices
    for _,r in sp[sp.testable & (sp.factor.ge(2)|sp.factor.le(.5))].iterrows():
        indices=stock_indices[r.security];stock=p.iloc[indices]
        pos=stock.date.searchsorted(r.date);pre=stock.iloc[max(0,pos-10):pos];post=stock.iloc[pos+1:pos+11]
        valid=len(pre)==10 and len(post)==10 and pre.volume.gt(0).all() and post.volume.gt(0).all() and pre.gap_days.dropna().le(7).all() and post.gap_days.dropna().le(7).all()
        rec={'security':r.security,'date':r.date,'factor':r.factor,'testable':bool(valid),'price_classification':r.classification}
        if valid:
            ratio=post.volume.median()/pre.volume.median(); rec.update(volume_ratio=ratio,log_ratio=float(np.log(ratio)),log_split=float(np.log(r.factor)),distance_adjusted=float(abs(np.log(ratio))),distance_unadjusted=float(abs(np.log(ratio/r.factor))))
        volume.append(rec)
    vol=pd.DataFrame(volume)
    out=ROOT/('adjustment_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
    sp.to_csv(out/'split_events.csv',index=False);cash.to_csv(out/'cash_events.csv',index=False);vol.to_csv(out/'volume_events.csv',index=False);split_conflicts.to_csv(out/'split_conflicts.csv',index=False)
    # Security metadata is current reference information, not an identity audit.
    cash.merge(master[['security','name','crncy','metadata_is_point_in_time']],on='security',how='left').loc[lambda f:f.testable&~f.cash_match].sort_values('raw_cash_error_bp',key=lambda x:x.abs(),ascending=False).head(100).to_csv(out/'cash_exceptions_top100.csv',index=False)
    yearly=cash[cash.testable].groupby(cash.date.dt.year).agg(events=('security','size'),securities=('security','nunique'),cash_matches=('cash_match','sum'),no_cash_matches=('no_cash_match','sum'))
    yearly.to_csv(out/'cash_by_year.csv')
    vv=vol[vol.testable].copy()
    summary={'price_rows':len(p),'price_securities':p.security.nunique(),'split_event_rows_stored':len(s),'dividend_event_rows_stored':len(d),
             'splits_by_class':sp.classification.value_counts().to_dict(),'cash_by_class':cash.classification.value_counts().to_dict(),
             'cash_testable_securities':cash.loc[cash.testable,'security'].nunique(),'cash_raw_matches':int(cash.raw_match.sum()),'cash_normalized_matches':int(cash.normalized_match.sum()),
             'cash_normalization_only_matches':int((cash.normalized_match&~cash.raw_match).sum()),'material_cash_events':int((cash.testable&cash.material_cash).sum()),
             'material_cash_class':cash.loc[cash.testable&cash.material_cash,'classification'].value_counts().to_dict(),
             'cash_by_type':cash[cash.testable].groupby('types').agg(n=('security','size'),matches=('cash_match','sum')).to_dict('index'),
             'ambiguous_cash_event_dates':int(cash.ambiguous_cash_events.eq(True).sum()),
             'split_conflicting_rows':len(split_conflicts),'volume_testable_events':len(vv),
             'volume_forward':{'n':int(vv.factor.ge(2).sum()),'median_ratio':float(vv.loc[vv.factor.ge(2),'volume_ratio'].median())},
             'volume_reverse':{'n':int(vv.factor.le(.5).sum()),'median_ratio':float(vv.loc[vv.factor.le(.5),'volume_ratio'].median())},
             'volume_closer_to_adjusted':int(vv.distance_adjusted.lt(vv.distance_unadjusted).sum()),
             'thresholds':{'cash_error_bp':1,'material_cash_yield_bp':5,'split_log_error':.005,'max_calendar_gap_days':7},
             'no_external_calls':True,'source_hashes':hashes}
    assert all(hash_file(Path(f))==h for f,h in hashes.items()),'Source changed'
    summary['source_hashes_unchanged']=True
    assert len(sp)==sum(summary['splits_by_class'].values())
    assert len(cash)==sum(summary['cash_by_class'].values())
    assert not sp.duplicated(['security','date']).any() and not cash.duplicated(['security','date']).any()
    assert set(cash.types.unique())<=set(['Regular Cash','Special Cash','Regular Cash + Special Cash'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)
    print('OUTPUT',out,flush=True)


if __name__=='__main__':main()
