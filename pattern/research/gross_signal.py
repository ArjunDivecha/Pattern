"""Gross n-tile diagnostics, matched-date legs, and fixed-weight contribution audit.
No transaction-cost or implementability assumptions. Missing outcomes stay visible.
"""
import numpy as np
import pandas as pd
from pattern.research.tiles import tile_report


def hac_mean_t(series,calendar,lags):
    """Bartlett HAC mean t-stat; calendar gaps retained, not compressed."""
    x=series.reindex(pd.DatetimeIndex(calendar)).astype(float)
    n=int(x.notna().sum())
    if n<30:return None
    residual=(x-x.mean()).to_numpy();v=float(np.nansum(residual**2))
    for lag in range(1,lags+1):
        v+=2*(1-lag/(lags+1))*float(np.nansum(residual[lag:]*residual[:-lag]))
    return float(x.mean()/(np.sqrt(v)/n)) if v>0 else None


def gross_diagnostics(scores,calendar,horizon=5):
    daily,summary,assigned=tile_report(scores,10)
    pivot=daily.pivot(index='date',columns='tile',values='forward_return')
    paired=pivot[[1,10]].rename(columns={1:'bottom',10:'top'}).dropna()
    paired['spread']=paired.top-paired.bottom
    common=pivot.dropna()
    common_curve=common.mean().rename('mean_return_common_dates').to_frame()
    common_curve['common_dates']=len(common)
    ic=[]
    for date,g in scores.groupby('date'):
        known=g.dropna(subset=['forward_return'])
        value=known.score.corr(known.forward_return,method='spearman') if len(known)>2 and known.score.nunique()>1 and known.forward_return.nunique()>1 else np.nan
        ic.append({'date':date,'rank_ic':value,'known_stocks':len(known),'total_stocks':len(g)})
    ic=pd.DataFrame(ic)
    calendar=pd.DatetimeIndex(calendar);pos=calendar.get_indexer(pivot.index)
    if (pos<0).any():raise ValueError('Signal date absent from calendar')
    cohort_dates=pivot.index[(pos-pos.min())%horizon==0]
    cohorts=paired.reindex(cohort_dates)
    # Retain unresolved cohorts; no renormalizing the portfolio history.
    sharpe=None
    if cohorts.spread.notna().all() and len(cohorts)>1 and cohorts.spread.std()>0:
        sharpe=float(cohorts.spread.mean()/cohorts.spread.std()*np.sqrt(252/horizon))
    extreme=assigned[assigned.date.isin(paired.index)&assigned.tile.isin([1,10])].copy()
    extreme['tile_count']=extreme.groupby(['date','tile']).security.transform('size')
    extreme['spread_contribution']=extreme.forward_return/extreme.tile_count*np.where(extreme.tile.eq(10),1,-1)
    check=extreme.groupby('date').spread_contribution.sum().reindex(paired.index)
    if not np.allclose(check,paired.spread):raise AssertionError('Contribution reconciliation failed')
    contribution=extreme.groupby('security').agg(sum_contribution=('spread_contribution','sum'),appearances=('date','size'),largest_abs_return=('forward_return',lambda x:x.abs().max()))
    contribution['mean_spread_contribution']=contribution.sum_contribution/len(paired) if len(paired) else np.nan
    contribution=contribution.sort_values('mean_spread_contribution',ascending=False)
    metrics={'signal_dates':len(pivot),'scored_stock_dates':len(scores),'average_stocks_per_date':float(scores.groupby('date').size().mean()),
             'unknown_outcomes':int(scores.forward_return.isna().sum()),'paired_extreme_dates':len(paired),'complete_all_decile_dates':len(common),
             'top_return_paired':float(paired.top.mean()),'bottom_return_paired':float(paired.bottom.mean()),'top_minus_bottom':float(paired.spread.mean()),
             'median_daily_spread':float(paired.spread.median()),'positive_spread_fraction':float(paired.spread.gt(0).mean()),
             'rank_ic':float(ic.rank_ic.mean()),'spread_hac_t':hac_mean_t(paired.spread,calendar[(calendar>=pivot.index.min())&(calendar<=pivot.index.max())],max(2*horizon,20)),
             'gross_cohort_sharpe':sharpe,'cohort_count':len(cohorts),'unresolved_cohorts':int(cohorts.spread.isna().sum()),
             'hac_lags':max(2*horizon,20),'return_basis':f'{horizon}-session arithmetic TRI return, gross, no annualization of mean spreads',
             'limitations':'Observed outcomes only for descriptive means; missing cohorts block full Sharpe; small trained model and fixed initial-date stock sample.'}
    if len(contribution):
        best=contribution.iloc[0];metrics.update(largest_positive_contributor=contribution.index[0],largest_contributor_mean_spread=float(best.mean_spread_contribution),
          spread_without_largest_contributor_fixed_weights=float(paired.spread.mean()-best.mean_spread_contribution))
    tails=[]
    for n in [5,10,20]:
        a,b,_=tile_report(scores,n);p=a.pivot(index='date',columns='tile',values='forward_return')[[1,n]].dropna()
        tails.append({'tiles':n,'paired_dates':len(p),'bottom':p[1].mean(),'top':p[n].mean(),'spread':(p[n]-p[1]).mean()})
    periods=[]
    # The first 30 sessions were already inspected; remainder is the extension.
    for name,dates in [('previously_seen_first30',pivot.index[:30]),('extension_after_first30',pivot.index[30:])]:
        p=paired.loc[paired.index.isin(dates)]
        periods.append({'period':name,'paired_dates':len(p),'top':p.top.mean(),'bottom':p.bottom.mean(),'spread':p.spread.mean()})
    tables={'deciles_daily':daily,'deciles_summary_available_dates':summary,'deciles_common_dates':common_curve,'assignments':assigned,
            'paired_spreads':paired,'rank_ic':ic,'nonoverlapping_gross_cohorts':cohorts,'stock_contributions':contribution,
            'contribution_details':extreme,'tail_comparisons':pd.DataFrame(tails),'seen_vs_extension':pd.DataFrame(periods)}
    return metrics,tables
