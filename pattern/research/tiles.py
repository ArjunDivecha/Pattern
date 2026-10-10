"""Rank first, retain unknown outcomes, then measure arithmetic n-tile separation."""
import numpy as np
import pandas as pd


def tile_report(scores,n=10):
    rows=[]; assignments=[]
    for date,g in scores.groupby('date',sort=True):
        g=g.sort_values(['score','security']).copy()
        if len(g)<n:continue
        g['tile']=(np.arange(len(g))*n//len(g)+1).astype(int)
        assignments.append(g)
        for tile,t in g.groupby('tile'):
            complete=t.forward_return.notna().all()
            rows.append(dict(date=date,tile=tile,stocks=len(t),unknown=int(t.forward_return.isna().sum()),
                forward_return=float(t.forward_return.mean()) if complete else np.nan))
    if not rows:raise ValueError('Insufficient stocks for n-tiles')
    daily=pd.DataFrame(rows)
    summary=daily.groupby('tile').agg(mean_forward_return=('forward_return','mean'),dates=('date','size'),resolved_dates=('forward_return','count'),average_stocks=('stocks','mean'),unknown_outcomes=('unknown','sum'))
    return daily,summary,pd.concat(assignments,ignore_index=True)


def diagnostics(scores,sessions,horizon,cost_bps=10):
    daily,summary,assigned=tile_report(scores)
    pivot=daily.pivot(index='date',columns='tile',values='forward_return')
    spreads=pd.DataFrame({'bottom':pivot[1],'top':pivot[10]});spreads['spread']=spreads.top-spreads.bottom
    ics=[]
    for date,g in scores.groupby('date'):
        known=g.dropna(subset=['forward_return'])
        ic=known.score.corr(known.forward_return,method='spearman') if len(known)>2 and known.score.nunique()>1 and known.forward_return.nunique()>1 else np.nan
        ics.append({'date':date,'rank_ic':ic,'known':len(known),'total':len(g)})
    ic=pd.DataFrame(ics)
    calendar=pd.DatetimeIndex(sessions)
    positions=calendar.get_indexer(spreads.index)
    keep=(positions-positions.min())%horizon==0
    cohorts=spreads.loc[keep].copy()
    cohorts['net_spread_cost_sensitivity']=cohorts.spread-4*cost_bps/10000
    complete=cohorts.spread.notna().all()
    def sharpe(x):
        return float(x.mean()/x.std(ddof=1)*np.sqrt(252/horizon)) if complete and len(x)>1 and x.std(ddof=1)>0 else None
    metrics={'mean_daily_rank_ic':float(ic.rank_ic.mean()),'scored_stock_dates':len(scores),
             'unknown_outcomes':int(scores.forward_return.isna().sum()),
             'mean_top_forward_return':float(spreads.top.mean()),'mean_bottom_forward_return':float(spreads.bottom.mean()),
             'mean_top_minus_bottom':float(spreads.spread.mean()),'resolved_spread_dates':int(spreads.spread.notna().sum()),
             'total_signal_dates':len(spreads),'nonoverlapping_cohorts':len(cohorts),
             'cohort_gross_sharpe':sharpe(cohorts.spread),'cohort_net_cost_sensitivity_sharpe':sharpe(cohorts.net_spread_cost_sensitivity),
             'one_way_cost_bps':cost_bps,'cost_per_cohort_spread':4*cost_bps/10000,
             'portfolio_limit':'Close-to-close $1 long/$1 short cohorts; no borrow, execution lag or NAV accounting. Pilot only.'}
    return daily,summary,assigned,spreads,ic,cohorts,metrics
