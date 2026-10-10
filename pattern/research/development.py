"""Small-sample development: chronological selection and nonoverlapping metrics.

Model comparison results are development diagnostics, not untouched-test evidence.
Missing returns never become zero and never affect ranking or sample selection.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def stratified_dates(dates, count, seed):
    dates = pd.DatetimeIndex(dates).unique().sort_values()
    if not 0 < count <= len(dates):
        raise ValueError('Invalid date count')
    rng = np.random.default_rng(seed)
    # Equal-sized chronological strata preserve broad representation even in small runs.
    return pd.DatetimeIndex([rng.choice(part) for part in np.array_split(dates.to_numpy(), count)])


def sample_partition(index, start, end, stocks, seed, count=None, dates=None):
    eligible = index.loc[index.date.between(start, end)]
    sizes = eligible.groupby('date').size()
    if dates is None:
        dates = stratified_dates(sizes[sizes >= stocks].index, count, seed)
    else:
        dates = pd.DatetimeIndex(dates)
        if not dates.isin(sizes[sizes >= stocks].index).all():
            raise ValueError('Evaluation schedule lacks sufficient input-eligible stocks')
    rng = np.random.default_rng(seed)
    rows = []
    for date, group in eligible[eligible.date.isin(dates)].groupby('date', sort=True):
        group = group.sort_values('security')
        rows.append(group.iloc[rng.choice(len(group), stocks, replace=False)])
    result = pd.concat(rows).sort_values(['date', 'security']).reset_index(drop=True)
    if result.duplicated(['date', 'security']).any():
        raise ValueError('Duplicate stock-date')
    return result


def verify_splits(parts, max_history, max_horizon):
    for left, right in zip(['train', 'validation'], ['validation', 'comparison']):
        a, b = parts[left], parts[right]
        if a.session_position.max() + max_horizon >= b.session_position.min() - max_history + 1:
            raise ValueError(f'Overlapping information intervals: {left}/{right}')
    if parts['comparison'].date.max() >= pd.Timestamp('2023-01-01'):
        raise ValueError('Reserved period accessed')


def cohort_schedule(calendar, start, end, horizon):
    calendar = pd.DatetimeIndex(calendar)
    first = int(calendar.searchsorted(pd.Timestamp(start)))
    stop = int(calendar.searchsorted(pd.Timestamp(end), side='right')) - horizon
    positions = np.arange(first, stop, horizon)
    if len(positions) < 2:
        raise ValueError('Insufficient complete cohorts')
    return calendar[positions]


def evaluate(scores, calendar, horizon, expected_dates, ntiles=10):
    """Fixed H-session close-to-close cohorts, $1 long and $1 short.

    Sharpe uses H-period P&L increments with sqrt(252/H), zero cash rate.
    This is a coarse holding-period Sharpe, not daily marked-to-market Sharpe.
    Observed-only diagnostics reweight known names AFTER assigning all scores.
    A separate strict Sharpe is withheld if any extreme-leg return is unknown.
    """
    s = scores.copy()
    calendar = pd.DatetimeIndex(calendar)
    expected_dates = pd.DatetimeIndex(expected_dates)
    if s.duplicated(['date', 'security']).any() or not np.isfinite(s.score).all():
        raise ValueError('Duplicate observations or non-finite scores')
    if not pd.DatetimeIndex(s.date.unique()).sort_values().equals(expected_dates.sort_values()):
        raise ValueError('Missing or unexpected cohort dates')
    positions = calendar.get_indexer(expected_dates)
    if (positions < 0).any() or not (np.diff(positions) == horizon).all():
        raise ValueError('Sharpe requires a complete, regularly spaced nonoverlapping schedule')
    if not (positions[-1] + horizon < len(calendar)):
        raise ValueError('Final cohort extends beyond calendar')
    s['forward_return'] = s.forward_return.where(np.isfinite(s.forward_return))
    records, assignments, ic = [], [], []
    for date, g in s.groupby('date', sort=True):
        if len(g) < ntiles * 2:
            raise ValueError('Too few stocks for tiles')
        g = g.sort_values(['score', 'security']).copy()
        g['tile'] = np.arange(len(g)) * ntiles // len(g) + 1
        assignments.append(g)
        known = g.dropna(subset=['forward_return'])
        rank_ic = known.score.corr(known.forward_return, method='spearman') if len(known)>2 and known.score.nunique()>1 and known.forward_return.nunique()>1 else np.nan
        ic.append({'date':date, 'rank_ic':rank_ic, 'known_stocks':len(known), 'total_stocks':len(g), 'unique_scores':g.score.nunique()})
        for tile, t in g.groupby('tile'):
            records.append({'date':date, 'tile':int(tile), 'stocks':len(t), 'unknown':int(t.forward_return.isna().sum()),
                            'observed_return':t.forward_return.mean(),
                            'strict_return':t.forward_return.mean() if t.forward_return.notna().all() else np.nan})
    daily = pd.DataFrame(records)
    ic = pd.DataFrame(ic).set_index('date')
    observed = daily.pivot(index='date', columns='tile', values='observed_return')
    strict = daily.pivot(index='date', columns='tile', values='strict_return')
    cohorts = pd.DataFrame({'bottom':observed[1], 'top':observed[ntiles]})
    cohorts['spread'] = cohorts.top - cohorts.bottom
    cohorts['strict_spread'] = strict[ntiles] - strict[1]
    cohorts['end_date'] = calendar[calendar.get_indexer(cohorts.index) + horizon]
    def sharpe(x):
        if x.isna().any() or len(x)<2 or x.std(ddof=1)<=0:
            return None
        return float(x.mean()/x.std(ddof=1)*np.sqrt(252/horizon))
    def summary(c, i):
        return {'cohorts':len(c), 'top':float(c.top.mean()), 'bottom':float(c.bottom.mean()),
                'spread':float(c.spread.mean()), 'rank_ic':float(i.rank_ic.mean()),
                'observed_cohort_sharpe':sharpe(c.spread), 'strict_cohort_sharpe':sharpe(c.strict_spread),
                'positive_spread_fraction':float(c.spread.gt(0).mean()),
                'unresolved_extreme_cohorts':int(c.strict_spread.isna().sum())}
    metrics = summary(cohorts, ic)
    metrics.update(scored_rows=len(s), stocks=s.security.nunique(), unknown_outcomes=int(s.forward_return.isna().sum()),
                   horizon=horizon, ntiles=ntiles, mean_is_per_horizon=True,
                   basis='Gross H-session close-to-close TRI; fixed $1 long/$1 short; zero cash rate; no costs. Observed metrics condition on available outcomes after ranking; strict Sharpe requires every extreme-leg outcome. Not daily NAV Sharpe.')
    profile = daily.groupby('tile').agg(mean_return=('observed_return','mean'), stocks=('stocks','sum'), unknown=('unknown','sum'))
    profile['unknown_fraction'] = profile.unknown/profile.stocks
    periods = pd.DataFrame([{'year':year, **summary(g, ic.loc[g.index])} for year,g in cohorts.groupby(cohorts.index.year)])
    return metrics, {'tile_profile':profile, 'tile_cohorts':daily, 'cohorts':cohorts, 'rank_ic':ic, 'by_year':periods, 'assignments':pd.concat(assignments)}
