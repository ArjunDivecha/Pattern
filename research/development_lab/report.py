"""Render a completed development run without rerunning models."""
from pathlib import Path
import argparse
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(root):
    root=Path(root).resolve()
    status=json.loads((root/'status.json').read_text())
    if status['state']!='complete':raise ValueError('Run is not complete')
    cfg=json.loads((root/'config.json').read_text())
    rows=pd.read_csv(root/'comparison.csv')
    columns=min(3,len(rows)); nrows=(len(rows)+columns-1)//columns
    fig,axes=plt.subplots(nrows,columns,figsize=(6*columns,4*nrows),squeeze=False)
    for unused in axes.ravel()[len(rows):]: unused.set_visible(False)
    lines=['# Development study','',
        'This study measures development performance and training behavior. It does not establish independent test performance.', '',
        f"Training budget: {cfg['max_epochs']} epochs per fit; seeds {cfg['seeds']}. Labels are absolute return direction. All returns are gross; no transaction costs.", '',
        '## Comparison', '',
        '| Experiment | Seed | Top return | Bottom return | Spread | Rank IC | Observed holding-period Sharpe | Strict Sharpe | Cohorts |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ax,(_,r) in zip(axes.ravel(),rows.iterrows()):
        dest=root/r.experiment/f'seed_{r.seed}'
        profile=pd.read_parquet(dest/'tile_profile.parquet')
        ax.bar(profile.index,profile.mean_return*100,color='#356b85')
        ax.axhline(0,color='black',linewidth=.6)
        ax.set(title=f"{r.experiment}, seed {r.seed}",xlabel='Predicted decile (low → high)',ylabel=f'Mean {r.horizon}-session return (%)',xticks=profile.index)
        strict='Unavailable' if pd.isna(r.strict_cohort_sharpe) else f'{r.strict_cohort_sharpe:.3f}'
        lines.append(f'| {r.experiment} | {r.seed} | {r.top:.3%} | {r.bottom:.3%} | {r.spread:.3%} | {r.rank_ic:.4f} | {r.observed_cohort_sharpe:.3f} | {strict} | {r.cohorts} |')
    fig.tight_layout();fig.savefig(root/'deciles.png',dpi=160);plt.close(fig)
    lines+=['',f'![Decile returns]({root}/deciles.png)','',
        'Observed metrics average known returns after ranking every sampled stock. Missing outcomes can bias these statistics. Strict Sharpe is unavailable if any extreme-leg cohort has a missing outcome. No missing return is assumed to be zero.', '',
        'Sharpe is computed from consecutive nonoverlapping H-session long-short P&L increments, annualized by sqrt(252/H), with zero cash rate. It is not daily NAV Sharpe. Raw 5-day and 20-day spreads have different holding periods and are not directly comparable.', '',
        '## Coverage and runtime','']
    for _,r in rows.iterrows():
        dest=root/r.experiment/f'seed_{r.seed}'
        norm=json.loads((dest/'normalization.json').read_text())
        epochs=pd.read_csv(dest/'epochs.csv')
        lines.append(f"- {r.experiment}, seed {r.seed}: {norm['train_rows']:,} labeled training rows; {norm['validation_rows']:,} validation rows; {norm['comparison_rows']:,} comparison rows; {r.unknown_outcomes:,} unknown comparison outcomes; {epochs.seconds.sum():.1f} seconds training plus validation.")
        profile=pd.read_parquet(dest/'tile_profile.parquet')
        lines.append(f"  Missing outcomes: bottom decile {profile.loc[1,'unknown_fraction']:.2%}; top decile {profile.loc[cfg['ntiles'],'unknown_fraction']:.2%}.")
    lines+=['','## Interpretation limits','',
        'Training: 1997–2012; checkpoint validation: July 2014–June 2016; development comparison: 2018–early October 2022. The split gaps protect up to 239 history sessions and 60 forward sessions. The reserved 2023+ period is untouched.', '',
        'Per-seed and per-year outputs are saved alongside each fit. An epoch-cap stop does not establish convergence. The fixed sample, scores and assignments are available for reconciliation.', '',
        'Current inputs are cached daily I20/MA20 images. Other lookbacks, MA removal and weekly representations need separately rendered sample packs. No old annual training was restarted and no cloud instance was provisioned.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return root/'report.md'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir');a=p.parse_args();print(report(a.run_dir))
