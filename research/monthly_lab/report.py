"""Monthly results, explicitly using calendar-month returns and sqrt(12) Sharpe."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/pattern-monthly-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def report(out):
    out=Path(out);rows=pd.read_csv(out/'comparison.csv')
    fig,axes=plt.subplots(2,3,figsize=(15,8));lines=['# Monthly chart experiment','',
      '12 monthly bars made from adjusted daily closes, monthly summed volume and trailing daily MA200 predict the next calendar month. Targets are absolute return direction; scores rank stocks into deciles.','',
      '| Model | Seed | Top | Bottom | Spread | Rank IC | Observed monthly Sharpe |',
      '|---|---:|---:|---:|---:|---:|---:|']
    fits=[]
    for ax,(_,r) in zip(axes.ravel(),rows.iterrows()):
        dest=out/'baselines'/r.experiment if r.kind=='baseline' else out/r.experiment/f'seed_{r.seed}'
        profile=pd.read_parquet(dest/'tile_profile.parquet')
        ax.bar(profile.index,profile.mean_return*100,color='#356b85');ax.axhline(0,color='black',linewidth=.5)
        name=r.experiment if r.kind=='baseline' else f'{r.experiment}, seed {r.seed}'
        ax.set(title=name,xlabel='Decile',ylabel='Mean next-month return (%)',xticks=profile.index)
        lines.append(f'| {r.experiment} | {r.seed if r.seed>=0 else "—"} | {r.top:.3%} | {r.bottom:.3%} | {r.spread:.3%} | {r.rank_ic:.4f} | {r.observed_cohort_sharpe:.3f} |')
        if r.kind=='cnn':fits.append({'seed':r.seed,**json.loads((dest/'fit_summary.json').read_text())})
    for ax in axes.ravel()[len(rows):]:ax.set_visible(False)
    fig.tight_layout();fig.savefig(out/'deciles.png',dpi=150);plt.close(fig)
    pd.DataFrame(fits).to_csv(out/'training_summary.csv',index=False)
    rows[rows.kind.eq('cnn')].groupby('experiment')[['spread','rank_ic','observed_cohort_sharpe']].agg(['mean','std','min','max']).to_csv(out/'seed_stability.csv')
    lines+=['',f'![Deciles]({out}/deciles.png)','',
       'All results are gross. Observed metrics reweight available outcomes after ranking; they may be biased by missing returns. Unknown counts/fractions by tile and strict Sharpe are saved for each model. No missing return is assumed zero.', '',
       'Monthly Sharpe uses consecutive nonoverlapping calendar-month long-short P&L and sqrt(12), with zero cash rate. Formation is at month-end close; this is a signal study, not an execution backtest.', '',
       'Training: 1997–2011; checkpoint validation: 2014–2015; comparison formations: January 2018–November 2022, with targets through December 2022. Samples include 100,000 training candidates, 20,000 validation candidates and 500 stocks per comparison month. Selection uses historical inputs only. Actual labeled training counts exclude unknown outcomes.', '',
       'The 200-day MA requires history before the first pictured month. The split check compares actual earliest input dates with prior-partition target dates. The 2023+ reserve is unused.', '',
       'Momentum ranks adjusted-close return from t−12 to t−1 month-end. Price-to-MA200 ranks current close divided by MA200 minus one. Both use the same stocks and target returns as the CNN. Seed dispersion is not a confidence interval over independent market histories.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return out/'report.md'
