"""Create gross-only full-year comparison report from saved scores and diagnostics."""
from pathlib import Path
import sys,json,os,tempfile
os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='pattern-gross-mpl-'))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    root=Path(root);annual=pd.read_csv(root/'annual_gross_metrics.csv').set_index('year')
    def pct(v):return f'{v:+.2%}'
    def num(v):return 'Withheld (missing outcomes)' if pd.isna(v) else f'{v:.2f}'
    rows=['# Full-year gross signal extension','',
          'Unchanged I10/R5 models, MA10, one seed, two training epochs. The same 100-security initial-date sample is retained; daily usable counts vary. Full-year out-of-sample scoring extends the earlier 30-session smoke test. No transaction costs are evaluated. This remains a small-model, small-universe diagnostic, not the full grid.',
          '', '| Metric | 2010 | 2011 |','|---|---:|---:|']
    fields=[('Signal dates','signal_dates',lambda x:str(int(x))),('Average stocks per date','average_stocks_per_date',lambda x:f'{x:.1f}'),
            ('Top decile, matched dates','top_return_paired',pct),('Bottom decile, matched dates','bottom_return_paired',pct),
            ('Top minus bottom','top_minus_bottom',pct),('Median daily spread','median_daily_spread',pct),
            ('Positive-spread dates','positive_spread_fraction',pct),('Mean rank IC','rank_ic',lambda x:f'{x:+.4f}'),
            ('Spread HAC t-stat (20 lags)','spread_hac_t',num),('Gross non-overlapping-cohort Sharpe','gross_cohort_sharpe',num),
            ('Resolved top/bottom dates','paired_extreme_dates',lambda x:str(int(x))),('Unknown stock outcomes','unknown_outcomes',lambda x:str(int(x))),
            ('Unresolved holding cohorts','unresolved_cohorts',lambda x:str(int(x)))]
    for title,key,fmt in fields:rows.append('| '+title+' | '+' | '.join(fmt(annual.loc[y,key]) for y in [2010,2011])+' |')
    rows+=['','Returns are mean five-session arithmetic total returns, not annualized returns. Top and bottom are calculated over the same resolved dates. Daily forward-return windows overlap; HAC t-statistics use 20 calendar-session lags as a supporting approximation. Missing tile outcomes are not filled or removed from score assignments; descriptive means condition on resolved dates. Any missing holding cohort blocks the full gross portfolio Sharpe.','',
           '## Decile ordering','', '![Decile ordering](deciles.png)','',
           'All ten bars within each year use one common set of dates on which every decile has known outcomes. This avoids differing date samples across bars; those dates differ from the paired-extreme table above.',
           '', '## Previously seen dates versus the extension','', '| Year | Period | Paired dates | Top | Bottom | Spread |','|---|---|---:|---:|---:|---:|']
    for year in [2010,2011]:
        t=pd.read_csv(root/str(year)/'seen_vs_extension.csv')
        for r in t.itertuples():rows.append(f'| {year} | {r.period} | {r.paired_dates} | {pct(r.top)} | {pct(r.bottom)} | {pct(r.spread)} |')
    rows+=['','## Tail sensitivity','','| Year | Tail size | Paired dates | Top minus bottom |','|---|---:|---:|---:|']
    for year in [2010,2011]:
        for r in pd.read_csv(root/str(year)/'tail_comparisons.csv').itertuples():rows.append(f'| {year} | {100/r.tiles:g}% | {r.paired_dates} | {pct(r.spread)} |')
    rows+=['','## Concentration','','Subtracting a stock’s original fixed-weight contribution is an attribution diagnostic, not an implementable selection rule. No source return is clipped, corrected or deleted. Contributions sum exactly to the observed paired spread.','','| Year | Largest positive contributor | Contribution to mean spread | Spread less that contribution |','|---|---|---:|---:|']
    for year in [2010,2011]:
        r=annual.loc[year];rows.append(f'| {year} | {r.largest_positive_contributor} | {pct(r.largest_contributor_mean_spread)} | {pct(r.spread_without_largest_contributor_fixed_weights)} |')
    rows+=['','## Verification and limits','','Saved scores read back exactly. All previously reported 30-session scores reproduce within numerical tolerance; their returns match. Original checkpoint hashes are unchanged. The 2023–2025 reserved block was not loaded. Contributions reconcile to every resolved daily spread.','',
           'The source sample was chosen from stocks observed at the initial 1996 date; this extension does not add IPOs or broaden to the full available universe. The architecture, sparse training sample, and two-epoch budget are unchanged from the smoke test. These results can assess persistence of that pilot, but cannot settle the value of the full horizon study.']
    (root/'report.md').write_text('\n'.join(rows)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,year in zip(axes,[2010,2011]):
        curve=pd.read_csv(root/str(year)/'deciles_common_dates.csv')
        ax.bar(curve.tile,curve.mean_return_common_dates*100,color=['#b45d42' if t==1 else '#275779' if t==10 else '#b9c9d4' for t in curve.tile])
        ax.axhline(0,color='#555',lw=.8);ax.set_xticks(range(1,11),[f'D{i}' for i in range(1,11)])
        ax.set_title(f'{year}: {int(curve.common_dates.iloc[0])} complete dates');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean five-session return (%)')
    fig.suptitle('Gross signal: decile ordering across full test years')
    fig.tight_layout();fig.savefig(root/'deciles.png',dpi=160)
    print(root/'report.md')

if __name__=='__main__':main(sys.argv[1])
