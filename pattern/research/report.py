"""Render saved pilot results without optional Markdown table dependencies."""
from pathlib import Path
import json
import pandas as pd


def write_report(out):
    out=Path(out);cfg=json.loads((out/'config.json').read_text());metrics=json.loads((out/'metrics.json').read_text())
    summary=pd.read_csv(out/'deciles_summary.csv')
    table=['| Decile | Mean forward return | Resolved dates / total | Mean stocks | Unknown outcomes |',
           '|---|---:|---:|---:|---:|']
    for r in summary.itertuples():
        table.append(f'| D{r.tile} | {r.mean_forward_return:.3%} | {r.resolved_dates}/{r.dates} | {r.average_stocks:.1f} | {r.unknown_outcomes} |')
    rows=['# Frozen-data pipeline pilot','', '**Operational test only — not a strategy-performance finding.**','',
          f"I{cfg['lookback']}/R{cfg['horizon']}; MA{cfg['ma']}; effective history {cfg['effective_history']} sessions. {cfg.get('stocks_selected',120)} stocks selected using first-date inputs, one seed, two epochs, reduced CNN.",
          '', '## Primary result: n-tile separation','',*table,'',
          'Returns are arithmetic holding-horizon returns, not annualized returns. D1 is lowest score; D10 is highest. All valid-input test scores are retained; unknown outcomes make a tile-date unresolved. Mean returns describe resolved dates only. Means for different tiles can therefore use different dates; compare the paired spread series, not the difference between unequal-sample means.',
          '', '## Supporting diagnostics','', '```json',json.dumps(metrics,indent=2),'```','',
          'Sharpe uses non-overlapping close-to-close cohorts, not overlapping daily forward labels. Cost sensitivity assumes 10bp per transaction on each leg; borrow and next-open execution are absent. The short pilot has too few independent cohorts for reliable Sharpe inference. Missing outcomes in a selected extreme cohort block its portfolio Sharpe.',
          '', '## Verification','', 'Chronological input/target intervals do not overlap across splits. Training-only normalization and balancing. Saved predictions read back exactly; reloaded checkpoint reproduces probabilities. No full-grid training was launched.']
    (out/'report.md').write_text('\n'.join(rows)+'\n')
    return out/'report.md'
