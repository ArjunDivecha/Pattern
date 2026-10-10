"""Run the authorized six-fit budget study and publish reports on completion."""
from pathlib import Path
import json
import argparse
import os
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from pattern.research.full_cache import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'research/development_lab/training_budget.json'))
    parser.add_argument('--run-dir',default=str(ROOT/'runs/development_training_budget_20260918'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    out=Path(args.run_dir).resolve()
    try:
        subprocess.run([sys.executable,str(ROOT/'research/development_lab/run.py'),
            '--config',args.config,'--run-dir',str(out)]+(['--resume'] if args.resume else []),cwd=ROOT,check=True)
        os.environ.setdefault('MPLCONFIGDIR','/tmp/pattern-budget-matplotlib')
        from research.development_lab.report import report
        report(out)
        rows=pd.read_csv(out/'comparison.csv')
        summaries=[]
        for _,r in rows.iterrows():
            folder=out/r.experiment/f'seed_{r.seed}'
            f=json.loads((folder/'fit_summary.json').read_text())
            summaries.append({'experiment':r.experiment,'seed':r.seed,**f})
        fits=pd.DataFrame(summaries);fits.to_csv(out/'training_summary.csv',index=False)
        stability=rows.groupby('experiment')[['spread','rank_ic','observed_cohort_sharpe']].agg(['mean','std','min','max'])
        stability.to_csv(out/'seed_stability.csv')
        lines=['\n## Training budget and seed stability\n',
            '| Experiment | Seed | Epochs | Best epoch | Stop reason | Best validation loss |',
            '|---|---:|---:|---:|---|---:|']
        for _,r in fits.iterrows():
            lines.append(f'| {r.experiment} | {r.seed} | {r.epochs} | {r.best_epoch} | {r.stop_reason} | {r.best_validation_loss:.5f} |')
        lines+=['','Validation loss selects checkpoints. Comparison returns do not control early stopping. Seed dispersion is saved in seed_stability.csv; it is not a confidence interval over independent market histories.']
        with (out/'report.md').open('a') as f:f.write('\n'.join(lines)+'\n')
        atomic_json(out/'pipeline_status.json',{'state':'complete','fits':len(rows),'reports_complete':True})
        print('ALL FITS AND REPORTS COMPLETE',out,flush=True)
    except BaseException as exc:
        if out.exists():atomic_json(out/'pipeline_status.json',{'state':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
