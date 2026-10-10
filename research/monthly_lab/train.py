"""Train the 12-month close-chart experiment, sharing the tested resumable trainer."""
from pathlib import Path
import sys,json,hashlib,argparse,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
import torch
from research.development_lab.run import train_and_score
from pattern.research.monthly import monthly_metrics,verify_monthly_splits
from pattern.research.full_cache import atomic_json
from pattern.research import tracking


def save_result(dest,scores,calendar,dates,ntiles):
    dest.mkdir(parents=True,exist_ok=True)
    metrics,tables=monthly_metrics(scores,calendar,dates,ntiles)
    scores.to_parquet(dest/'scores.parquet',index=False)
    atomic_json(dest/'metrics.json',metrics)
    for name,table in tables.items():table.to_parquet(dest/f'{name}.parquet',index=True)
    return metrics,tables


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=Path(args.run_dir).resolve();cfg=json.loads((out/'config.json').read_text())
    if not args.resume and (out/'comparison.csv').exists():raise ValueError('Use --resume for an existing training run')
    files=[Path(__file__),ROOT/'research/monthly_lab/build.py',ROOT/'research/monthly_lab/report.py',ROOT/'pattern/research/monthly.py',
        ROOT/'research/development_lab/run.py',ROOT/'pattern/research/checkpoint.py',ROOT/'pattern/research/tracking.py',
        ROOT/'pattern/research/development.py',ROOT/'pattern/models/cnn.py',ROOT/'pattern/models/blocks.py',ROOT/'pattern/config.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if args.resume and hashes!=json.loads((out/'source_hashes.json').read_text()):raise ValueError('Resume source differs')
    atomic_json(out/'source_hashes.json',hashes)
    try:
        device=torch.device(cfg['device'])
        if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no CPU fallback')
        torch.set_num_threads(cfg['threads'])
        sample=pd.read_parquet(out/'sample.parquet');images=np.load(out/'images.npy',mmap_mode='r')
        verify_monthly_splits(sample)
        if images.shape!=(len(sample),1,64,36):raise ValueError('Wrong monthly image pack')
        calendar=pd.DatetimeIndex(pd.read_parquet(out/'month_calendar.parquet').date)
        comp=sample[sample.partition.eq('comparison')]
        dates=pd.DatetimeIndex(comp.date.unique()).sort_values()
        if not (np.diff(dates.to_period('M').asi8)==1).all():raise ValueError('Missing comparison month')
        atomic_json(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'pandas':pd.__version__,'device':str(device),'pid':os.getpid()})
        rows=[];baseline={}
        for name in ['momentum_12_1','price_to_ma200']:
            scores=comp[['security','date','sample_id']].copy();scores['forward_return']=comp.forward_return_1;scores['score']=comp[name]
            metrics,tables=save_result(out/'baselines'/name,scores,calendar,dates,cfg['ntiles'])
            baseline[name]=metrics;rows.append({'experiment':name,'seed':-1,'kind':'baseline',**metrics})
        pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
        for exp in cfg['experiments']:
            for seed in cfg['seeds']:
                atomic_json(out/'status.json',{'state':'training','experiment':exp['name'],'seed':seed,'frequency':'monthly'})
                dest=out/exp['name']/f'seed_{seed}'
                scores=train_and_score(cfg,exp,seed,sample,images,12,dest,device,dates)
                metrics,tables=save_result(dest,scores,calendar,dates,cfg['ntiles'])
                if cfg['wandb']['enabled']:
                    import wandb
                    for name,m in baseline.items():
                        for key in ['spread','rank_ic','observed_cohort_sharpe']:wandb.run.summary[f'baseline/{name}/{key}']=m[key]
                tracking.finish_metrics(cfg,dest,metrics,tables)
                rows.append({'experiment':exp['name'],'seed':seed,'kind':'cnn',**metrics})
                pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
                print('MONTHLY FIT COMPLETE',seed,json.dumps(metrics),flush=True)
        atomic_json(out/'status.json',{'state':'complete','fits':len(cfg['seeds']),'baselines':2})
        from research.monthly_lab.report import report
        report(out)
        atomic_json(out/'pipeline_status.json',{'state':'complete','fits':len(cfg['seeds']),'reports_complete':True})
    except BaseException as e:
        if cfg['wandb']['enabled']:
            import wandb
            if wandb.run is not None:wandb.finish(exit_code=1)
        atomic_json(out/'status.json',{'state':'failed','error':repr(e)})
        raise

if __name__=='__main__':main()
