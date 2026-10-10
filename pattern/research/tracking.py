"""Opt-in W&B logging of scalar training and aggregate evaluation metrics."""
import hashlib
import json
from pathlib import Path
from pattern.research.full_cache import atomic_json


def start(cfg,exp,seed,dest):
    options=cfg.get('wandb',{})
    if not options.get('enabled'):return None
    import wandb
    dest=Path(dest)
    run_id=hashlib.sha256(str(dest.resolve()).encode()).hexdigest()[:16]
    run=wandb.init(project=options.get('project','pattern-cnn'),entity=options.get('entity'),
        id=run_id,resume='allow',mode='online',name=f"{exp['name']}-seed-{seed}",group=dest.parents[1].name,
        dir=str(dest),config={'experiment':exp,'seed':seed,'max_epochs':cfg['max_epochs'],
            'patience':cfg['patience'],'batch_size':cfg['batch_size'],'device':cfg['device'],
            'sample_seed':cfg['sample_seed'],'target':'absolute direction','training_start':'fresh initialization',
            'train_period':cfg['train'],'validation_period':cfg['validation'],'comparison_period':cfg['comparison']},
        settings=wandb.Settings(init_timeout=90,disable_git=True,save_code=False))
    run.define_metric('epoch')
    run.define_metric('train/*',step_metric='epoch')
    run.define_metric('validation/*',step_metric='epoch')
    run.define_metric('timing/*',step_metric='epoch')
    atomic_json(dest/'wandb.json',{'id':run.id,'url':run.url,'project':run.project,'entity':run.entity})
    print('LIVE WANDB',run.url,flush=True)
    return run


def log_epoch(run,row,lr,best,stale):
    if run is None:return
    run.log({'epoch':row['epoch'],'train/loss':row['train_loss'],'validation/loss':row['validation_loss'],
        'train/accuracy':row['train_accuracy'],'validation/accuracy':row['validation_accuracy'],
        'train/learning_rate':lr,'validation/best_loss':best,'validation/stale_epochs':stale,
        'timing/epoch_seconds':row['seconds']})


def finish_metrics(cfg,dest,metrics,tables):
    if not cfg.get('wandb',{}).get('enabled'):return
    import wandb
    run=wandb.run
    if run is None:raise RuntimeError('Requested online W&B run is missing')
    for k,v in metrics.items():run.summary['comparison/'+k]=v
    for k,v in json.loads((Path(dest)/'fit_summary.json').read_text()).items():run.summary['fit/'+k]=v
    profile=tables['tile_profile'].reset_index()
    run.log({'comparison/decile_profile':wandb.Table(dataframe=profile),
             'comparison/by_year':wandb.Table(dataframe=tables['by_year'])})
    run.finish()
