"""Build a monthly research derivative of the frozen daily panel; no vendor calls."""
from pathlib import Path
import sys, json, hashlib, argparse, os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from pattern.research.full_cache import atomic_json
from pattern.research.monthly import aggregate_stock,candidate_windows,sample_months,verify_monthly_splits,render_monthly,month_calendar


def build(cfg,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'config.json',cfg);atomic_json(out/'status.json',{'state':'loading_daily'})
    ref=json.loads((ROOT/cfg['dataset_pointer']).read_text());source=Path(ref['path'])
    if hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()!=ref['manifest_sha256']:raise ValueError('Frozen manifest differs')
    filters=[('date','>=',pd.Timestamp(cfg['source_start'])),('date','<=',pd.Timestamp(cfg['source_end']))]
    calendar=pd.DatetimeIndex(pd.read_parquet(source/'candidate_sessions.parquet').date)
    calendar=calendar[(calendar>=cfg['source_start'])&(calendar<=cfg['source_end'])]
    months=month_calendar(calendar);pd.DataFrame({'date':months}).to_parquet(out/'month_calendar.parquet',index=False)
    print('Loading existing daily closes, volume and TRI',flush=True)
    daily=pd.read_parquet(source/'prices.parquet',columns=['security','date','close','volume','total_return_index_gross_dividends'],filters=filters)
    for file,columns in [('quality.parquet',['security','date','return_endpoint_ok']),('calendar_quality.parquet',['security','date','calendar_segment'])]:
        q=pd.read_parquet(source/file,columns=columns,filters=filters)
        daily=daily.merge(q,on=['security','date'],validate='one_to_one');del q
    n_daily=len(daily);daily=daily.sort_values(['security','date']);parts=[]
    for i,(security,g) in enumerate(daily.groupby('security',sort=True)):
        part=aggregate_stock(g,calendar,cfg['ma_days'])
        if len(part):parts.append(part)
        if i%250==0:
            print('MONTHLY STOCKS',i,flush=True);atomic_json(out/'status.json',{'state':'aggregating','stocks_processed':i})
    del daily
    monthly=pd.concat(parts,ignore_index=True);del parts
    monthly.to_parquet(out/'monthly_bars.parquet',index=False)
    candidates=candidate_windows(monthly,cfg['lookback_months'])
    endpoints=pd.Series(months,index=months.to_period('M'))
    candidates['target_date']=(candidates.date.dt.to_period('M')+1).map(endpoints)
    candidates=candidates[candidates.target_date.notna()]
    candidates.groupby('date').agg(stocks=('security','nunique'),known_labels=('forward_return_1','count')).to_csv(out/'eligible_stocks_by_month.csv')
    selected=[]
    for name in ['train','validation','comparison']:
        c=cfg[name]
        p=sample_months(candidates,c['start'],c['end'],total=c.get('rows'),stocks=c.get('stocks'),seed=cfg['sample_seed'])
        p['partition']=name;selected.append(p)
    sample=pd.concat(selected,ignore_index=True);sample['sample_id']=np.arange(len(sample))
    verify_monthly_splits(sample)
    if not np.isfinite(sample[['momentum_12_1','price_to_ma200']].to_numpy()).all():raise ValueError('Missing historical baseline input')
    sample.to_parquet(out/'sample.parquet',index=False)
    sample.assign(year=sample.date.dt.year).groupby(['partition','year']).agg(rows=('security','size'),stocks=('security','nunique'),months=('date','nunique')).to_csv(out/'coverage.csv')
    images=np.lib.format.open_memmap(out/'images.npy',mode='w+',dtype='uint8',shape=(len(sample),1,64,36))
    atomic_json(out/'status.json',{'state':'rendering','samples':len(sample)})
    chosen={k:g for k,g in sample.groupby('security')};audit=[]
    for i,(security,g) in enumerate(monthly.groupby('security',sort=True)):
        if security not in chosen:continue
        g=g.sort_values('month_number').reset_index(drop=True)
        a=g[['open','high','low','close','volume','ma200']].to_numpy(float)
        for _,s in chosen[security].iterrows():
            end=int(s.window_end);block=a[end-11:end+1];images[int(s.sample_id)]=render_monthly(block)
            if len(audit)<4 and (not audit or audit[-1]['security']!=security):
                audit.append({'security':security,'date':str(s.date.date()),'sample_id':int(s.sample_id)})
                g.iloc[end-11:end+1].to_csv(out/f'example_{len(audit)}_bars.csv',index=False)
        if i%500==0:print('RENDER STOCKS',i,flush=True)
    images.flush()
    atomic_json(out/'examples.json',audit)
    atomic_json(out/'metadata.json',{'dataset':ref,'daily_rows_read':n_daily,'monthly_bars':len(monthly),
        'input_eligible_stock_months':len(candidates),'sample_rows':len(sample),'image_shape':[1,64,36],
        'bar_basis':'first/max/min/last stored adjusted daily close; sum stored daily volume',
        'ma_basis':'200 daily adjusted closes including current month-end, computed before aggregation',
        'label_basis':'next-calendar-month gross-dividend TRI ratio; unknown outcomes retained',
        'selection_uses_future_returns':False,'reserved_period':'2023 onward not used','annualization':12})
    os.environ.setdefault('MPLCONFIGDIR','/tmp/pattern-monthly-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(audit),figsize=(12,6),squeeze=False)
    for ax,record in zip(axes[0],audit):
        ax.imshow(images[record['sample_id'],0],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        ax.set_title(f"{record['security']}\n{record['date']}");ax.axis('off')
    fig.tight_layout();fig.savefig(out/'example_charts.png',dpi=150);plt.close(fig)
    atomic_json(out/'status.json',{'state':'complete','sample_rows':len(sample)})
    print('MONTHLY BUILD COMPLETE',len(sample),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--config',default=str(Path(__file__).with_name('config.json')));a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    try:build(cfg,Path(a.run_dir))
    except BaseException as e:
        if Path(a.run_dir).exists():atomic_json(Path(a.run_dir)/'status.json',{'state':'failed','error':repr(e)})
        raise
