"""Quick analysis of expanding-pathway predictions."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

RUN = Path("runs/expanding/20260419_174908_cdef6809")
df = pd.read_parquet(RUN / "predictions.parquet")
stats = pd.read_csv(RUN / "window_stats.csv")

print(f"rows={len(df):,}  tickers={df['ticker'].nunique()}  "
      f"dates={df['end_date'].nunique()}  windows={df['window'].nunique()}")
print(f"date range: {df['end_date'].min().date()} → {df['end_date'].max().date()}")
print()
print("Window stats summary:")
print(f"  total train time (sum): {stats['wall_seconds'].sum()/3600:.2f} h")
print(f"  mean wall/window:       {stats['wall_seconds'].mean()/60:.1f} min")
print(f"  median peak_gpu_mem:    {stats['gpu_peak_mem_gb'].median():.2f} GB")
print()

print("=== Overall AUC ===")
auc = roc_auc_score(df['label'], df['p_up_mean'])
print(f"  Test AUC = {auc:.4f}")

print()
print("=== Per-window AUC ===")
for w, g in df.groupby('window'):
    a = roc_auc_score(g['label'], g['p_up_mean'])
    yr = g['end_date'].dt.year.mode().iloc[0]
    print(f"  w{w:02d}  test_year~{yr}  n={len(g):,}  AUC={a:.4f}")

print()
print("=== Long-Short decile portfolio (per end_date, equal-weight) ===")
df['decile10'] = df.groupby('end_date')['p_up_mean'].transform(
    lambda x: pd.qcut(x.rank(method='first'), 10, labels=False, duplicates='drop')
)
ls_rows = []
for d, g in df.groupby('end_date'):
    long_ret = g.loc[g['decile10'] == 9, 'forward_return'].mean()
    short_ret = g.loc[g['decile10'] == 0, 'forward_return'].mean()
    ls_rows.append({'end_date': d, 'long': long_ret, 'short': short_ret,
                    'ls': long_ret - short_ret, 'n': len(g)})
ls = pd.DataFrame(ls_rows).sort_values('end_date')

# 20-day forward return; non-overlapping monthly windows would be cleaner but
# with daily end_dates and 20-day horizons we just take all dates and annualize
# the per-observation mean.
n_obs = len(ls)
mean_ls = ls['ls'].mean()
std_ls = ls['ls'].std()
# Assume ~252 trading days; 20-day forward return observed daily → naive
# annualization mean*252/20 is wrong because these observations are overlapping.
# Use monthly subsample (approximately every 20 days) for a cleaner check.
monthly = ls.iloc[::20].copy()
m_mean = monthly['ls'].mean()
m_std = monthly['ls'].std()
ann_ret = m_mean * (252/20)
ann_vol = m_std * np.sqrt(252/20)
sharpe = ann_ret/ann_vol if ann_vol > 0 else np.nan
print(f"  LS daily (overlapping 20d): mean={mean_ls*1e4:.1f}bps  std={std_ls*1e4:.1f}bps  N={n_obs:,}")
print(f"  LS monthly-sample (≈20-day non-overlap): "
      f"ann_ret={ann_ret*100:+.2f}%  ann_vol={ann_vol*100:.2f}%  Sharpe={sharpe:.2f}  N={len(monthly)}")

# Newey-West t-stat on the overlapping daily series.
def newey_west_t(x, lag=19):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mu = x.mean()
    xc = x - mu
    g0 = (xc**2).mean()
    s = g0
    for k in range(1, lag+1):
        gk = (xc[k:] * xc[:-k]).mean()
        w = 1 - k/(lag+1)
        s += 2*w*gk
    se = np.sqrt(s/n)
    return mu/se if se > 0 else np.nan

t = newey_west_t(ls['ls'].values, lag=19)
print(f"  LS Newey-West t-stat (lag=19): {t:+.2f}")

print()
print("=== Year-by-year LS ===")
ls['year'] = ls['end_date'].dt.year
by_year = ls.groupby('year')['ls'].agg(['count', 'mean', 'std'])
by_year['ann_pct'] = by_year['mean'] * (252/20) * 100
# share of positive days gives a rough sense of signal stability
by_year['pct_pos'] = ls.groupby('year')['ls'].apply(lambda x: (x>0).mean()*100)
print(by_year.round(3).to_string())

print()
print("=== Per-decile mean forward return (bps, equal-weight) ===")
dec_rets = df.groupby('decile10')['forward_return'].mean() * 1e4
print(dec_rets.round(1).to_string())

print()
print("=== Per-window LS summary ===")
for w, g in df.groupby('window'):
    ls_w = []
    for d, gg in g.groupby('end_date'):
        l = gg.loc[gg['decile10']==9, 'forward_return'].mean()
        s = gg.loc[gg['decile10']==0, 'forward_return'].mean()
        ls_w.append(l-s)
    if not ls_w:
        continue
    m = np.mean(ls_w)
    yr = g['end_date'].dt.year.mode().iloc[0]
    print(f"  w{w:02d} test={yr}  LS daily mean={m*1e4:+.1f}bps  ann={m*(252/20)*100:+.2f}%")
