"""
=============================================================================
SCRIPT NAME: plot_mid_cell_since2020.py
=============================================================================

INPUT FILES:
- runs/expanding/20260419_174908_cdef6809/predictions_monthly.parquet
- data/r1000_ohlcv_database.parquet

OUTPUT FILES:
- runs/expanding/.../backtest_liquid_grid/mid_cell_since2020_cum.pdf
- runs/expanding/.../backtest_liquid_grid/mid_cell_since2020_cum.png
- runs/expanding/.../backtest_liquid_grid/mid_cell_since2020_monthly.xlsx

DESCRIPTION:
Re-derives the Mid-dv x High-vol x Low-mom cell from the R1000 expanding
predictions, computes monthly 50/50 LS returns, and plots cumulative
growth of $1 from 2020-01-01 onward (gross and net of 8 bps half-spread
+ 1% expense-like drag).  Also reports yearly returns and metrics.
=============================================================================
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


PRED = Path("runs/expanding/20260419_174908_cdef6809/predictions_monthly.parquet")
OHLCV = Path("data/r1000_ohlcv_database.parquet")
OUT = Path("runs/expanding/20260419_174908_cdef6809/backtest_liquid_grid")
OUT.mkdir(parents=True, exist_ok=True)

SINCE = "2020-01-01"
HALF_SPREAD_BPS = 8.0       # Mid-dv tertile realistic half-spread
ANNUAL_OVERHEAD = 0.0       # set >0 to model financing/admin


def newey_west_t(x, lag=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n == 0: return np.nan
    mu = x.mean(); xc = x - mu
    s = (xc**2).mean()
    for k in range(1, lag+1):
        if k >= n: break
        gk = (xc[k:]*xc[:-k]).mean(); w = 1 - k/(lag+1)
        s += 2*w*gk
    se = np.sqrt(max(s,0)/n)
    return float(mu/se) if se > 0 else np.nan


def compute_features(ohlcv):
    df = ohlcv[["Ticker","Date","Close","Volume","AdjClose"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker","Date"])
    g = df.groupby("Ticker")
    df["dv"] = df["Close"] * df["Volume"]
    df["dv_60d"] = g["dv"].transform(lambda s: s.rolling(60, min_periods=15).mean())
    df["logp"] = np.log(df["AdjClose"].replace(0, np.nan))
    df["mom_12_1"] = g["logp"].shift(21) - g["logp"].shift(252+21)
    df["ret"] = g["AdjClose"].pct_change()
    df["vol_60d"] = g["ret"].transform(lambda s: s.rolling(60, min_periods=20).std() * np.sqrt(252))
    return (df[["Ticker","Date","dv_60d","mom_12_1","vol_60d"]]
            .rename(columns={"Ticker":"ticker","Date":"end_date"}))


def bucket3(df, col):
    labels = ["Low","Mid","High"]
    def assign(g):
        if g[col].notna().sum() < 3:
            return pd.Series([pd.NA]*len(g), index=g.index)
        return pd.qcut(g[col].rank(method="first"), 3, labels=labels, duplicates="drop")
    return df.groupby("end_date", group_keys=False).apply(assign)


print("Loading …")
preds = pd.read_parquet(PRED)
preds["end_date"] = pd.to_datetime(preds["end_date"])
ohlcv = pd.read_parquet(OHLCV)
feats = compute_features(ohlcv)
preds = preds.merge(feats, on=["ticker","end_date"], how="left")

print("Bucketing …")
preds["dv_b"]  = bucket3(preds, "dv_60d")
preds["mom_b"] = bucket3(preds, "mom_12_1")
preds["vol_b"] = bucket3(preds, "vol_60d")
univ = preds[["dv_b","mom_b","vol_b"]].notna().all(axis=1)
p = preds[univ].copy()

cell = p[(p["dv_b"]=="Mid") & (p["vol_b"]=="High") & (p["mom_b"]=="Low")]
print(f"Mid-dv × High-vol × Low-mom cell: {len(cell):,} rows, "
      f"{cell['end_date'].nunique()} months, "
      f"{cell['end_date'].min().date()} → {cell['end_date'].max().date()}")

print("Building monthly LS returns …")
rows = []
for date, g in cell.groupby("end_date"):
    if len(g) < 4:
        continue
    n = len(g)
    rank = g["p_up_mean"].rank(method="first") / n
    top = g[rank > 0.5]; bot = g[rank <= 0.5]
    rows.append({"end_date": date,
                 "n": n, "n_top": len(top), "n_bot": len(bot),
                 "TOP": top["forward_return"].mean(),
                 "BOT": bot["forward_return"].mean(),
                 "EW":  g["forward_return"].mean(),
                 "top_set": frozenset(top["ticker"]),
                 "bot_set": frozenset(bot["ticker"])})
pf = pd.DataFrame(rows).sort_values("end_date").reset_index(drop=True)
pf["LS"] = pf["TOP"] - pf["BOT"]

# One-sided turnover
top_turn = [np.nan]; bot_turn = [np.nan]
for i in range(1, len(pf)):
    pt, pb = pf.loc[i-1,"top_set"], pf.loc[i-1,"bot_set"]
    ct, cb = pf.loc[i,"top_set"],   pf.loc[i,"bot_set"]
    top_turn.append(len(ct ^ pt) / max(len(ct)+len(pt),1))
    bot_turn.append(len(cb ^ pb) / max(len(cb)+len(pb),1))
pf["turn_top"] = top_turn; pf["turn_bot"] = bot_turn

# Round-trip cost per month per leg = 2 × half_spread × turnover
# Total monthly drag = 2 sides × turnover × (2 × half_spread/10000)
pf["cost_per_month"] = (pf["turn_top"].fillna(pf["turn_top"].mean())
                         + pf["turn_bot"].fillna(pf["turn_bot"].mean())) \
                         * (HALF_SPREAD_BPS / 1e4) * 2  # both sides
pf["LS_net"] = pf["LS"] - pf["cost_per_month"] - ANNUAL_OVERHEAD/12

pf = pf.drop(columns=["top_set","bot_set"])

# Filter since 2020
ps = pf[pf["end_date"] >= SINCE].copy().reset_index(drop=True)
print(f"\nSince {SINCE}: {len(ps)} months ({ps['end_date'].min().date()} → {ps['end_date'].max().date()})")

# Cumulative growth
ps["cum_LS_gross"] = (1 + ps["LS"]).cumprod()
ps["cum_LS_net"]   = (1 + ps["LS_net"]).cumprod()
ps["cum_TOP"]      = (1 + ps["TOP"]).cumprod()
ps["cum_BOT"]      = (1 + ps["BOT"]).cumprod()
ps["cum_EW"]       = (1 + ps["EW"]).cumprod()

# Summary stats
def stats(r, label, n=len(ps)):
    r = r.dropna()
    mu, sd = r.mean(), r.std()
    cagr = (1+r).prod()**(12/max(len(r),1)) - 1
    sh   = (mu*12)/(sd*np.sqrt(12)) if sd > 0 else np.nan
    t    = newey_west_t(r.to_numpy())
    return {"strat": label, "months": len(r),
            "monthly_mean_pct": mu*100, "monthly_std_pct": sd*100,
            "cagr_pct": cagr*100, "sharpe": sh, "nw_t": t}

summary = pd.DataFrame([
    stats(ps["LS"],      "LS gross"),
    stats(ps["LS_net"],  "LS net (8bps half-spread, both sides)"),
    stats(ps["TOP"],     "TOP only (long)"),
    stats(ps["BOT"],     "BOT only (short pays you the negative)"),
    stats(ps["EW"],      "EW cell"),
])
print()
print("Summary (since 2020):")
print(summary.round(2).to_string(index=False))

# Yearly returns
ps["year"] = ps["end_date"].dt.year
yearly = ps.groupby("year").agg(
    months=("end_date","count"),
    univ=("n","mean"),
    n_top=("n_top","mean"),
    turn_top=("turn_top","mean"),
    LS_gross_sum=("LS","sum"),
    LS_net_sum=("LS_net","sum"),
    TOP_sum=("TOP","sum"),
    BOT_sum=("BOT","sum"),
    EW_sum=("EW","sum"),
).reset_index()
yearly["LS_gross_pct"] = np.expm1(yearly["LS_gross_sum"])*100
yearly["LS_net_pct"]   = np.expm1(yearly["LS_net_sum"])*100
yearly["TOP_pct"]      = np.expm1(yearly["TOP_sum"])*100
yearly["BOT_pct"]      = np.expm1(yearly["BOT_sum"])*100
yearly["EW_pct"]       = np.expm1(yearly["EW_sum"])*100
print()
print("Yearly returns (since 2020):")
print(yearly[["year","months","univ","n_top","turn_top",
              "LS_gross_pct","LS_net_pct","TOP_pct","BOT_pct","EW_pct"]].round(2).to_string(index=False))

# Save monthly + yearly
with pd.ExcelWriter(OUT / "mid_cell_since2020_monthly.xlsx",
                     engine="openpyxl") as xw:
    summary.round(3).to_excel(xw, sheet_name="summary", index=False)
    yearly.round(2).to_excel(xw,  sheet_name="yearly", index=False)
    ps.round(4).to_excel(xw,      sheet_name="monthly", index=False)
print(f"\nwrote {OUT / 'mid_cell_since2020_monthly.xlsx'}")

# ── Chart ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False,
                           gridspec_kw={"height_ratios":[3,1]})

ax = axes[0]
ax.plot(ps["end_date"], ps["cum_LS_gross"], color="#0a5cad", lw=2.2,
        label=f"LS gross  ({summary.iloc[0]['cagr_pct']:+.1f}% CAGR, Sh {summary.iloc[0]['sharpe']:+.2f}, t {summary.iloc[0]['nw_t']:+.2f})")
ax.plot(ps["end_date"], ps["cum_LS_net"],   color="#d62728", lw=2.2,
        label=f"LS net  ({summary.iloc[1]['cagr_pct']:+.1f}% CAGR, Sh {summary.iloc[1]['sharpe']:+.2f}, t {summary.iloc[1]['nw_t']:+.2f})")
ax.plot(ps["end_date"], ps["cum_EW"],       color="#888888", lw=1.0, ls="--",
        label=f"EW cell  ({summary.iloc[4]['cagr_pct']:+.1f}% CAGR)")
ax.set_title("Mid-dv × High-vol × Low-mom — cumulative growth of \\$1 since 2020")
ax.set_ylabel("growth of \\$1")
ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=9)
ax.axhline(1.0, color="black", lw=0.5, alpha=0.5)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Year bar chart on bottom panel
ax2 = axes[1]
yr = yearly.copy()
x = np.arange(len(yr)); w = 0.35
ax2.bar(x-w/2, yr["LS_gross_pct"], w, color="#0a5cad", label="LS gross")
ax2.bar(x+w/2, yr["LS_net_pct"],   w, color="#d62728", label="LS net")
ax2.axhline(0, color="black", lw=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(yr["year"].astype(int).astype(str))
ax2.set_ylabel("Annual return (%)")
ax2.set_title("Yearly LS returns")
ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)
for i, (g, n) in enumerate(zip(yr["LS_gross_pct"], yr["LS_net_pct"])):
    ax2.text(i-w/2, g + (1 if g>=0 else -2.5), f"{g:+.0f}", ha="center", fontsize=7)
    ax2.text(i+w/2, n + (1 if n>=0 else -2.5), f"{n:+.0f}", ha="center", fontsize=7)

plt.tight_layout()
out_pdf = OUT / "mid_cell_since2020_cum.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"wrote {out_pdf}")
print(f"wrote {out_pdf.with_suffix('.png')}")
