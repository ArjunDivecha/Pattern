"""Top-3 vs Bottom-3 decile portfolio, expanding pathway."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

RUN = Path("runs/expanding/20260419_174908_cdef6809")
df = pd.read_parquet(RUN / "predictions.parquet")

df["decile"] = df.groupby("end_date")["p_up_mean"].transform(
    lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop")
)
panel = (df.groupby(["end_date", "decile"])["forward_return"].mean()
           .unstack("decile").sort_index())

# Top-3: mean of deciles 7,8,9 (D8,D9,D10).  Bot-3: mean of deciles 0,1,2.
panel["TOP3"] = panel[[7, 8, 9]].mean(axis=1)
panel["BOT3"] = panel[[0, 1, 2]].mean(axis=1)
panel["LS3"]  = panel["TOP3"] - panel["BOT3"]
panel["LS10"] = panel[9] - panel[0]

# Non-overlapping monthly sample.
monthly = panel.iloc[::20].copy()

def ann_stats(r: pd.Series, horizon_days=20):
    r = r.dropna()
    m = r.mean()
    s = r.std()
    n = len(r)
    ann_ret = m * (252/horizon_days)
    ann_vol = s * np.sqrt(252/horizon_days)
    sharpe = ann_ret/ann_vol if ann_vol > 0 else np.nan
    # compounded cumulative
    cum = (1 + r).cumprod().iloc[-1]
    # compounded annualised
    years = n * horizon_days / 252
    ann_comp = cum ** (1/years) - 1 if years > 0 else np.nan
    # t-stat (iid SE; monthly sample ≈ non-overlapping)
    t = m / (s/np.sqrt(n)) if s > 0 else np.nan
    return {"n": n, "cum_x": cum, "ann_comp": ann_comp*100,
            "ann_arith": ann_ret*100, "ann_vol": ann_vol*100,
            "sharpe": sharpe, "t": t}

def newey_west_t(x, lag=19):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    n = len(x); mu = x.mean(); xc = x - mu
    g0 = (xc**2).mean(); s = g0
    for k in range(1, lag+1):
        gk = (xc[k:] * xc[:-k]).mean()
        w = 1 - k/(lag+1); s += 2*w*gk
    se = np.sqrt(s/n); return mu/se if se > 0 else np.nan

print("                      TOP3        BOT3        LS (T3-B3)    LS (D10-D1)")
for key in ["cum_x", "ann_comp", "ann_arith", "ann_vol", "sharpe", "t"]:
    row = []
    for name in ["TOP3", "BOT3", "LS3", "LS10"]:
        v = ann_stats(monthly[name])[key]
        row.append(f"{v:>10.3f}")
    print(f"  {key:<16}  {row[0]}  {row[1]}  {row[2]}  {row[3]}")

# Newey-West t on daily overlapping series.
nwt3  = newey_west_t(panel["LS3"].values,  lag=19)
nwt10 = newey_west_t(panel["LS10"].values, lag=19)
print(f"  Newey-West t(19) on overlapping daily series:  LS3 = {nwt3:+.2f}   LS10 = {nwt10:+.2f}")
print()

# Year-by-year for LS3.
panel["year"] = panel.index.year
by_year = panel.groupby("year")["LS3"].agg(["count","mean","std"])
by_year["ann_pct"] = by_year["mean"]*252/20*100
by_year["pct_pos"] = panel.groupby("year")["LS3"].apply(lambda x: (x>0).mean()*100)
print("Year-by-year LS3 (top-3 − bot-3):")
print(by_year.round(3).to_string())
print()

# --- Plot ---
cum = (1 + monthly[["TOP3","BOT3","LS3","LS10"]]).cumprod()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(cum.index, cum["TOP3"], color="#2e7d32", lw=1.8, label="TOP-3 (D8+D9+D10)")
ax1.plot(cum.index, cum["BOT3"], color="#c62828", lw=1.8, label="BOT-3 (D1+D2+D3)")
ax1.set_yscale("log")
ax1.set_ylabel("Cumulative value (log, $1 start)")
ax1.set_title("Top-3 vs Bottom-3 decile portfolios — expanding pathway, 1999 → 2026\n"
              "(equal-weight across 30% of universe each side, 20-day non-overlap)")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc="upper left")

ax2.plot(cum.index, cum["LS3"],  color="black",      lw=1.8, label="LS T3−B3 (top/bot 30%)")
ax2.plot(cum.index, cum["LS10"], color="tab:blue", lw=1.2, alpha=0.75, label="LS D10−D1 (top/bot 10%)")
ax2.axhline(1.0, color="grey", lw=0.7, ls="--")
ax2.set_yscale("log")
ax2.set_ylabel("LS cumulative (log)")
ax2.set_xlabel("End date")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc="upper left")

# annotate final values
last = cum.iloc[-1]
for col, color in [("TOP3","#2e7d32"),("BOT3","#c62828")]:
    ax1.annotate(f"{last[col]:.2f}×", (cum.index[-1], last[col]),
                 fontsize=9, ha="left", va="center", color=color)
for col, color in [("LS3","black"),("LS10","tab:blue")]:
    ax2.annotate(f"{last[col]:.2f}×", (cum.index[-1], last[col]),
                 fontsize=9, ha="left", va="center", color=color)

plt.tight_layout()
out_pdf = RUN / "top3_vs_bot3.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_pdf.with_suffix(".png"), dpi=140, bbox_inches="tight")
print(f"wrote {out_pdf}")
