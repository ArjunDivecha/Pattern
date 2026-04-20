"""Cumulative-return chart, all 10 deciles + long-short, expanding pathway."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

RUN = Path("runs/expanding/20260419_174908_cdef6809")
df = pd.read_parquet(RUN / "predictions.parquet")

# Assign cross-sectional deciles per end_date.
df["decile"] = df.groupby("end_date")["p_up_mean"].transform(
    lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop")
)

# Mean forward_return by (end_date, decile); equal-weight inside each decile.
panel = (df.groupby(["end_date", "decile"])["forward_return"].mean()
           .unstack("decile").sort_index())

# Sample every 20 trading days for non-overlapping compounding.
monthly = panel.iloc[::20].copy()

# Long-short (decile 10 − decile 1, i.e., col 9 − col 0).
monthly["LS"] = monthly[9] - monthly[0]

# Cumulative compound (1 + r).cumprod().  LS compounds its own return so it
# stays positive and works on a log y-axis.
cum = (1 + monthly).cumprod()

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                               gridspec_kw={"height_ratios": [2.3, 1]})

# Top: deciles (log scale).
cmap = plt.cm.RdYlGn
for d in range(10):
    col = cmap(d / 9)
    ax1.plot(cum.index, cum[d], color=col, lw=1.4,
             label=f"D{d+1}" + ("  (short)" if d == 0 else "  (long)" if d == 9 else ""))
ax1.set_yscale("log")
ax1.set_ylabel("Cumulative value (log, $1 start)")
ax1.set_title("Cumulative return by p_up_mean decile — expanding pathway, 1999-03 → 2026-03\n"
              "(equal-weight within decile, 20-day non-overlapping compounding)")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc="upper left", ncol=2, fontsize=9)

# Bottom: long-short (compounded, log scale).
ax2.plot(cum.index, cum["LS"], color="black", lw=1.8, label="LS = D10 − D1 (compounded)")
ax2.axhline(1.0, color="grey", lw=0.7, ls="--")
ax2.set_yscale("log")
ax2.set_ylabel("LS cum. value (log, $1 start)")
ax2.set_xlabel("End date")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc="upper left")

# Annotate final values.
last = cum.iloc[-1]
for d in range(10):
    ax1.annotate(f"{last[d]:.1f}×", (cum.index[-1], last[d]),
                 fontsize=8, ha="left", va="center", color=cmap(d/9))
ax2.annotate(f"{last['LS']:.1f}×", (cum.index[-1], last["LS"]),
             fontsize=9, ha="left", va="center", color="black")

plt.tight_layout()
out_pdf = Path("runs/expanding/20260419_174908_cdef6809/decile_cumulative.pdf")
plt.savefig(out_pdf, bbox_inches="tight")
out_png = out_pdf.with_suffix(".png")
plt.savefig(out_png, dpi=140, bbox_inches="tight")
plt.close()

print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
print()
print("Final cumulative values (20d-compounded since 1999-03):")
for d in range(10):
    ann = (last[d] ** (1/(len(monthly)*20/252)) - 1) * 100
    print(f"  D{d+1:>2}: {last[d]:6.2f}×  ann={ann:+.2f}%")
ls_ann_comp = (last["LS"] ** (1/(len(monthly)*20/252)) - 1) * 100
print(f"  LS  : {last['LS']:6.2f}×  ann={ls_ann_comp:+.2f}%")
