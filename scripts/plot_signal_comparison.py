"""
=============================================================================
SCRIPT NAME: plot_signal_comparison.py
=============================================================================

INPUT FILES (all produced earlier in the Stage 0/1 review):
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/stage1_eval/p_up_mean/signal_card.xlsx
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/stage1_eval/rev_5d_oriented/signal_card.xlsx
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/stage1b_eval/combo_resid_wstr/signal_card.xlsx
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/stage1c_eval/resid_all/signal_card.xlsx
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/stage1c_eval/resid_all_size_sector/signal_card.xlsx
  (each: sheet `floored_monthly` for the monthly book returns,
          signal_card.json for mean one-sided turnover)

OUTPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/charts/signal_comparison.pdf
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/charts/signal_comparison.png
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/charts/neutralization_ladder.pdf
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/charts/neutralization_ladder.png

DESCRIPTION:
Cumulative-return charts for the Stage 0/1 kill-or-confirm review (README
addenda L.3-L.5). Plots the decile long/short book of each ranking signal.

Deliberate choices, all of which the project has been bitten by before:
- SIMPLE (portfolio) returns and a $5 price floor -- the headline convention
  from addenda J and K. The old mean-log numbers are not plotted at all.
- BOTH gross and net panels. Every signal here turns over ~89% of the book per
  month, so gross alone is misleading; net uses a flat
  `2 x turnover x half_spread` monthly drag at 8bp.
- LOG y-axis, because a 27-year compounded series is unreadable linearly.
- Light mode, per house convention.

USAGE:
  python3 scripts/plot_signal_comparison.py
=============================================================================
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "runs" / "experiments"
OUT = EXP / "charts"

HALF_SPREAD_BP = 8.0

# (key, directory, display label, colour, linewidth, linestyle)
SIGNALS = [
    ("p_up_mean",            EXP / "stage1_eval"  / "p_up_mean",
     "CNN p_up — as deployed",              "#c0392b", 2.2, "-"),
    ("rev_5d_oriented",      EXP / "stage1_eval"  / "rev_5d_oriented",
     "Weekly reversal (WSTR)",              "#7f8c8d", 1.8, "--"),
    ("resid_all",            EXP / "stage1c_eval" / "resid_all",
     "CNN neutralized (baselines)",         "#2471a3", 2.2, "-"),
    ("resid_all_size_sector", EXP / "stage1c_eval" / "resid_all_size_sector",
     "CNN neutralized (+size+sector)",      "#148f77", 2.2, "-"),
    ("combo_resid_wstr",     EXP / "stage1b_eval" / "combo_resid_wstr",
     "Combo: neutralized CNN + WSTR",       "#b7950b", 2.0, "-."),
]


def load(dirpath: Path) -> tuple[pd.DataFrame, float] | None:
    xls, js = dirpath / "signal_card.xlsx", dirpath / "signal_card.json"
    if not xls.exists():
        return None
    d = pd.read_excel(xls, sheet_name="floored_monthly")
    d["end_date"] = pd.to_datetime(d["end_date"])
    turn = json.loads(js.read_text())["variants"]["floored"]["turnover_one_sided"]
    return d.sort_values("end_date"), float(turn or 0.0)


def cum(series: pd.Series, drag: float = 0.0) -> pd.Series:
    return (1 + series - drag).cumprod()


def panel(ax, data, col, since, drag_bp, title, ylab):
    for key, _, label, colour, lw, ls in SIGNALS:
        if key not in data:
            continue
        d, turn = data[key]
        if since:
            d = d[d.end_date >= pd.Timestamp(since)]
        if len(d) < 6:
            continue
        drag = 2 * turn * (drag_bp / 1e4) if drag_bp else 0.0
        c = cum(d[col], drag)
        # A blown-up book (a month <= -100%) has no meaningful compounded path;
        # truncate at the blow-up rather than plotting a sign-flipped fantasy.
        bad = (1 + d[col] - drag) <= 0
        if bad.any():
            first = int(np.argmax(bad.to_numpy()))
            c = c.iloc[:first]
            d = d.iloc[:first]
            label += "  [book blew up]"
        ax.plot(d.end_date.iloc[:len(c)], c, color=colour, lw=lw, ls=ls,
                label=f"{label}   ×{c.iloc[-1]:.2f}" if len(c) else label)
    ax.axhline(1.0, color="black", lw=0.6, alpha=0.5)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})

    data = {}
    for key, path, *_ in SIGNALS:
        r = load(path)
        if r:
            data[key] = r
        else:
            print(f"  [skip] {key}: no signal_card.xlsx at {path}")

    # ── Figure 1: gross vs net, full sample and since 2020 ──────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    panel(axes[0][0], data, "ls_decile", None, 0,
          "A. Full sample 1999–2026 — GROSS", "Growth of $1 (log)")
    panel(axes[0][1], data, "ls_decile", None, HALF_SPREAD_BP,
          f"B. Full sample — NET of {HALF_SPREAD_BP:.0f}bp half-spread", "")
    panel(axes[1][0], data, "ls_decile", "2020-01-01", 0,
          "C. Since 2020 — GROSS", "Growth of $1 (log)")
    panel(axes[1][1], data, "ls_decile", "2020-01-01", HALF_SPREAD_BP,
          f"D. Since 2020 — NET of {HALF_SPREAD_BP:.0f}bp", "")
    fig.suptitle("Pattern CNN — decile long/short, portfolio (simple) returns, $5 price floor",
                 fontsize=13, y=0.985)
    fig.text(0.5, 0.005,
             "Gross of borrow. Turnover ~89%/month one-sided for every signal. "
             "Panels C/D are the same books restricted to the last 63 months.",
             ha="center", fontsize=8, color="#555555")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    for ext in ("pdf", "png"):
        plt.savefig(OUT / f"signal_comparison.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'signal_comparison.pdf'}")

    # ── Figure 2: the neutralization ladder ─────────────────────────────────
    ladder = [
        ("p_up_mean",             EXP / "stage1_eval"  / "p_up_mean",
         "raw p_up",                    "#c0392b"),
        ("resid_vol",             EXP / "stage1b_eval" / "resid_vol",
         "− vol tilt",                  "#e59866"),
        ("resid_all",             EXP / "stage1c_eval" / "resid_all",
         "− all 4 baselines",           "#2471a3"),
        ("resid_all_size",        EXP / "stage1c_eval" / "resid_all_size",
         "− baselines − size",          "#7d3c98"),
        ("resid_all_size_sector", EXP / "stage1c_eval" / "resid_all_size_sector",
         "− baselines − size − sector", "#148f77"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, since, ttl in ((axes[0], None, "Full sample 1999–2026"),
                           (axes[1], "2020-01-01", "Since 2020")):
        for key, path, label, colour in ladder:
            r = load(path)
            if not r:
                continue
            d, _ = r
            if since:
                d = d[d.end_date >= pd.Timestamp(since)]
            if len(d) < 6:
                continue
            c = cum(d["ls_decile"])
            ax.plot(d.end_date, c, color=colour, lw=2.0,
                    label=f"{label}   ×{c.iloc[-1]:.2f}")
        ax.axhline(1.0, color="black", lw=0.6, alpha=0.5)
        ax.set_yscale("log")
        ax.set_title(ttl, fontsize=11, loc="left")
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
        ax.xaxis.set_major_locator(mdates.YearLocator(3 if since is None else 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_ylabel("Growth of $1 (log)", fontsize=9)
    fig.suptitle("Stripping the CNN's factor tilts, one at a time  (gross, decile LS)",
                 fontsize=13, y=0.99)
    fig.text(0.5, 0.005,
             "Full sample: the edge survives every control. Since 2020: it does not "
             "survive size+sector — but no variant reaches t=2 in that window either way.",
             ha="center", fontsize=8, color="#555555")
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    for ext in ("pdf", "png"):
        plt.savefig(OUT / f"neutralization_ladder.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'neutralization_ladder.pdf'}")


if __name__ == "__main__":
    main()
