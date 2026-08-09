"""
=============================================================================
SCRIPT NAME: rlcc_full_stats.py
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/tracker/rlcc-top500/nav_daily.csv
      published by scripts/publish_rlcc.py
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/tracker/rlcc-top500/meta.json

OUTPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/rlcc_stats/rlcc_full_stats.xlsx
      sheets: by_period, by_year, drawdowns, cost_sensitivity, distribution, monthly
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/rlcc_stats/rlcc_stats.pdf / .png

DESCRIPTION:
Full performance statistics for the rlcc top-500 close-location reversal book:
returns, volatility, drawdowns, turnover, cost sensitivity, distribution and
market exposure.

Conventions, consistent with the rest of this project:
- Returns are SIMPLE (portfolio) arithmetic, never mean-log (README addendum K).
- The book is DOLLAR-NEUTRAL, so the benchmark is cash. The `ew_univ` column is
  the universe's own beta, carried for context and used here only to measure
  correlation and beta -- never as a benchmark to compute excess against.
- Gross of costs and of borrow; costs are shown as an explicit sensitivity
  rather than folded into a single headline.
- Drawdowns are computed on the daily NAV, so they reflect intra-hold marks
  rather than month-end snapshots.

USAGE:
  python3 scripts/rlcc_full_stats.py
=============================================================================
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "tracker" / "rlcc-top500"
OUT = ROOT / "runs" / "experiments" / "rlcc_stats"

PERIODS = [("Full (1999+)", None), ("2005+", "2005-01-01"),
           ("2015+", "2015-01-01"), ("2020+", "2020-01-01")]
COST_BP = [0, 2, 5, 8, 15, 25]


def max_dd(nav: pd.Series):
    peak = nav.cummax()
    dd = nav / peak - 1.0
    trough = dd.idxmin()
    pk = nav.loc[:trough].idxmax()
    rec = nav.loc[trough:][nav.loc[trough:] >= nav.loc[pk]]
    rec_date = rec.index[0] if len(rec) else None
    return (float(dd.min() * 100), pk, trough, rec_date,
            (trough - pk).days, ((rec_date - trough).days if rec_date is not None else None))


def top_drawdowns(nav: pd.Series, k: int = 5) -> pd.DataFrame:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    rows, s = [], dd.copy()
    for _ in range(k):
        if s.min() >= -1e-9:
            break
        tr = s.idxmin()
        pk = nav.loc[:tr].idxmax()
        after = nav.loc[tr:]
        rec = after[after >= nav.loc[pk]]
        rd = rec.index[0] if len(rec) else None
        rows.append({"depth_pct": round(float(s.loc[tr] * 100), 2),
                     "peak": pk.date(), "trough": tr.date(),
                     "recovered": rd.date() if rd is not None else "not yet",
                     "decline_days": (tr - pk).days,
                     "recovery_days": (rd - tr).days if rd is not None else None})
        end = rd if rd is not None else s.index[-1]
        s.loc[pk:end] = 0.0
    return pd.DataFrame(rows)


def period_stats(nav: pd.Series, turn_m: float, bench: pd.Series | None) -> dict:
    r = nav.pct_change(fill_method=None).dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
    vol = r.std() * np.sqrt(252) * 100
    dn = r[r < 0].std() * np.sqrt(252) * 100
    dd, pk, tr, rd, ddays, rdays = max_dd(nav)
    m = nav.resample("ME").last().pct_change(fill_method=None).dropna()
    out = {
        "years": round(yrs, 1), "CAGR_%": round(cagr, 2),
        "vol_%": round(vol, 2), "sharpe": round(cagr / vol, 3) if vol else None,
        "sortino": round(cagr / dn, 3) if dn else None,
        "max_dd_%": round(dd, 2), "calmar": round(cagr / abs(dd), 3) if dd else None,
        "dd_peak": str(pk.date()), "dd_trough": str(tr.date()),
        "dd_recovery_days": rdays,
        "best_month_%": round(m.max() * 100, 2), "worst_month_%": round(m.min() * 100, 2),
        "hit_rate_monthly_%": round((m > 0).mean() * 100, 1),
        "skew_monthly": round(float(m.skew()), 2),
        "turnover_1sided_mo": round(turn_m, 3),
    }
    if bench is not None and len(bench) > 60:
        b = bench.reindex(r.index).pct_change(fill_method=None)
        j = pd.concat([r.rename("y"), b.rename("x")], axis=1).dropna()
        if len(j) > 60 and j.x.std() > 0:
            out["beta_to_universe"] = round(float(np.polyfit(j.x, j.y, 1)[0]), 3)
            out["corr_to_universe"] = round(float(j.y.corr(j.x)), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.src / "nav_daily.csv", parse_dates=["date"]).set_index("date")
    meta = json.loads((args.src / "meta.json").read_text())
    nav, bench = d["nav"], d["ew_univ"]
    turn_m = float(d["turnover"].dropna().mean())

    # ── by period ────────────────────────────────────────────────────────────
    rows = []
    for lbl, since in PERIODS:
        n = nav if since is None else nav[nav.index >= pd.Timestamp(since)]
        b = bench if since is None else bench[bench.index >= pd.Timestamp(since)]
        if len(n) < 260:
            continue
        rows.append({"period": lbl, **period_stats(n, turn_m, b)})
    per = pd.DataFrame(rows)

    # ── by year ──────────────────────────────────────────────────────────────
    yr = []
    for y, gg in nav.groupby(nav.index.year):
        if len(gg) < 60:
            continue
        prev = nav[nav.index < f"{y}-01-01"]
        start = prev.iloc[-1] if len(prev) else gg.iloc[0]
        rr = gg.pct_change(fill_method=None).dropna()
        dd, *_ = max_dd(gg)
        yr.append({"year": int(y), "return_%": round((gg.iloc[-1] / start - 1) * 100, 2),
                   "vol_%": round(rr.std() * np.sqrt(252) * 100, 2),
                   "max_dd_%": round(dd, 2), "days": len(gg)})
    yrs = pd.DataFrame(yr)

    # ── cost sensitivity ─────────────────────────────────────────────────────
    cs = []
    for lbl, since in PERIODS:
        n = nav if since is None else nav[nav.index >= pd.Timestamp(since)]
        if len(n) < 260:
            continue
        y = (n.index[-1] - n.index[0]).days / 365.25
        g = ((n.iloc[-1] / n.iloc[0]) ** (1 / y) - 1) * 100
        row = {"period": lbl, "gross_%": round(g, 2)}
        for bp in COST_BP:
            row[f"net@{bp}bp"] = round(g - 12 * 2 * turn_m * bp / 1e4 * 100, 2)
        cs.append(row)
    cost = pd.DataFrame(cs)

    dds = top_drawdowns(nav)
    mret = nav.resample("ME").last().pct_change(fill_method=None).dropna()
    dist = pd.DataFrame([{
        "n_months": len(mret), "mean_%": round(mret.mean() * 100, 3),
        "median_%": round(mret.median() * 100, 3), "std_%": round(mret.std() * 100, 3),
        "skew": round(float(mret.skew()), 3), "kurtosis": round(float(mret.kurt()), 3),
        "pct_positive": round((mret > 0).mean() * 100, 1),
        "worst_%": round(mret.min() * 100, 2), "best_%": round(mret.max() * 100, 2),
        "p5_%": round(mret.quantile(.05) * 100, 2), "p95_%": round(mret.quantile(.95) * 100, 2),
    }])

    print("=== BY PERIOD ===")
    print(per.to_string(index=False))
    print("\n=== COST SENSITIVITY (annualized, %) ===")
    print(cost.to_string(index=False))
    print("\n=== 5 WORST DRAWDOWNS ===")
    print(dds.to_string(index=False))
    print("\n=== MONTHLY DISTRIBUTION ===")
    print(dist.to_string(index=False))
    print("\n=== BY YEAR ===")
    print(yrs.to_string(index=False))

    with pd.ExcelWriter(args.out_dir / "rlcc_full_stats.xlsx", engine="openpyxl") as xw:
        per.to_excel(xw, sheet_name="by_period", index=False)
        yrs.to_excel(xw, sheet_name="by_year", index=False)
        dds.to_excel(xw, sheet_name="drawdowns", index=False)
        cost.to_excel(xw, sheet_name="cost_sensitivity", index=False)
        dist.to_excel(xw, sheet_name="distribution", index=False)
        d.reset_index().to_excel(xw, sheet_name="nav_daily", index=False)

    # ── chart ────────────────────────────────────────────────────────────────
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "font.size": 9})
    fig, ax = plt.subplots(3, 1, figsize=(13, 10),
                           gridspec_kw={"height_ratios": [3, 1.3, 1.3]}, sharex=True)
    ax[0].plot(nav.index, nav, color="#1a5490", lw=1.6, label="rlcc top-500 L/S")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of $100 (log)")
    ax[0].set_title("rlcc close-location reversal — top 500 US, decile L/S, next-open entry, gross",
                    fontsize=12, loc="left")
    ax[0].grid(True, which="both", alpha=0.25); ax[0].legend(loc="upper left", fontsize=9)
    dd = (nav / nav.cummax() - 1) * 100
    ax[1].fill_between(dd.index, dd, 0, color="#c0392b", alpha=0.55)
    ax[1].set_ylabel("Drawdown (%)"); ax[1].grid(True, alpha=0.25)
    roll = nav.pct_change(fill_method=None).rolling(252).mean() / \
        nav.pct_change(fill_method=None).rolling(252).std() * np.sqrt(252)
    ax[2].plot(roll.index, roll, color="#148f77", lw=1.2)
    ax[2].axhline(0, color="black", lw=0.6); ax[2].axhline(1, color="#888", lw=0.6, ls=":")
    ax[2].set_ylabel("Rolling 1y Sharpe"); ax[2].grid(True, alpha=0.25)
    ax[2].xaxis.set_major_locator(mdates.YearLocator(2))
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(0.5, 0.005, f"Gross of costs and borrow. Turnover {turn_m:.1%} one-sided/month. "
             f"Dollar-neutral, so the benchmark is cash. {meta.get('universe','')}",
             ha="center", fontsize=8, color="#555")
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    for ext in ("pdf", "png"):
        plt.savefig(args.out_dir / f"rlcc_stats.{ext}", dpi=150, bbox_inches="tight")
    print(f"\nwrote {args.out_dir/'rlcc_full_stats.xlsx'}")
    print(f"wrote {args.out_dir/'rlcc_stats.pdf'}")


if __name__ == "__main__":
    main()
