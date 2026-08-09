"""
=============================================================================
SCRIPT NAME: rlcc_sweep.py
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_ohlcv_spliced.parquet
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/expanding/20260419_174908_cdef6809/predictions_monthly.parquet
      used only for its (ticker, end_date) monthly formation grid

OUTPUT FILES:
- {out_dir}/rlcc_sweep.parquet     one row per (variant, split) with full stats
- {out_dir}/rlcc_sweep.xlsx        same, plus the stability ranking
- {out_dir}/sweep_meta.json

DESCRIPTION:
Parameter sweep for the `rlcc` close-location signal (README addendum L.11),
which at the fair entry (next-morning open) is the strongest implementable
signal found in this review: +6.18%/yr net, t 3.41, 2005+.

Before spending the NKY lockbox on it, we need to know whether that result sits
on a broad plateau or a needle. A signal that works only at one exact parameter
setting is almost always a fluke.

THE SIGNAL, generalized:

    clv_t        = where the close sits in the day's range (3 definitions)
    signal_t     = mean(clv over last n bars) - mean(clv over the L bars before)
    oriented     = -signal        (buy stocks that closed LOW in their range)

Swept:
    clv_def   mid | hl | oc      -- (C-(H+L)/2)/(H-L) | (C-L)/(H-L) | (C-O)/(H-L)
    n_last    1, 2, 3, 5         -- how many recent bars form the signal
    lookback  5, 10, 20, 60      -- the comparison baseline length
    norm      raw | z | n01      -- normalization of clv within the window
                                    (n01 reproduces the original construction)

ANTI-OVERFITTING DESIGN. Every variant is reported on a TIME SPLIT:
    TUNE     2005-01 .. 2015-12   (~132 months)
    VALIDATE 2016-01 .. 2026-03   (~123 months)
The sweep is judged on whether a region of the parameter space works in BOTH
halves, not on the best single number. A variant that tops the tune half and
dies in the validate half is noise, and the sweep is designed to make that
visible rather than hide it. The NKY lockbox stays sealed throughout.

Entry is the NEXT-MORNING OPEN throughout -- the realistic implementation
established in L.11. Entering at the t+1 close would delete day t+1, which is
where this signal's return accrues.

USAGE:
  python3 scripts/rlcc_sweep.py --out-dir runs/experiments/rlcc_sweep
=============================================================================
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OHLCV = ROOT / "data" / "r1000_ohlcv_spliced.parquet"
GRID = ROOT / "runs" / "expanding" / "20260419_174908_cdef6809" / "predictions_monthly.parquet"

MIN_PRICE = 5.0
HOLD = 20
TUNE = ("2005-01-01", "2015-12-31")
VALID = ("2016-01-01", "2030-01-01")

CLV_DEFS = ["mid", "hl", "oc"]
N_LASTS = [1, 2, 3, 5]
LOOKBACKS = [5, 10, 20, 60]
NORMS = ["raw", "z", "n01"]


def nwt(x, lag=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan
    mu = x.mean(); xc = x - mu
    s = (xc ** 2).mean()
    for k in range(1, lag + 1):
        if k >= n:
            break
        s += 2 * (1 - k / (lag + 1)) * (xc[k:] * xc[:-k]).mean()
    return float(mu / np.sqrt(max(s, 0) / n))


def load_panel():
    o = pd.read_parquet(OHLCV, columns=["Ticker", "Date", "Open", "High", "Low",
                                        "Close", "AdjClose", "Volume"])
    o["Date"] = pd.to_datetime(o["Date"])
    o = o.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    o["adj"] = o["AdjClose"] / o["Close"].replace(0, np.nan)
    o["AdjOpen"] = o["Open"] * o["adj"]
    return o


def clv_series(o: pd.DataFrame, how: str) -> pd.Series:
    rng = (o["High"] - o["Low"]).where(lambda s: s > 0)
    if how == "mid":
        return (o["Close"] - 0.5 * (o["High"] + o["Low"])) / rng
    if how == "hl":
        return (o["Close"] - o["Low"]) / rng
    return (o["Close"] - o["Open"]) / rng


def build_signal(o, g, clv, n_last, lookback, norm):
    """mean(clv, last n) - mean(clv, the L bars before that), optionally normalized."""
    w = n_last + lookback
    if norm == "z":
        m = clv.groupby(g).transform(lambda s: s.rolling(w, min_periods=w // 2).mean())
        sd = clv.groupby(g).transform(lambda s: s.rolling(w, min_periods=w // 2).std())
        x = (clv - m) / sd.replace(0, np.nan)
    elif norm == "n01":
        lo = clv.groupby(g).transform(lambda s: s.rolling(w, min_periods=w // 2).min())
        hi = clv.groupby(g).transform(lambda s: s.rolling(w, min_periods=w // 2).max())
        x = (clv - lo) / (hi - lo).replace(0, np.nan)
    else:
        x = clv
    recent = x.groupby(g).transform(lambda s: s.rolling(n_last, min_periods=1).mean())
    prior = x.groupby(g).transform(
        lambda s: s.rolling(lookback, min_periods=max(2, lookback // 2)).mean().shift(n_last))
    return -(recent - prior)          # oriented: low close-in-range = buy


def evaluate(df, col, lo, hi):
    """Decile LS on next-open entry, net @8bp, over [lo, hi]."""
    x = df[(df.end_date >= pd.Timestamp(lo)) & (df.end_date <= pd.Timestamp(hi))]
    x = x.dropna(subset=[col, "fwd"])
    rows, prev = [], None
    for dt, g in x.groupby("end_date"):
        if len(g) < 50:
            continue
        q = g[col].rank(pct=True, method="first")
        t_, b_ = g[q > 0.9], g[q <= 0.1]
        if len(t_) < 2 or len(b_) < 2:
            continue
        cur = (frozenset(t_.ticker), frozenset(b_.ticker))
        to = np.nan
        if prev:
            to = np.mean([len(cur[i] ^ prev[i]) / max(len(cur[i]) + len(prev[i]), 1)
                          for i in (0, 1)])
        prev = cur
        rows.append({"r": np.expm1(t_.fwd).mean() - np.expm1(b_.fwd).mean(), "to": to})
    if len(rows) < 24:
        return None
    d = pd.DataFrame(rows)
    r, to = d["r"], d["to"].mean()
    drag = 2 * to * 8 / 1e4
    return {"months": len(r),
            "gross_pct": float(((1 + r).prod() ** (12 / len(r)) - 1) * 100),
            "net_pct": float(((1 + r - drag).prod() ** (12 / len(r)) - 1) * 100),
            "sharpe": float((r.mean() * 12) / (r.std() * np.sqrt(12))),
            "t": nwt(r.to_numpy()), "turnover": float(to)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-price", type=float, default=MIN_PRICE)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("loading panel …")
    o = load_panel()
    g = o["Ticker"]

    # next-open entry, 20 trading days held (open to open)
    po = o.pivot_table(index="Date", columns="Ticker", values="AdjOpen", aggfunc="last").sort_index()
    lo_ = np.log(po.where(po > 0))
    fwd = (lo_.shift(-(HOLD + 1)) - lo_.shift(-1)).stack().rename("fwd").reset_index()
    fwd.columns = ["end_date", "ticker", "fwd"]

    grid = pd.read_parquet(GRID, columns=["ticker", "end_date"])
    grid["end_date"] = pd.to_datetime(grid["end_date"])

    px = o[["Ticker", "Date", "Close"]].rename(columns={"Ticker": "ticker", "Date": "end_date"})
    base = grid.merge(px, on=["ticker", "end_date"], how="left")
    base = base[base["Close"] >= args.min_price].merge(fwd, on=["ticker", "end_date"], how="left")
    base = base.dropna(subset=["fwd"])
    print(f"  grid {len(base):,} rows  {base.end_date.nunique()} months")

    combos = list(itertools.product(CLV_DEFS, N_LASTS, LOOKBACKS, NORMS))
    print(f"  sweeping {len(combos)} variants …")

    clv_cache = {c: clv_series(o, c) for c in CLV_DEFS}
    out = []
    for i, (cdef, n_last, lb, norm) in enumerate(combos, 1):
        sig = build_signal(o, g, clv_cache[cdef], n_last, lb, norm)
        s = pd.DataFrame({"ticker": o["Ticker"], "end_date": o["Date"], "sig": sig})
        d = base.merge(s, on=["ticker", "end_date"], how="left")
        rec = {"clv_def": cdef, "n_last": n_last, "lookback": lb, "norm": norm}
        for split, (a, b) in (("tune", TUNE), ("validate", VALID)):
            r = evaluate(d, "sig", a, b)
            if r is None:
                rec = None
                break
            for k, v in r.items():
                rec[f"{split}_{k}"] = v
        if rec:
            out.append(rec)
        if i % 12 == 0:
            print(f"    {i}/{len(combos)}  [{time.time()-t0:.0f}s]")

    res = pd.DataFrame(out)
    # stability: the weaker of the two halves, so a variant must work in BOTH
    res["min_net"] = res[["tune_net_pct", "validate_net_pct"]].min(axis=1)
    res["min_t"] = res[["tune_t", "validate_t"]].min(axis=1)
    res = res.sort_values("min_net", ascending=False)
    res.to_parquet(args.out_dir / "rlcc_sweep.parquet", index=False)
    with pd.ExcelWriter(args.out_dir / "rlcc_sweep.xlsx", engine="openpyxl") as xw:
        res.to_excel(xw, sheet_name="sweep", index=False)

    cols = ["clv_def", "n_last", "lookback", "norm", "tune_net_pct", "tune_t",
            "validate_net_pct", "validate_t", "min_net", "min_t"]
    print(f"\n=== TOP 15 by WORST-half net (both halves must work) ===")
    print(res[cols].head(15).round(2).to_string(index=False))
    print(f"\n=== BOTTOM 5 ===")
    print(res[cols].tail(5).round(2).to_string(index=False))

    (args.out_dir / "sweep_meta.json").write_text(json.dumps(
        {"variants": len(res), "tune": TUNE, "validate": VALID, "hold_days": HOLD,
         "entry": "next-morning open", "runtime_s": round(time.time() - t0, 1)}, indent=2))
    print(f"\nwrote {args.out_dir/'rlcc_sweep.xlsx'}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
