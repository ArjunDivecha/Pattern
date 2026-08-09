"""
=============================================================================
SCRIPT NAME: rlcc_international.py
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/Australia/*.csv
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/UK500/*.csv
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/Japan/*.csv
      per-ticker daily bars: timestamps, open, high, low, close, volume, amount

OUTPUT FILES:
- {out_dir}/international_results.xlsx   per-market headline + month-end split
- {out_dir}/international_results.json

DESCRIPTION:
OUT-OF-SAMPLE REPLICATION of the `rlcc` month-end close-location signal
(README addenda L.11, L.12) on three developed markets never touched by this
review: Australia, the UK, and Japan.

WHY THIS REPLACES THE SINGLE NKY LOCKBOX SHOT. The lockbox reserved one test on
Japan. But a lone Japanese failure would have been uninterpretable — Japan's
month-end structure genuinely differs (March fiscal year-end, different index
review calendars, a domestic institutional base), so failure could mean "the
effect is not real" OR "Japanese month-end flows differ." Three independent
markets convert an ambiguous single test into a pattern that can be read
either way: consistent success is strong evidence, consistent failure is a
clean refutation, and a split result localizes the effect to a market
structure.

PRE-COMMITTED BEFORE RUNNING (no parameter is fitted on these markets):

  Signal      EXACTLY the R1000-swept optimum, transplanted unchanged:
              clv = (C - L) / (H - L)
              x   = clv normalized to [0,1] over a trailing 62-day window
              sig = -( mean(x, last 2 bars) - mean(x, prior 60 bars) )
  Formation   last trading day of each calendar month
  Entry       next trading day's OPEN  (the fair entry from L.11)
  Hold        20 trading days, open to open
  Book        decile long/short, equal weight
  Universe    exclude the cheapest 5% by price cross-sectionally each date
              (a currency-agnostic stand-in for the R1000 $5 floor)
  Window      2005+, matching the primary window of PREREGISTRATION.md §3
  Costs       reported gross and net at 8bp and 25bp half-spread

  SUCCESS CRITERION, fixed in advance: the effect replicates if it is positive
  with t >= 2.0 in at least TWO of the three markets. Anything less is a
  failure to replicate, and is to be reported as such.

  The month-end concentration (L.12: 6.5x at month-end vs other days on R1000)
  is tested separately per market, since that is the actual mechanism claim.

Survivorship caveat: these panels appear to be built from current index
membership, so the long leg is likely flattered. A cross-sectional long/short
is far less exposed to this than a long-only book — both legs are drawn from
the same survivor set — but the absolute levels should be read as optimistic.

USAGE:
  python3 scripts/rlcc_international.py --out-dir runs/experiments/rlcc_intl
=============================================================================
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

KRONOS = Path("/Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/"
              "shiyu-coder-Kronos/data")
MARKETS = ["Australia", "UK500", "Japan"]

N_LAST, LOOKBACK = 2, 60           # the R1000-swept optimum, NOT refitted here
WIN = N_LAST + LOOKBACK
HOLD = 20
PRICE_PCT = 0.05
SINCE = "2005-01-01"
MIN_NAMES = 40                     # need a usable decile


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


def load_market(name: str) -> pd.DataFrame:
    frames = []
    for f in glob.glob(str(KRONOS / name / "*.csv")):
        if "manifest" in f.lower():
            continue
        try:
            d = pd.read_csv(f, usecols=["timestamps", "open", "high", "low",
                                        "close", "volume"])
        except Exception:
            continue
        if len(d) < 300:
            continue
        d["ticker"] = Path(f).stem
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    o = pd.concat(frames, ignore_index=True)
    o["date"] = pd.to_datetime(o["timestamps"]).dt.tz_localize(None).dt.normalize()
    o = (o.drop(columns=["timestamps"])
           .sort_values(["ticker", "date"])
           .drop_duplicates(["ticker", "date"])
           .reset_index(drop=True))
    return o


def build(o: pd.DataFrame) -> pd.DataFrame:
    g = o["ticker"]
    rng = (o["high"] - o["low"]).where(lambda s: s > 0)
    clv = (o["close"] - o["low"]) / rng
    lo = clv.groupby(g).transform(lambda s: s.rolling(WIN, min_periods=WIN // 2).min())
    hi = clv.groupby(g).transform(lambda s: s.rolling(WIN, min_periods=WIN // 2).max())
    x = (clv - lo) / (hi - lo).replace(0, np.nan)
    rec = x.groupby(g).transform(lambda s: s.rolling(N_LAST, min_periods=1).mean())
    pri = x.groupby(g).transform(
        lambda s: s.rolling(LOOKBACK, min_periods=LOOKBACK // 2).mean().shift(N_LAST))
    o = o.copy()
    o["sig"] = -(rec - pri)
    return o


def ls_series(o: pd.DataFrame, month_end_only: bool):
    """Decile LS, next-open entry, HOLD days, open-to-open."""
    P = o.pivot_table(index="date", columns="ticker", values="open", aggfunc="last").sort_index()
    C = o.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    S = o.pivot_table(index="date", columns="ticker", values="sig", aggfunc="last").sort_index()
    L = np.log(P.where(P > 0))
    fwd = L.shift(-(HOLD + 1)) - L.shift(-1)
    floor = C.quantile(PRICE_PCT, axis=1)
    ok = S.notna() & fwd.notna() & C.ge(floor, axis=0)
    Sm = S.where(ok)
    idx = pd.DatetimeIndex(Sm.index)
    if month_end_only:
        me = pd.Series(idx).groupby([idx.year, idx.month]).transform("max").values
        Sm = Sm[idx == me]
        fwd = fwd.loc[Sm.index]
    r = Sm.rank(axis=1, pct=True)
    lon = (r > 0.9) & Sm.notna()
    sho = (r <= 0.1) & Sm.notna()
    n = Sm.notna().sum(axis=1)
    wl = lon.div(lon.sum(axis=1).replace(0, np.nan), axis=0)
    ws = sho.div(sho.sum(axis=1).replace(0, np.nan), axis=0)
    ls = ((wl - ws) * np.expm1(fwd)).sum(axis=1).where(n >= MIN_NAMES)
    tk = {d: (frozenset(lon.columns[lon.loc[d]]), frozenset(sho.columns[sho.loc[d]]))
          for d in Sm.index if n.get(d, 0) >= MIN_NAMES}
    return ls.dropna(), tk, n


def turnover(tk, order):
    v, prev = [], None
    for d in order:
        if d not in tk:
            continue
        cur = tk[d]
        if prev:
            v.append(np.mean([len(cur[i] ^ prev[i]) / max(len(cur[i]) + len(prev[i]), 1)
                              for i in (0, 1)]))
        prev = cur
    return float(np.mean(v)) if v else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("PRE-COMMITTED: R1000-swept params transplanted unchanged "
          f"(n_last={N_LAST}, lookback={LOOKBACK}, hold={HOLD}, next-open entry).")
    print("SUCCESS = positive with t >= 2.0 in at least 2 of 3 markets.\n")

    head, mech = [], []
    for m in MARKETS:
        o = load_market(m)
        if o.empty:
            print(f"  {m}: no data"); continue
        o = build(o)
        o = o[o["date"] >= pd.Timestamp(SINCE)]
        ls, tk, n = ls_series(o, month_end_only=True)
        if len(ls) < 24:
            print(f"  {m}: too few months ({len(ls)})"); continue
        to = turnover(tk, list(ls.index))
        gross = ((1 + ls).prod() ** (12 / len(ls)) - 1) * 100
        rec = {"market": m, "names": int(o.ticker.nunique()),
               "months": len(ls), "mean_univ": float(n[n >= MIN_NAMES].mean()),
               "gross_pct": round(gross, 2),
               "sharpe": round(float((ls.mean() * 12) / (ls.std() * np.sqrt(12))), 3),
               "t": round(nwt(ls.to_numpy()), 2), "turnover": round(to, 3)}
        for bp in (8, 25):
            rec[f"net@{bp}bp"] = round(gross - 12 * 2 * to * bp / 1e4 * 100, 2)
        rec["PASS"] = "YES" if (rec["t"] >= 2.0 and rec["gross_pct"] > 0) else "no"
        head.append(rec)

        # mechanism: month-end vs all other formation days
        all_ls, _, _ = ls_series(o, month_end_only=False)
        i = pd.DatetimeIndex(all_ls.index)
        me = pd.Series(i).groupby([i.year, i.month]).transform("max").values
        a, b = all_ls[i == me], all_ls[i != me]
        mech.append({"market": m,
                     "month_end_mean_pct": round(float(a.mean() * 100), 3),
                     "month_end_n": len(a), "month_end_t": round(nwt(a.to_numpy()), 2),
                     "other_days_mean_pct": round(float(b.mean() * 100), 3),
                     "other_days_n": len(b), "other_days_t": round(nwt(b.to_numpy()), 2),
                     "ratio": (round(float(a.mean() / b.mean()), 2)
                               if b.mean() not in (0, np.nan) else None)})
        print(f"  {m} done  [{time.time()-t0:.0f}s]")

    H, M = pd.DataFrame(head), pd.DataFrame(mech)
    print("\n=== OUT-OF-SAMPLE REPLICATION — month-end rlcc, 2005+, decile LS ===\n")
    print(H.to_string(index=False))
    print("\n=== MECHANISM — month-end vs all other formation days ===\n")
    print(M.to_string(index=False))
    npass = int((H.PASS == "YES").sum()) if len(H) else 0
    print(f"\n  markets passing (positive, t>=2.0): {npass} of {len(H)}")
    print(f"  PRE-COMMITTED VERDICT: "
          f"{'REPLICATES' if npass >= 2 else 'FAILS TO REPLICATE'}")

    with pd.ExcelWriter(args.out_dir / "international_results.xlsx", engine="openpyxl") as xw:
        H.to_excel(xw, sheet_name="headline", index=False)
        M.to_excel(xw, sheet_name="month_end_mechanism", index=False)
    (args.out_dir / "international_results.json").write_text(json.dumps(
        {"headline": H.to_dict("records"), "mechanism": M.to_dict("records"),
         "n_pass": npass, "verdict": "REPLICATES" if npass >= 2 else "FAILS",
         "params": {"n_last": N_LAST, "lookback": LOOKBACK, "hold": HOLD,
                    "entry": "next open", "since": SINCE}}, indent=2, default=str))
    print(f"\nwrote {args.out_dir/'international_results.xlsx'}")


if __name__ == "__main__":
    main()
