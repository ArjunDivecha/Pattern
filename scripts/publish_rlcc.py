"""
=============================================================================
SCRIPT NAME: publish_rlcc.py
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_ohlcv_spliced.parquet
      Bloomberg history + yfinance forward, built by splice_daily_update.py

OUTPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/tracker/rlcc-top500/nav_daily.csv
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/tracker/rlcc-top500/holdings_latest.csv
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/tracker/rlcc-top500/meta.json

DESCRIPTION:
Publishes the `rlcc` close-location reversal book for Tracker.

THE STRATEGY. At each month-end, rank the 500 most liquid US names by where
they closed inside that day's high-low range relative to a trailing baseline.
Buy the decile that closed LOWEST in range, short the decile that closed
HIGHEST, enter at the NEXT MORNING'S OPEN, hold 20 trading days.

    clv  = (Close - Low) / (High - Low)
    x    = clv normalized to [0,1] over a trailing 21-day window
    sig  = -( x_last - mean(x, prior 20 bars) )        higher = buy

Economic story: a stock pushed to the bottom of its daily range by an impatient
or forced seller tends to revert as liquidity returns. The book is paid for
supplying that liquidity.

DESIGN CHOICES, and why:

- ORIGINAL parameters (n_last=1, lookback=20), NOT the sweep optimum
  (n_last=2, lookback=60). The swept version scores better on this panel, but
  it was selected on this panel; the original was not. For a forward track the
  un-selected parameters are the honest ones, and the sweep showed a broad
  plateau rather than a peak, so little is given up.
- TOP 500 BY DOLLAR VOLUME. The full universe backtests better (+8.06% vs
  +6.84% gross) but concentrates in small illiquid names. Top-500 is the
  capacity-realistic version.
- NEXT-OPEN ENTRY. The signal uses the day-t close, so entry at that close is
  not achievable; the next open is. Entering at the t+1 CLOSE instead would
  discard day t+1, which is where most of the return accrues.
- $5 price floor, decile long/short, equal weight, gross of costs
  (house convention).

MEASURED, 2005+ (219 months, decile LS, next-open entry):
    gross +6.84%/yr | Sharpe 0.59 | NW t 2.24 | turnover 0.90 one-sided/month
    net @8bp +5.10% | net @15bp +3.59%

HONEST STATUS -- carried in meta.json flags so it travels with the data.
This signal FAILED its out-of-sample replication (README L.13): tested on
Australia, the UK and Japan with parameters transplanted unchanged, it passed
in 1 of 3 markets against a pre-committed bar of 2 of 3, and the month-end
concentration that motivated it inverted in two of them. It is tracked as PAPER
precisely because forward evidence is the only clean test left -- every
backtest number on this panel has been through many hands.

USAGE:
  python3 scripts/publish_rlcc.py
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OHLCV = ROOT / "data" / "r1000_ohlcv_spliced.parquet"
OUT = ROOT / "data" / "tracker" / "rlcc-top500"

N_LAST, LOOKBACK = 1, 20
TOP_N = 500
MIN_PRICE = 5.0
HOLD = 20
DECILE = 0.1
START = "1999-01-01"


def build_signal(o: pd.DataFrame) -> pd.DataFrame:
    g = o["Ticker"]
    rng = (o["High"] - o["Low"]).where(lambda s: s > 0)
    clv = (o["Close"] - o["Low"]) / rng
    w = N_LAST + LOOKBACK
    lo = clv.groupby(g).transform(lambda s: s.rolling(w, min_periods=w // 2).min())
    hi = clv.groupby(g).transform(lambda s: s.rolling(w, min_periods=w // 2).max())
    x = (clv - lo) / (hi - lo).replace(0, np.nan)
    rec = x.groupby(g).transform(lambda s: s.rolling(N_LAST, min_periods=1).mean())
    pri = x.groupby(g).transform(
        lambda s: s.rolling(LOOKBACK, min_periods=LOOKBACK // 2).mean().shift(N_LAST))
    o["sig"] = -(rec - pri)
    o["dv60"] = (o["Close"] * o["Volume"]).groupby(g).transform(
        lambda s: s.rolling(60, min_periods=15).mean())
    return o


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".csv.tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlcv", type=Path, default=OHLCV)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    args = ap.parse_args()

    print("Loading …")
    o = pd.read_parquet(args.ohlcv, columns=["Ticker", "Date", "Open", "High",
                                             "Low", "Close", "AdjClose", "Volume"])
    o["Date"] = pd.to_datetime(o["Date"])
    o = o.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    o["adj"] = o["AdjClose"] / o["Close"].replace(0, np.nan)
    o["AdjOpen"] = o["Open"] * o["adj"]
    o = build_signal(o)
    o = o[o["Date"] >= pd.Timestamp(START)]

    P = o.pivot_table(index="Date", columns="Ticker", values="AdjOpen", aggfunc="last").sort_index()
    S = o.pivot_table(index="Date", columns="Ticker", values="sig", aggfunc="last").sort_index()
    C = o.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last").sort_index()
    DV = o.pivot_table(index="Date", columns="Ticker", values="dv60", aggfunc="last").sort_index()
    # pivot_table drops all-NaN columns, so a name with a signal but no usable
    # AdjOpen survives in S and vanishes from P. Selecting it into a leg then
    # raises on the mark. Intersect once, here, rather than guarding every use.
    common = P.columns.intersection(S.columns).intersection(
        C.columns).intersection(DV.columns)
    P, S, C, DV = P[common], S[common], C[common], DV[common]
    cal = P.index
    # month-end formation dates
    ser = pd.Series(cal)
    me = ser.groupby([ser.dt.year, ser.dt.month]).transform("max")
    forms = [d for d in pd.DatetimeIndex(sorted(set(me))) if d in S.index]
    print(f"  {len(forms)} month-end formation dates  {forms[0].date()} -> {forms[-1].date()}")

    rows, hold_rows, prev = [], None, None
    nav, bench = 100.0, 100.0
    for i, d in enumerate(forms):
        s, c, dv = S.loc[d], C.loc[d], DV.loc[d]
        m = s.notna() & c.ge(MIN_PRICE) & dv.notna()
        if m.sum() < 100:
            continue
        keep = dv[m].nlargest(min(args.top_n, int(m.sum()))).index
        s2 = s[keep].dropna()
        if len(s2) < 100:
            continue
        q = s2.rank(pct=True, method="first")
        L = list(s2.index[q > 1 - DECILE]); Sh = list(s2.index[q <= DECILE])
        if len(L) < 5 or len(Sh) < 5:
            continue
        # entry at next open; exit at the open HOLD days later (or next formation)
        j = cal.searchsorted(d) + 1
        if j >= len(cal):
            break
        entry_d = cal[j]
        exit_i = min(j + HOLD, len(cal) - 1)
        nxt = forms[i + 1] if i + 1 < len(forms) else None
        if nxt is not None:
            exit_i = min(exit_i, cal.searchsorted(nxt) + 1)
        p0 = P.loc[entry_d]
        turn = np.nan
        cur = (frozenset(L), frozenset(Sh))
        if prev:
            turn = np.mean([len(cur[k] ^ prev[k]) / max(len(cur[k]) + len(prev[k]), 1)
                            for k in (0, 1)])
        prev = cur
        # daily marks across the hold
        for k in range(j, exit_i + 1):
            dd = cal[k]
            rel = (P.loc[dd] / p0) - 1.0
            rl, rs = rel[L].mean(skipna=True), rel[Sh].mean(skipna=True)
            rew = rel[s2.index].mean(skipna=True)
            if not np.isfinite(rl) or not np.isfinite(rs):
                continue
            rows.append({"date": dd, "nav": nav * (1 + (rl - rs)),
                         "ew_univ": bench * (1 + (rew if np.isfinite(rew) else 0)),
                         "leg_long": rl, "leg_short": rs,
                         "n_long": len(L), "n_short": len(Sh),
                         "turnover": turn, "form_date": d})
        if rows:
            nav, bench = rows[-1]["nav"], rows[-1]["ew_univ"]
        hold_rows = (d, entry_d, L, Sh, p0, s2)

    nav_df = pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date")
    nav_df["ret_1d_pct"] = nav_df["nav"].pct_change(fill_method=None) * 100
    nav_df["cum_ret_pct"] = (nav_df["nav"] / 100 - 1) * 100
    nav_df["bench_cum_ret_pct"] = (nav_df["ew_univ"] / 100 - 1) * 100
    nav_df["n_positions"] = nav_df["n_long"] + nav_df["n_short"]
    nav_df = nav_df[["date", "nav", "ew_univ", "ret_1d_pct", "cum_ret_pct",
                     "bench_cum_ret_pct", "n_positions", "n_long", "n_short",
                     "leg_long", "leg_short", "turnover", "form_date"]]
    atomic_csv(nav_df, args.out_dir / "nav_daily.csv")

    fd, entry_d, L, Sh, p0, s2 = hold_rows
    last = nav_df["date"].max()
    pt = P.loc[last]
    hold = pd.DataFrame([
        {"ticker": t.replace(" US Equity", ""), "side": side,
         "weight": (1.0 / len(L) if side == "long" else -1.0 / len(Sh)),
         "rlcc_score": float(s2.get(t, np.nan)),
         "entry_date": str(entry_d.date()), "px_entry": float(p0.get(t, np.nan)),
         "px_last": float(pt.get(t, np.nan)),
         "ret_since_entry": float(pt.get(t, np.nan) / p0.get(t, np.nan) - 1)
         if np.isfinite(p0.get(t, np.nan)) else np.nan}
        for side, names in (("long", L), ("short", Sh)) for t in names])
    hold = hold.sort_values(["side", "rlcc_score"], ascending=[True, False])
    atomic_csv(hold, args.out_dir / "holdings_latest.csv")

    r = nav_df["nav"].pct_change(fill_method=None).dropna()
    yrs = (nav_df["date"].max() - nav_df["date"].min()).days / 365.25
    def win(since):
        w = nav_df[nav_df["date"] >= pd.Timestamp(since)]
        if len(w) < 60:
            return None
        wr = w["nav"].pct_change(fill_method=None).dropna()
        y = (w["date"].max() - w["date"].min()).days / 365.25
        return {"cagr_pct": round(float(((w["nav"].iloc[-1] / w["nav"].iloc[0]) ** (1 / y) - 1) * 100), 2),
                "sharpe": round(float(wr.mean() / wr.std() * np.sqrt(252)), 3), "days": len(w)}

    meta = {
        "sleeve": "rlcc-top500",
        "name": "rlcc close-location reversal — top 500 US",
        "as_of": str(nav_df["date"].max().date()),
        "series_start": str(nav_df["date"].min().date()),
        "last_rebalance": str(fd.date()),
        "entry_date": str(entry_d.date()),
        "n_long": len(L), "n_short": len(Sh), "n_positions": len(L) + len(Sh),
        "nav": float(nav_df["nav"].iloc[-1]),
        "universe": f"top {args.top_n} US by 60d dollar volume, price >= ${MIN_PRICE:.0f}",
        "params": {"n_last": N_LAST, "lookback": LOOKBACK, "hold_days": HOLD,
                   "entry": "next-morning open", "decile": DECILE},
        "benchmark": "cash (dollar-neutral); ew_univ column is the cell's beta",
        "cagr_pct": round(float(((nav_df["nav"].iloc[-1] / 100) ** (1 / yrs) - 1) * 100), 2),
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3),
        "mean_turnover": round(float(nav_df["turnover"].mean(skipna=True)), 3),
        "since_2005": win("2005-01-01"), "since_2020": win("2020-01-01"),
        "flags": [
            "gross_of_costs",
            "nav_is_index_not_dollars",
            "dollar_neutral_long_short_requires_single_stock_shorting",
            "FAILED_out_of_sample_replication_1_of_3_markets_see_README_L13",
            "month_end_concentration_did_NOT_replicate_internationally",
            "original_params_used_deliberately_swept_params_were_selected_on_this_panel",
            "backtest_leg_is_not_a_track_record_forward_evidence_starts_now",
        ],
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"  {nav_df['date'].min().date()} -> {nav_df['date'].max().date()}  "
          f"({len(nav_df):,} days)")
    print(f"  NAV {meta['nav']:.2f}  CAGR {meta['cagr_pct']:+.2f}%  "
          f"Sharpe {meta['sharpe']:+.3f}  turnover {meta['mean_turnover']:.3f}")
    print(f"  since 2005: {meta['since_2005']}")
    print(f"  since 2020: {meta['since_2020']}")
    print(f"  book: {len(L)} long / {len(Sh)} short, formed {fd.date()}, entered {entry_d.date()}")
    print(f"  wrote {args.out_dir}/")


if __name__ == "__main__":
    main()
