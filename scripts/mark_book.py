"""
=============================================================================
SCRIPT NAME: mark_book.py
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_ohlcv_spliced.parquet
      Daily OHLCV + AdjClose (total-return basis).  Built by splice_daily_update.py.
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_tradeable_universe.csv
      Staleness manifest.  Names with tradeable=False are DROPPED -- a stale
      price silently poisons dv_60d / vol_60d / mom_12_1 and can put a name in
      the wrong tertile.
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/expanding/20260419_174908_cdef6809/predictions_monthly.parquet
      BACKTEST leg.  Per-year expanding-window ensembles, through 2026-03-19.
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/live/live_predictions.parquet
      PAPER leg.  Window-27 ensemble applied to recent month-ends.

OUTPUT FILES (one directory per sleeve, under data/tracker/<sleeve>/):
- nav_daily.csv          date, nav, ew_cell, leg_long, leg_short, n_long, n_short, track
                         Column shape matches what FDT publishes, so the sleeve
                         drops into Tracker's STRATEGY_PERF with no template change.
- holdings_latest.csv    ticker, side, weight, p_up_mean, entry_date, px_entry, px_last, ret_since_entry
- meta.json              as_of, paper_start, n_positions, counts, flags

DESCRIPTION:
Builds and marks the Pattern CNN long/short book so Tracker can read a daily
NAV.  Tracker is read-only and never runs this -- Pattern publishes, Tracker
consumes.

Two sleeves, both 50/50 dollar-neutral long/short inside a cross-sectional cell:

  pattern-trash    Low dollar-volume x High vol x Low momentum
                   The alpha.  ~90-125 names, ~25 bp half-spread, hard to
                   borrow, capacity $500k-$1M per side.
  pattern-megacap  High dollar-volume x High vol
                   ~126-147 names, ~2.5 bp half-spread, easy borrow, deepest
                   capacity.  Best cost-adjusted cell since 2020 (Sharpe 1.23)
                   but did NOT work full-sample -- a lead, not a result.

MECHANIC.  At each rebalance date the cell is selected, ranked by p_up_mean,
split 50/50 long/short, and each leg equal-weighted.  Those names are then held
fixed until the next rebalance and marked EVERY DAY, so intra-period
performance is visible instead of only appearing at month end.  Leg return is
measured from the rebalance close on an AdjClose (total-return) basis:

    leg_ret(t) = mean_j( P_j(t) / P_j(rebal) - 1 )
    LS(t)      = leg_long(t) - leg_short(t)
    NAV(t)     = NAV(rebal) * (1 + LS(t))

Benchmark is the equal-weight return of the sleeve's OWN cell universe, carried
as a column in the same file.  Same reasoning as FDT publishing its ew34 column:
identical calendar, identical price source, no join gaps, and the excess line
then isolates CNN selection skill from the cell's beta.

A $5 PRICE FLOOR is applied at selection.  Sub-penny bankruptcy shells sort
into the small/high-vol/loser cell by construction and a one-tick move there is
a +4,800% return; that inflated the published backtest roughly 2x.  See
README addendum J.

Returns are GROSS of costs, per house convention.

RETURN CONVENTION -- WHY THIS SERIES IS LOWER THAN THE PUBLISHED BACKTEST.
The backtest reports LS as the difference of MEAN LOG returns across names
(`forward_return` in the predictions file is a 20-trading-day LOG return,
verified: median |log(P_t+20/P_t) - forward_return| = 0.00000).  An
equal-weighted portfolio does not earn the mean log return -- it earns the mean
SIMPLE return.  Mean(log) sits below log(1+mean(simple)) by roughly sigma^2/2,
and because the short leg is SUBTRACTED its dispersion penalty is ADDED to the
reported spread.  The trash tier's bottom leg has a cross-sectional SD of 0.178
per 20-day period, i.e. a sigma^2/2 term of ~1.58%/month.

Measured on the trash tier with the $5 floor, 282 months:

    LOG convention (as published)      +11.29%/yr
    SIMPLE convention (a portfolio)     +4.15%/yr
    difference                          +7.14pp   <- an averaging artifact

This script deliberately uses real portfolio arithmetic -- actual prices, held
shares, simple returns -- so its NAV is what a book would have earned, not the
cross-sectional log statistic.  It reproduces the +4.15% independently (+4.24%
including the daily-marking calendar), which is the cross-check that the
implementation is right and the convention was the problem.

A related consequence: with NO price floor the simple convention produces
months worse than -100% (shorting a name that 10x's loses 900%), so the naive
unhedged decile book is not a computable portfolio at all.  The log convention
hid that tail.  The $5 floor is what makes this series well-defined.

THE BACKTEST/PAPER SPLIT.  Everything up to the last backtest date is BACKTEST
and everything after is PAPER.  The `track` column records which, meta.json
carries `paper_start`, and the two must stay distinguishable on any page that
renders this series -- the backtest leg has already been revised once (README J)
and must not quietly become the track record.

USAGE:
  python3 scripts/mark_book.py                      # both sleeves, full history
  python3 scripts/mark_book.py --sleeve pattern-trash
  python3 scripts/mark_book.py --start 2020-01-01   # shorter backfill

NOTES:
- Idempotent: rewrites the whole series from source each run, atomically.
- Requires: pandas, pyarrow, numpy.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

OHLCV = ROOT / "data" / "r1000_ohlcv_spliced.parquet"
UNIVERSE = ROOT / "data" / "r1000_tradeable_universe.csv"
PRED_BT = ROOT / "runs" / "expanding" / "20260419_174908_cdef6809" / "predictions_monthly.parquet"
PRED_LIVE = ROOT / "runs" / "live" / "live_predictions.parquet"
OUT_ROOT = ROOT / "data" / "tracker"

MIN_PRICE = 5.0

SLEEVES = {
    "pattern-trash": {
        "name": "Pattern CNN — trash tier",
        "cell": {"dv_b": "Low", "vol_b": "High", "mom_b": "Low"},
    },
    "pattern-megacap": {
        "name": "Pattern CNN — mega-cap high-vol",
        "cell": {"dv_b": "High", "vol_b": "High"},
    },
}


# ── features ─────────────────────────────────────────────────────────────────

def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv[["Ticker", "Date", "Close", "Volume", "AdjClose"]].copy()
    df = df.sort_values(["Ticker", "Date"])
    g = df.groupby("Ticker")
    df["dv"] = df["Close"] * df["Volume"]
    df["dv_60d"] = g["dv"].transform(lambda s: s.rolling(60, min_periods=15).mean())
    df["logp"] = np.log(df["AdjClose"].replace(0, np.nan))
    df["mom_12_1"] = g["logp"].shift(21) - g["logp"].shift(252 + 21)
    df["ret"] = g["AdjClose"].pct_change(fill_method=None)
    df["vol_60d"] = g["ret"].transform(
        lambda s: s.rolling(60, min_periods=20).std() * np.sqrt(252))
    return (df[["Ticker", "Date", "Close", "dv_60d", "mom_12_1", "vol_60d"]]
            .rename(columns={"Ticker": "ticker", "Date": "end_date"}))


def bucket3(df: pd.DataFrame, col: str) -> pd.Series:
    def assign(g):
        if g[col].notna().sum() < 3:
            return pd.Series([pd.NA] * len(g), index=g.index)
        return pd.qcut(g[col].rank(method="first"), 3,
                       labels=["Low", "Mid", "High"], duplicates="drop")
    return df.groupby("end_date", group_keys=False).apply(assign)


# ── book construction + daily marking ────────────────────────────────────────

def build_sleeve(preds: pd.DataFrame, cell: dict, px: pd.DataFrame,
                 trading_days: np.ndarray, bt_end: pd.Timestamp):
    """Return (nav_daily DataFrame, holdings_latest DataFrame)."""
    # NOTE: the price floor is applied UPSTREAM, before bucketing, to match
    # backtest_liquid_grid.py --min-price.  Filtering after bucketing would
    # leave the tertile boundaries set by names that are then discarded, giving
    # a different (and smaller) cell than the backtest reports.
    sel = preds
    for k, v in cell.items():
        sel = sel[sel[k] == v]

    rebals = sorted(sel["end_date"].unique())
    if not rebals:
        return pd.DataFrame(), pd.DataFrame()

    # score_live.py emits the CURRENT day as the canonical "month-end" of the
    # running month, so on any mid-month run the newest prediction date is not a
    # real rebalance.  Left in, the book would re-form every single day the job
    # runs -- daily turnover on a monthly strategy.  Drop prediction dates that
    # fall in the latest (incomplete) calendar month of the price data; the
    # month's book stays the one set at the last completed month-end.
    if len(trading_days):
        cur = pd.Timestamp(trading_days.max()).to_period("M")
        dropped = [d for d in rebals if pd.Timestamp(d).to_period("M") == cur]
        rebals = [d for d in rebals if pd.Timestamp(d).to_period("M") != cur]
        if dropped and not rebals:
            rebals = sorted(sel["end_date"].unique())[:1]
    if not rebals:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    nav = 100.0
    bench = 100.0
    last_book = None

    for i, rb in enumerate(rebals):
        g = sel[sel["end_date"] == rb]
        if len(g) < 4:
            continue
        r = g["p_up_mean"].rank(method="first") / len(g)
        longs = g.loc[r > 0.5, "ticker"].tolist()
        shorts = g.loc[r <= 0.5, "ticker"].tolist()
        allnames = longs + shorts
        if len(longs) < 2 or len(shorts) < 2:
            continue

        # hold window: (rb, next_rebal]  -- marked every trading day
        end = rebals[i + 1] if i + 1 < len(rebals) else trading_days.max()
        days = trading_days[(trading_days > rb) & (trading_days <= end)]
        if len(days) == 0:
            continue

        try:
            p0 = px.loc[rb, allnames]
        except KeyError:
            continue
        p0 = p0[p0.notna() & (p0 > 0)]
        L = [t for t in longs if t in p0.index]
        S = [t for t in shorts if t in p0.index]
        if len(L) < 2 or len(S) < 2:
            continue

        nav_start, bench_start = nav, bench
        for d in days:
            if d not in px.index:
                continue
            pt = px.loc[d, p0.index]
            rel = (pt / p0) - 1.0
            rl = rel[L].mean(skipna=True)
            rs = rel[S].mean(skipna=True)
            rew = rel.mean(skipna=True)
            if not np.isfinite(rl) or not np.isfinite(rs):
                continue
            ls = rl - rs
            nav = nav_start * (1.0 + ls)
            bench = bench_start * (1.0 + (rew if np.isfinite(rew) else 0.0))
            rows.append({
                "date": d, "nav": nav, "ew_cell": bench,
                "leg_long": rl, "leg_short": rs,
                "n_long": len(L), "n_short": len(S),
                "track": "backtest" if d <= bt_end else "paper",
            })

        last_book = {"rebal": rb, "L": L, "S": S, "p0": p0,
                     "scores": g.set_index("ticker")["p_up_mean"]}

    nav_df = pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date")
    if not nav_df.empty:
        # Kronos-compatible convenience columns so Tracker's adapter stays thin.
        nav_df["ret_1d_pct"] = nav_df["nav"].pct_change(fill_method=None) * 100
        nav_df["cum_ret_pct"] = (nav_df["nav"] / 100.0 - 1.0) * 100
        nav_df["bench_cum_ret_pct"] = (nav_df["ew_cell"] / 100.0 - 1.0) * 100
        nav_df["n_positions"] = nav_df["n_long"] + nav_df["n_short"]
        nav_df = nav_df[["date", "nav", "ew_cell", "ret_1d_pct", "cum_ret_pct",
                         "bench_cum_ret_pct", "n_positions", "n_long", "n_short",
                         "leg_long", "leg_short", "track"]]

    # holdings as of the most recent rebalance, marked at the last close
    hold = pd.DataFrame()
    if last_book is not None:
        d_last = nav_df["date"].max()
        pt = px.loc[d_last, last_book["p0"].index]
        recs = []
        for side, names in (("long", last_book["L"]), ("short", last_book["S"])):
            w = (1.0 / len(names)) * (1 if side == "long" else -1)
            for t in names:
                recs.append({
                    "ticker": t, "side": side, "weight": w,
                    "p_up_mean": float(last_book["scores"].get(t, np.nan)),
                    "entry_date": pd.Timestamp(last_book["rebal"]).date(),
                    "px_entry": float(last_book["p0"][t]),
                    "px_last": float(pt.get(t, np.nan)),
                })
        hold = pd.DataFrame(recs)
        hold["ret_since_entry"] = hold["px_last"] / hold["px_entry"] - 1.0
        hold = hold.sort_values(["side", "p_up_mean"], ascending=[True, False])

    return nav_df, hold


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".csv.tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlcv", type=Path, default=OHLCV)
    ap.add_argument("--universe", type=Path, default=UNIVERSE)
    ap.add_argument("--pred-backtest", type=Path, default=PRED_BT)
    ap.add_argument("--pred-live", type=Path, default=PRED_LIVE)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--sleeve", default="all", choices=["all", *SLEEVES])
    ap.add_argument("--start", default="1999-01-01")
    args = ap.parse_args()

    print("Loading OHLCV …")
    o = pd.read_parquet(args.ohlcv)
    o["Date"] = pd.to_datetime(o["Date"])

    uni = pd.read_csv(args.universe)
    tradeable = set(uni.loc[uni["tradeable"].astype(bool), "Ticker"])
    n_stale = int((~uni["tradeable"].astype(bool)).sum())
    print(f"  tradeable universe: {len(tradeable):,}  (stale excluded: {n_stale})")

    print("Computing features …")
    feats = compute_features(o)

    print("Loading predictions …")
    bt = pd.read_parquet(args.pred_backtest)[["ticker", "end_date", "p_up_mean"]]
    bt["end_date"] = pd.to_datetime(bt["end_date"])
    bt_end = bt["end_date"].max()
    lv = pd.read_parquet(args.pred_live)[["ticker", "end_date", "p_up_mean"]]
    lv["end_date"] = pd.to_datetime(lv["end_date"])
    lv = lv[lv["end_date"] > bt_end]           # paper leg starts where backtest ends
    paper_start = lv["end_date"].min() if len(lv) else None
    preds = pd.concat([bt, lv], ignore_index=True)
    print(f"  backtest through {bt_end.date()}  ({len(bt):,} rows)")
    print(f"  paper from       {paper_start.date() if paper_start is not None else '--'}"
          f"  ({len(lv):,} rows)")

    # The tradeable manifest is a TODAY snapshot, so it must be applied ONLY to
    # the forward/paper leg.  Applying it to history would be survivorship bias
    # of the worst kind here: it deletes exactly the names that cratered and
    # delisted, which is where the SHORT book earns most of its return.
    # Measured: filtering history cut the trash-tier CAGR from +11.9% to +3.0%.
    before = len(preds)
    preds = preds[(preds["end_date"] <= bt_end) | preds["ticker"].isin(tradeable)]
    print(f"  tradeable filter applied to paper leg only: {before:,} -> {len(preds):,}")
    preds = preds[preds["end_date"] >= pd.Timestamp(args.start)]
    preds = preds.merge(feats, on=["ticker", "end_date"], how="left")
    preds = preds.dropna(subset=["dv_60d", "vol_60d", "mom_12_1", "Close"])

    # Price floor BEFORE bucketing -- same order as backtest_liquid_grid.py
    # --min-price, so the tertile boundaries match the published cell.
    before = len(preds)
    preds = preds[preds["Close"] >= MIN_PRICE]
    print(f"  price floor ${MIN_PRICE:.2f}: {before:,} -> {len(preds):,} "
          f"({len(preds)/before*100:.1f}% kept)")

    print("Bucketing …")
    preds["dv_b"] = bucket3(preds, "dv_60d")
    preds["vol_b"] = bucket3(preds, "vol_60d")
    preds["mom_b"] = bucket3(preds, "mom_12_1")
    preds = preds[preds[["dv_b", "vol_b", "mom_b"]].notna().all(axis=1)]
    print(f"  usable rows: {len(preds):,}  "
          f"{preds['end_date'].min().date()} -> {preds['end_date'].max().date()}")

    print("Pivoting prices …")
    need = set(preds["ticker"])
    op = o[o["Ticker"].isin(need) & (o["Date"] >= pd.Timestamp(args.start))]
    px = op.pivot_table(index="Date", columns="Ticker", values="AdjClose",
                        aggfunc="last").sort_index()
    trading_days = px.index.to_numpy()
    print(f"  price matrix: {px.shape[0]:,} days x {px.shape[1]:,} tickers")

    targets = list(SLEEVES) if args.sleeve == "all" else [args.sleeve]
    summary = {}
    for sid in targets:
        spec = SLEEVES[sid]
        print(f"\n=== {sid} — {spec['name']} ===")
        nav_df, hold = build_sleeve(preds, spec["cell"], px, trading_days, bt_end)
        if nav_df.empty:
            print("  no data — skipped")
            continue

        dest = args.out_root / sid
        atomic_write_csv(nav_df, dest / "nav_daily.csv")
        if not hold.empty:
            atomic_write_csv(hold, dest / "holdings_latest.csv")

        r = nav_df["nav"].pct_change(fill_method=None).dropna()
        yrs = (nav_df["date"].max() - nav_df["date"].min()).days / 365.25
        cagr = ((nav_df["nav"].iloc[-1] / 100.0) ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan
        sharpe = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
        paper = nav_df[nav_df["track"] == "paper"]

        def window_stats(since: str) -> dict | None:
            w = nav_df[nav_df["date"] >= pd.Timestamp(since)]
            if len(w) < 30:
                return None
            wr = w["nav"].pct_change(fill_method=None).dropna()
            wy = (w["date"].max() - w["date"].min()).days / 365.25
            growth = w["nav"].iloc[-1] / w["nav"].iloc[0]
            exc = (w["nav"].iloc[-1] / w["nav"].iloc[0]) - (w["ew_cell"].iloc[-1] / w["ew_cell"].iloc[0])
            return {
                "cagr_pct": round(float((growth ** (1 / wy) - 1) * 100), 3) if wy > 0 else None,
                "sharpe": round(float((wr.mean() * 252) / (wr.std() * np.sqrt(252))), 3)
                          if wr.std() > 0 else None,
                "vs_ew_cell_total_pct": round(float(exc * 100), 2),
                "days": int(len(w)),
            }

        meta = {
            "sleeve": sid,
            "name": spec["name"],
            "cell": spec["cell"],
            "as_of": str(nav_df["date"].max().date()),
            "series_start": str(nav_df["date"].min().date()),
            "backtest_through": str(bt_end.date()),
            "paper_start": (str(paper["date"].min().date()) if len(paper) else None),
            "paper_days": int(len(paper)),
            "n_long": int(nav_df["n_long"].iloc[-1]),
            "n_short": int(nav_df["n_short"].iloc[-1]),
            "n_positions": int(nav_df["n_long"].iloc[-1] + nav_df["n_short"].iloc[-1]),
            "nav": float(nav_df["nav"].iloc[-1]),
            "benchmark": "EW cell universe",
            "min_price": MIN_PRICE,
            "stale_excluded": n_stale,
            "cagr_pct": None if not np.isfinite(cagr) else round(float(cagr), 3),
            "sharpe": None if not np.isfinite(sharpe) else round(float(sharpe), 3),
            "since_2020": window_stats("2020-01-01"),
            "since_2015": window_stats("2015-01-01"),
            "flags": [
                "gross_of_costs",
                "nav_is_index_not_dollars",
                "simple_return_portfolio_arithmetic_NOT_the_mean_log_convention_"
                "used_in_README_backtests_see_mark_book_docstring",
                "long_short_requires_single_stock_shorting",
                "price_floor_5usd_applied_see_README_addendum_J",
                "backtest_leg_before_paper_start_not_a_track_record",
            ],
            "generated": datetime.now().isoformat(timespec="seconds"),
        }
        (dest / "meta.json").write_text(json.dumps(meta, indent=2))

        print(f"  {nav_df['date'].min().date()} -> {nav_df['date'].max().date()}  "
              f"({len(nav_df):,} days)")
        print(f"  NAV {nav_df['nav'].iloc[-1]:.4f}   CAGR {cagr:+.2f}%   Sharpe {sharpe:+.2f}")
        print(f"  paper leg: {len(paper)} days from "
              f"{paper['date'].min().date() if len(paper) else '--'}")
        print(f"  book: {meta['n_long']} long / {meta['n_short']} short")
        print(f"  wrote {dest / 'nav_daily.csv'}")
        summary[sid] = meta

    idx = args.out_root / "sleeves.json"
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {idx}")


if __name__ == "__main__":
    main()
