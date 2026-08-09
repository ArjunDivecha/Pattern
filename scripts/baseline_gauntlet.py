"""
=============================================================================
SCRIPT NAME: baseline_gauntlet.py
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/expanding/20260419_174908_cdef6809/predictions_monthly.parquet
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_ohlcv_spliced.parquet

OUTPUT FILES:
- {out_dir}/baseline_signals.parquet   p_up + every baseline + residual, one row
                                       per (ticker, end_date). Feed any column
                                       to evaluate_signal.py --score-col.
- {out_dir}/gauntlet_summary.xlsx      IC table, head-to-head, double sort
- {out_dir}/gauntlet_summary.json      machine-readable verdict inputs

DESCRIPTION:
STAGE 1 OF THE KILL-OR-CONFIRM PLAN: does the CNN know anything a free signal
doesn't?

The paper itself names weekly short-term reversal (WSTR) and TREND as the
closest competitors to its image forecasts, yet this repo has never compared
p_up against ANY simple signal on R1000 -- every backtest ranks by p_up and
uses momentum/vol only to define buckets. So the headline "CNN alpha" has never
been shown to be distinct from reversal.

Builds, at each month-end formation date:
  rev_1m    1-month reversal (STR)      -- lower past return = buy
  rev_5d    1-week reversal (WSTR)      -- lower past return = buy
  mom_12_1  12-1 momentum               -- higher = buy
  vol_60d   60d realized vol            -- lower = buy (low-vol anomaly)
  combo     z(-rev_1m) + z(-vol_60d)    -- the obvious cheap blend

Then three tests:

  1. IC       cross-sectional Spearman of each signal vs the 20-day forward
              return, and of p_up vs each baseline (how much of p_up simply IS
              a baseline?). Reported with a Newey-West t on the monthly IC
              series, since ICs are serially correlated.

  2. RESIDUAL per month, regress p_up on ALL baselines cross-sectionally and
              keep the residual. Ranking on the residual asks: what survives
              once everything free has been stripped out? This is the sharpest
              form of the question.

  3. DOUBLE   within each STR quintile, sort by p_up into terciles. If the
      SORT    p_up spread survives inside STR buckets, it is not just reversal
              wearing a hat.

The scoring of each signal as a portfolio is deliberately NOT done here -- it
is delegated to scripts/evaluate_signal.py so every number in this stage comes
off the same measuring stick.

USAGE:
  python3 scripts/baseline_gauntlet.py \
      --out-dir runs/experiments/stage1_baselines
  # then, per signal:
  python3 scripts/evaluate_signal.py \
      --signal runs/experiments/stage1_baselines/baseline_signals.parquet \
      --score-col rev_1m --lower-is-better --label STR --out-dir <...>

NOTES:
- All returns here are the repo's 20-day LOG `forward_return`; the harness
  converts to simple for portfolio math. ICs are rank-based so the convention
  does not affect them.
- A $5 price floor is applied before every test, for the reasons in README
  addendum J.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED = (ROOT / "runs" / "expanding" / "20260419_174908_cdef6809"
                / "predictions_monthly.parquet")
DEFAULT_OHLCV = ROOT / "data" / "r1000_ohlcv_spliced.parquet"
MIN_PRICE = 5.0

# signal -> +1 if a HIGH value should be bought, -1 if a LOW value should be
SIGN = {
    "p_up_mean": +1,
    "rev_1m": -1,
    "rev_5d": -1,
    "mom_12_1": +1,
    "vol_60d": -1,
    "combo": +1,
}
BASELINES = ["rev_1m", "rev_5d", "mom_12_1", "vol_60d", "combo"]


def newey_west_t(x: np.ndarray, lag: int = 3) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean(); xc = x - mu
    s = (xc ** 2).mean()
    for k in range(1, lag + 1):
        if k >= n: break
        s += 2 * (1 - k / (lag + 1)) * (xc[k:] * xc[:-k]).mean()
    se = np.sqrt(max(s, 0) / n)
    return float(mu / se) if se > 0 else float("nan")


def build_features(ohlcv_path: Path) -> pd.DataFrame:
    o = pd.read_parquet(ohlcv_path,
                        columns=["Ticker", "Date", "Close", "Volume", "AdjClose"])
    o["Date"] = pd.to_datetime(o["Date"])
    o = o.sort_values(["Ticker", "Date"])
    g = o.groupby("Ticker")
    o["logp"] = np.log(o["AdjClose"].replace(0, np.nan))
    o["rev_1m"] = o["logp"] - g["logp"].shift(21)          # STR
    o["rev_5d"] = o["logp"] - g["logp"].shift(5)           # WSTR
    o["mom_12_1"] = g["logp"].shift(21) - g["logp"].shift(252 + 21)
    o["ret"] = g["AdjClose"].pct_change(fill_method=None)
    o["vol_60d"] = g["ret"].transform(
        lambda s: s.rolling(60, min_periods=20).std() * np.sqrt(252))
    o["dv"] = o["Close"] * o["Volume"]
    o["dv_60d"] = g["dv"].transform(lambda s: s.rolling(60, min_periods=15).mean())
    return (o[["Ticker", "Date", "Close", "rev_1m", "rev_5d", "mom_12_1",
               "vol_60d", "dv_60d"]]
            .rename(columns={"Ticker": "ticker", "Date": "end_date"}))


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-price", type=float, default=MIN_PRICE)
    ap.add_argument("--min-names", type=int, default=50)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading …")
    p = pd.read_parquet(args.predictions)
    p["end_date"] = pd.to_datetime(p["end_date"])
    feats = build_features(args.ohlcv)
    df = p.merge(feats, on=["ticker", "end_date"], how="left")
    df = df[df["Close"] >= args.min_price]
    df = df.dropna(subset=["forward_return", "p_up_mean", "rev_1m", "rev_5d",
                           "mom_12_1", "vol_60d"])

    # combo built per date so the z-scores are cross-sectional
    df["combo"] = df.groupby("end_date", group_keys=False).apply(
        lambda g: zscore(-g["rev_1m"]) + zscore(-g["vol_60d"]))
    print(f"  usable rows {len(df):,}  months {df.end_date.nunique()}  "
          f"{df.end_date.min().date()} -> {df.end_date.max().date()}")

    signals = ["p_up_mean"] + BASELINES

    # ── 1. IC ────────────────────────────────────────────────────────────────
    ic_rows, xcorr_rows, resid_parts, dsort_rows = [], [], [], []
    for dt, g in df.groupby("end_date"):
        if len(g) < args.min_names:
            continue
        r = g["forward_return"]
        rec = {"end_date": dt, "n": len(g)}
        for s in signals:
            rec[s] = SIGN[s] * g[s].corr(r, method="spearman")
        ic_rows.append(rec)

        xc = {"end_date": dt}
        for s in BASELINES:
            # correlate ORIENTED signals: does p_up rank names like this does?
            xc[s] = (SIGN["p_up_mean"] * g["p_up_mean"]).corr(
                SIGN[s] * g[s], method="spearman")
        xcorr_rows.append(xc)

        # ── 2. residual p_up (cross-sectional, this month only) ─────────────
        X = np.column_stack([zscore(SIGN[s] * g[s]).to_numpy() for s in BASELINES])
        X = np.column_stack([np.ones(len(g)), X])
        y = zscore(g["p_up_mean"]).to_numpy()
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        resid = np.full(len(g), np.nan)
        if ok.sum() > len(BASELINES) + 5:
            beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            resid[ok] = y[ok] - X[ok] @ beta
        resid_parts.append(pd.DataFrame({
            "ticker": g["ticker"].to_numpy(), "end_date": dt,
            "p_up_resid": resid}))

        # ── 3. double sort: p_up tercile spread inside STR quintiles ────────
        if len(g) >= 100:
            gq = g.assign(strq=pd.qcut((-g["rev_1m"]).rank(method="first"), 5,
                                       labels=range(1, 6)))
            for q, sub in gq.groupby("strq", observed=True):
                if len(sub) < 15:
                    continue
                t = sub["p_up_mean"].rank(method="first", pct=True)
                hi = sub.loc[t > 2 / 3, "forward_return"]
                lo = sub.loc[t <= 1 / 3, "forward_return"]
                if len(hi) >= 3 and len(lo) >= 3:
                    dsort_rows.append({
                        "end_date": dt, "str_quintile": int(q),
                        "spread_simple": np.expm1(hi).mean() - np.expm1(lo).mean()})

    ic = pd.DataFrame(ic_rows).sort_values("end_date")
    xcorr = pd.DataFrame(xcorr_rows).sort_values("end_date")
    resid_df = pd.concat(resid_parts, ignore_index=True)

    ic_tab = pd.DataFrame([{
        "signal": s,
        "mean_IC": round(float(ic[s].mean()), 5),
        "IC_t_NW": round(newey_west_t(ic[s].to_numpy()), 2),
        "pct_months_positive": round(float((ic[s] > 0).mean() * 100), 1),
    } for s in signals]).sort_values("mean_IC", ascending=False)

    xc_tab = pd.DataFrame([{
        "vs_p_up": s,
        "mean_rank_corr": round(float(xcorr[s].mean()), 4),
        "abs_corr": round(float(xcorr[s].abs().mean()), 4),
    } for s in BASELINES]).sort_values("abs_corr", ascending=False)

    print("\n=== 1. Predictive IC (Spearman vs 20d forward return, oriented) ===")
    print(ic_tab.to_string(index=False))
    print("\n=== 2. How much of p_up IS a baseline? (cross-sectional rank corr) ===")
    print(xc_tab.to_string(index=False))

    ds_tab = pd.DataFrame()
    if dsort_rows:
        ds = pd.DataFrame(dsort_rows)
        ds_tab = ds.groupby("str_quintile").agg(
            months=("spread_simple", "size"),
            mean_spread_pct=("spread_simple", lambda x: x.mean() * 100),
        ).reset_index()
        ds_tab["t_NW"] = [
            round(newey_west_t(ds.loc[ds.str_quintile == q, "spread_simple"].to_numpy()), 2)
            for q in ds_tab.str_quintile]
        ds_tab["mean_spread_pct"] = ds_tab["mean_spread_pct"].round(4)
        print("\n=== 3. p_up tercile spread WITHIN STR quintiles (per 20d, simple) ===")
        print("    (q1 = biggest recent losers … q5 = biggest recent winners)")
        print(ds_tab.to_string(index=False))

    # ── emit a signal file the harness can score directly ───────────────────
    out = df[["ticker", "end_date", "forward_return", "p_up_mean"] + BASELINES].copy()
    out = out.merge(resid_df, on=["ticker", "end_date"], how="left")
    # pre-orient so every column is "higher = buy" for the harness
    for s in BASELINES:
        out[f"{s}_oriented"] = SIGN[s] * out[s]
    out.to_parquet(args.out_dir / "baseline_signals.parquet", index=False)

    with pd.ExcelWriter(args.out_dir / "gauntlet_summary.xlsx", engine="openpyxl") as xw:
        ic_tab.to_excel(xw, sheet_name="predictive_IC", index=False)
        xc_tab.to_excel(xw, sheet_name="p_up_vs_baselines", index=False)
        if len(ds_tab):
            ds_tab.to_excel(xw, sheet_name="double_sort", index=False)
        ic.to_excel(xw, sheet_name="IC_monthly", index=False)

    (args.out_dir / "gauntlet_summary.json").write_text(json.dumps({
        "months": int(ic.end_date.nunique()),
        "predictive_IC": ic_tab.to_dict("records"),
        "p_up_vs_baselines": xc_tab.to_dict("records"),
        "double_sort": ds_tab.to_dict("records") if len(ds_tab) else [],
    }, indent=2, default=str))

    print(f"\nwrote {args.out_dir / 'baseline_signals.parquet'}")
    print(f"wrote {args.out_dir / 'gauntlet_summary.xlsx'}")


if __name__ == "__main__":
    main()
