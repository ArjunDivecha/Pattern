"""
=============================================================================
SCRIPT NAME: backtest_sse_momentum.py
=============================================================================

INPUT FILES:
- data/sse_underlying_ohlcv.csv     daily OHLCV for the underlying stocks
- data/sse_wrapper_ohlcv.csv        daily OHLCV for the leveraged ETFs
- data/sse_pairs.csv                pair table (underlying / long_etf / short_etf)
- runs/sse_underlying_expanding/predictions.parquet
                                    (used only to align month-ends with CNN)

OUTPUT FILES:
- {out_dir}/sse_momentum_summary.xlsx     per-signal × per-variant summary
- {out_dir}/sse_momentum_portfolios.parquet  monthly portfolio returns
- {out_dir}/sse_momentum_cum.pdf          cumulative growth of $1

DESCRIPTION:
Tests classical cross-sectional signals on the SSE universe using the same
long-leveraged + inverse-leveraged ETF mechanics as backtest_sse.py.

Signals tested at each month-end T:
  mom_12_1 : log return from T-252 to T-21         (classic momentum)
  mom_6_1  : log return from T-126 to T-21         (short-horizon momentum)
  mom_3_1  : log return from T-63  to T-21
  rev_1m   : log return from T-21 to T             (one-month reversal)
  vol_60d  : 60-day realized vol                   (low-vol bet, inverted rank)
  trend    : (close − 200d MA) / 200d MA           (trend-following)

For each signal we form 50/50 cross-sectional LS books:
  - top-half signal → long via 2x long-ETF
  - bot-half signal → 'short' via inverse-leveraged ETF (long inverse position)

Reports gross + net of fixed costs (1% p.a. expense + 5 bps half-spread per
leg).  Includes an equal-weight long-ETF basket benchmark and the prior CNN
result for comparison.
=============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def newey_west_t(x, lag=0):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    n = len(x)
    if n == 0: return np.nan
    mu = x.mean(); xc = x - mu
    s = (xc**2).mean()
    for k in range(1, lag+1):
        if k >= n: break
        gk = (xc[k:] * xc[:-k]).mean()
        w = 1 - k/(lag+1)
        s += 2*w*gk
    se = np.sqrt(max(s,0)/n)
    return float(mu/se) if se > 0 else np.nan


def compute_signals(ohlcv):
    df = ohlcv[["Ticker", "Date", "AdjClose", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["px"] = df["AdjClose"].fillna(df["Close"])
    df = df.sort_values(["Ticker", "Date"])
    g = df.groupby("Ticker")
    df["logp"] = np.log(df["px"].replace(0, np.nan))
    df["ret1d"] = g["px"].pct_change()
    df["mom_12_1"] = g["logp"].shift(21) - g["logp"].shift(252+21)
    df["mom_6_1"]  = g["logp"].shift(21) - g["logp"].shift(126+21)
    df["mom_3_1"]  = g["logp"].shift(21) - g["logp"].shift(63+21)
    df["rev_1m"]   = g["logp"].diff(21)
    df["vol_60d"]  = g["ret1d"].transform(lambda s: s.rolling(60, min_periods=20).std() * np.sqrt(252))
    df["ma200"]    = g["px"].transform(lambda s: s.rolling(200, min_periods=100).mean())
    df["trend"]    = (df["px"] - df["ma200"]) / df["ma200"]
    return df


def month_end_dates(df, cal):
    """Return the calendar of (ticker, month-end date, signals, fwd return) per ticker."""
    df = df.copy()
    df["ym"] = df["Date"].dt.to_period("M")
    me = (df.sort_values(["Ticker", "Date"])
            .groupby(["Ticker", "ym"], as_index=False, sort=False)
            .tail(1)
            .reset_index(drop=True))
    return me


def fwd_wrapper_return(wrap, ticker, end_date, horizon=21):
    px = wrap.loc[wrap["Ticker"] == ticker].sort_values("Date")
    if len(px) == 0: return np.nan
    pcol = "AdjClose" if "AdjClose" in px.columns and px["AdjClose"].notna().any() else "Close"
    sub = px[px["Date"] >= end_date]
    if len(sub) < horizon + 1: return np.nan
    p0 = sub.iloc[0][pcol]
    p1 = sub.iloc[horizon][pcol]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0: return np.nan
    return float(p1 / p0 - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", type=Path, default=Path("data/sse_underlying_ohlcv.csv"))
    ap.add_argument("--wrappers",   type=Path, default=Path("data/sse_wrapper_ohlcv.csv"))
    ap.add_argument("--pairs",      type=Path, default=Path("data/sse_pairs.csv"))
    ap.add_argument("--out-dir",    type=Path, default=Path("runs/sse_underlying_expanding/backtest_sse_momentum"))
    ap.add_argument("--horizon-days", type=int, default=21)
    ap.add_argument("--half-spread-bps", type=float, default=5.0)
    ap.add_argument("--expense-pa",   type=float, default=0.01)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading …")
    und = pd.read_csv(args.underlying, parse_dates=["Date"])
    wr  = pd.read_csv(args.wrappers,   parse_dates=["Date"])
    pairs = pd.read_csv(args.pairs)
    best = (pairs.sort_values(["underlying", "note"],
              key=lambda s: s if s.name != "note" else
                  s.map({"asymmetric":0,"symmetric":1,"symmetric-mixed":2,
                         "symmetric-alt":3,"alt":4,"long-only":9}).fillna(5))
            .drop_duplicates("underlying", keep="first")
            .reset_index(drop=True))

    print("Computing signals on underlyings …")
    feats = compute_signals(und)
    me = month_end_dates(feats, None)
    me = me.rename(columns={"Ticker": "underlying", "Date": "end_date"})

    # Pull forward wrapper return at each (underlying, end_date)
    print("Computing wrapper forward returns at month-ends …")
    me = me.merge(best, on="underlying", how="inner")
    rows = []
    for _, r in me.iterrows():
        if not isinstance(r["long_etf"], str) or not r["long_etf"]: continue
        rL = fwd_wrapper_return(wr, r["long_etf"], r["end_date"], args.horizon_days)
        rI = (fwd_wrapper_return(wr, r["short_etf_valid"], r["end_date"], args.horizon_days)
              if isinstance(r["short_etf_valid"], str) and r["short_etf_valid"] else np.nan)
        # 21-day forward underlying return
        sub = und.loc[und["Ticker"] == r["underlying"]].sort_values("Date")
        sub = sub[sub["Date"] >= r["end_date"]]
        rU = np.nan
        if len(sub) >= args.horizon_days + 1:
            p0 = sub.iloc[0]["AdjClose"] if pd.notna(sub.iloc[0]["AdjClose"]) else sub.iloc[0]["Close"]
            p1 = sub.iloc[args.horizon_days]["AdjClose"] if pd.notna(sub.iloc[args.horizon_days]["AdjClose"]) else sub.iloc[args.horizon_days]["Close"]
            if p0 and p1 and p0 > 0:
                rU = float(p1/p0 - 1.0)
        rows.append({
            "end_date": r["end_date"], "underlying": r["underlying"],
            "mom_12_1": r["mom_12_1"], "mom_6_1": r["mom_6_1"], "mom_3_1": r["mom_3_1"],
            "rev_1m": r["rev_1m"], "vol_60d": r["vol_60d"], "trend": r["trend"],
            "long_etf": r["long_etf"], "short_etf": r["short_etf_valid"],
            "r_long_w": rL, "r_inv_w": rI, "r_u": rU,
        })
    mon = pd.DataFrame(rows).sort_values(["end_date","underlying"])
    mon = mon[mon["end_date"] >= "2022-08-01"]
    mon.to_parquet(args.out_dir / "sse_monthly_momentum.parquet", index=False)
    print(f"  rows={len(mon):,}")

    # ── Run each signal × {complete-pairs, long-only} ────────────────────────
    SIGNALS = {
        "mom_12_1": ("higher → long", "mom_12_1", +1),
        "mom_6_1":  ("higher → long", "mom_6_1",  +1),
        "mom_3_1":  ("higher → long", "mom_3_1",  +1),
        "rev_1m":   ("LOWER → long  (mean revert)", "rev_1m", -1),
        "low_vol":  ("LOWER vol → long",            "vol_60d", -1),
        "trend":    ("higher → long",               "trend",   +1),
    }
    cost_per_mo = args.expense_pa * (args.horizon_days/252.0) + 2*2*(args.half_spread_bps/1e4)

    all_pf = []
    summaries = []
    for sig_name, (desc, col, sign) in SIGNALS.items():
        for variant in ["complete-pairs", "long-only"]:
            pf_rows = []
            for date, g in mon.groupby("end_date"):
                if variant == "complete-pairs":
                    cell = g.dropna(subset=["r_long_w", "r_inv_w", col])
                else:
                    cell = g.dropna(subset=["r_long_w", col])
                if len(cell) < 4: continue
                # Rank by signal (sign +1 means high = long; sign -1 inverts)
                score = sign * cell[col]
                rank = score.rank(method="first") / len(cell)
                top = cell[rank > 0.5]
                bot = cell[rank <= 0.5]
                if variant == "long-only":
                    r_port = top["r_long_w"].mean()
                    pf_rows.append({"end_date": date, "n": len(cell),
                                    "n_top": len(top),
                                    "r_long_top": top["r_long_w"].mean(),
                                    "r_port_gross": r_port,
                                    "r_und_top": top["r_u"].mean(),
                                    "r_und_bot": bot["r_u"].mean()})
                else:
                    bot_p = bot.dropna(subset=["r_inv_w"])
                    if len(bot_p) < 2: continue
                    rL = top["r_long_w"].mean()
                    rI = bot_p["r_inv_w"].mean()
                    pf_rows.append({"end_date": date, "n": len(cell),
                                    "n_top": len(top), "n_bot": len(bot_p),
                                    "r_long_top": rL, "r_inv_bot": rI,
                                    "r_port_gross": 0.5*(rL + rI),
                                    "r_und_top": top["r_u"].mean(),
                                    "r_und_bot": bot["r_u"].mean()})
            pf = pd.DataFrame(pf_rows)
            if not len(pf): continue
            pf["r_port_net"] = pf["r_port_gross"] - cost_per_mo
            pf["signal"] = sig_name; pf["variant"] = variant
            all_pf.append(pf)

            rg = pf["r_port_gross"].dropna()
            rn = pf["r_port_net"].dropna()
            cagr_g = (1+rg).prod()**(12/max(len(rg),1)) - 1
            cagr_n = (1+rn).prod()**(12/max(len(rn),1)) - 1
            shg = (rg.mean()*12)/(rg.std()*np.sqrt(12)) if rg.std() > 0 else np.nan
            shn = (rn.mean()*12)/(rn.std()*np.sqrt(12)) if rn.std() > 0 else np.nan
            t_g = newey_west_t(rg.to_numpy())
            t_n = newey_west_t(rn.to_numpy())
            # Underlying L/S sanity (no wrapper)
            uls = pf["r_und_top"] - pf["r_und_bot"]
            ucagr = (1+uls.dropna()).prod()**(12/max(len(uls.dropna()),1)) - 1
            ut = newey_west_t(uls.dropna().to_numpy())
            summaries.append({
                "signal": sig_name, "variant": variant, "rule": desc,
                "months": len(pf), "mean_univ": pf["n"].mean(),
                "gross_cagr_pct": cagr_g*100, "net_cagr_pct": cagr_n*100,
                "monthly_mean_pct": rg.mean()*100, "monthly_std_pct": rg.std()*100,
                "sharpe_gross": shg, "sharpe_net": shn,
                "nw_t_gross": t_g, "nw_t_net": t_n,
                "underlying_LS_cagr_pct": ucagr*100,
                "underlying_LS_t": ut,
            })

    # Equal-weight basket benchmark (just buy every long-ETF, no signal)
    pf_bench_rows = []
    for date, g in mon.groupby("end_date"):
        cell = g.dropna(subset=["r_long_w"])
        if len(cell) < 4: continue
        pf_bench_rows.append({"end_date": date, "n": len(cell),
                              "r_port_gross": cell["r_long_w"].mean()})
    pf_b = pd.DataFrame(pf_bench_rows)
    pf_b["r_port_net"] = pf_b["r_port_gross"] - args.expense_pa*(args.horizon_days/252.0) - 2*(args.half_spread_bps/1e4)  # one-leg only
    pf_b["signal"] = "EW_basket"; pf_b["variant"] = "long-only"
    all_pf.append(pf_b)
    rg = pf_b["r_port_gross"].dropna()
    cagr_g = (1+rg).prod()**(12/max(len(rg),1)) - 1
    cagr_n = (1+pf_b["r_port_net"].dropna()).prod()**(12/max(len(pf_b["r_port_net"].dropna()),1)) - 1
    summaries.append({
        "signal": "EW_basket", "variant": "long-only", "rule": "buy every long-ETF",
        "months": len(pf_b), "mean_univ": pf_b["n"].mean(),
        "gross_cagr_pct": cagr_g*100, "net_cagr_pct": cagr_n*100,
        "monthly_mean_pct": rg.mean()*100, "monthly_std_pct": rg.std()*100,
        "sharpe_gross": (rg.mean()*12)/(rg.std()*np.sqrt(12)) if rg.std() > 0 else np.nan,
        "sharpe_net":   np.nan,
        "nw_t_gross":   newey_west_t(rg.to_numpy()),
        "nw_t_net":     np.nan,
        "underlying_LS_cagr_pct": np.nan, "underlying_LS_t": np.nan,
    })

    summary = pd.DataFrame(summaries).sort_values(
        ["variant", "gross_cagr_pct"], ascending=[True, False]).reset_index(drop=True)

    print()
    cols = ["signal","variant","rule","months","mean_univ",
            "gross_cagr_pct","net_cagr_pct","monthly_mean_pct","monthly_std_pct",
            "sharpe_gross","nw_t_gross","underlying_LS_cagr_pct","underlying_LS_t"]
    print(summary[cols].round(2).to_string(index=False))

    all_pf_df = pd.concat(all_pf, ignore_index=True)
    all_pf_df.to_parquet(args.out_dir / "sse_momentum_portfolios.parquet", index=False)
    with pd.ExcelWriter(args.out_dir / "sse_momentum_summary.xlsx",
                          engine="openpyxl") as xw:
        summary.round(3).to_excel(xw, sheet_name="summary", index=False)
        for (sig, var), g in all_pf_df.groupby(["signal", "variant"]):
            g.round(4).to_excel(xw, sheet_name=f"{sig[:18]}_{var[:8]}", index=False)
    print(f"\nwrote {args.out_dir / 'sse_momentum_summary.xlsx'}")

    # ── Cumulative plot — long-only variant only for clarity ────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    long_only = all_pf_df[all_pf_df["variant"] == "long-only"]
    for sig, g in long_only.groupby("signal"):
        g = g.sort_values("end_date")
        cum = (1 + g["r_port_gross"]).cumprod()
        ax.plot(g["end_date"], cum, label=sig, lw=1.3)
    ax.set_title("SSE momentum-style signals — long-only gross growth of $1")
    ax.set_ylabel("growth of $1"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    out_pdf = args.out_dir / "sse_momentum_cum.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
