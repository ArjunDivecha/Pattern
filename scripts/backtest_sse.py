"""
=============================================================================
SCRIPT NAME: backtest_sse.py
=============================================================================

INPUT FILES:
- runs/sse_underlying_expanding/predictions.parquet
                                  ticker, end_date, p_up_mean, ...
- data/sse_wrapper_ohlcv.csv      OHLCV for each ETF wrapper
- data/sse_pairs.csv              underlying -> long_etf / short_etf

OUTPUT FILES:
- {out_dir}/sse_monthly.parquet      per-(end_date, underlying) wrapper returns
- {out_dir}/sse_portfolio.parquet    monthly TOP / BOT / LS returns
- {out_dir}/sse_summary.xlsx         per-variant CAGR / Sharpe / NW t / costs
- {out_dir}/sse_cum.pdf              cumulative LS net of costs

DESCRIPTION:
Backtests the CNN signal using single-stock leveraged ETFs as the actual
trading instruments — no shorting needed.  At each month-end we:

  1. Rank underlyings by p_up_mean (CNN ensemble mean).
  2. Bottom-half (predicted DOWN) -> buy the inverse-leveraged ETF.
  3. Top-half (predicted UP)      -> buy the long-leveraged ETF.
  4. Hold ~21 trading days, then rebalance.

Portfolio return per dollar of capital:
  R_t = mean(R_long_wrapper for top names) * w_long
      + mean(R_inverse_wrapper for bot names) * w_inv

For an equal-capital split (w_long = w_inv = 0.5) the portfolio gross
return = 0.5 * (R_long_mean + R_inv_mean).  This is reported under the
'$-equal' variant.  We also report 'lev-matched' which sizes positions
inversely to wrapper leverage so the underlying notional exposure is
matched on each side.

Costs modelled:
  - Per-leg half-spread (default 5 bps), round-trip on full rebalance
  - Wrapper expense ratio (default 1.0 % p.a.) prorated over hold period
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


# ── helpers ───────────────────────────────────────────────────────────────────

def newey_west_t(x: np.ndarray, lag: int = 0) -> float:
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan
    mu = x.mean(); xc = x - mu
    s = (xc ** 2).mean()
    for k in range(1, lag + 1):
        if k >= n: break
        gk = (xc[k:] * xc[:-k]).mean()
        w = 1 - k / (lag + 1)
        s += 2 * w * gk
    se = np.sqrt(max(s, 0) / n)
    return float(mu / se) if se > 0 else np.nan


def month_end_to_fwd_return(ohlcv: pd.DataFrame, ticker: str,
                            end_date: pd.Timestamp,
                            horizon: int = 21) -> float:
    """Close-to-close return for `ticker` over the next `horizon` trading days
    starting AT or AFTER end_date (so the position is set on end_date close)."""
    px = ohlcv.loc[ohlcv["Ticker"] == ticker].sort_values("Date")
    if len(px) == 0:
        return np.nan
    # Use AdjClose if present (handles splits in wrappers — Direxion does ROC
    # but mostly clean); fall back to Close.
    pcol = "AdjClose" if "AdjClose" in px.columns and px["AdjClose"].notna().any() else "Close"
    # Find the close on end_date or the first trading day after
    sub = px[px["Date"] >= end_date]
    if len(sub) < horizon + 1:
        return np.nan
    p0 = sub.iloc[0][pcol]
    p1 = sub.iloc[horizon][pcol]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return np.nan
    return float(p1 / p0 - 1.0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path,
                    default=Path("runs/sse_underlying_expanding/predictions.parquet"))
    ap.add_argument("--wrappers", type=Path,
                    default=Path("data/sse_wrapper_ohlcv.csv"))
    ap.add_argument("--pairs", type=Path,
                    default=Path("data/sse_pairs.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("runs/sse_underlying_expanding/backtest_sse"))
    ap.add_argument("--horizon-days", type=int, default=21)
    ap.add_argument("--half-spread-bps", type=float, default=5.0,
                    help="per-leg half-spread in basis points (default 5)")
    ap.add_argument("--expense-ratio-pa", type=float, default=0.01,
                    help="annual expense ratio (default 1.0pct)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inputs …")
    preds = pd.read_parquet(args.predictions)
    preds["end_date"] = pd.to_datetime(preds["end_date"])
    wr = pd.read_csv(args.wrappers, parse_dates=["Date"])
    pairs = pd.read_csv(args.pairs)
    # Take first long_etf per underlying (prefer Direxion / GraniteShares — the
    # earliest, most liquid wrapper).  Build best-pair table by underlying.
    pairs_sorted = pairs.sort_values(
        ["underlying", "note", "sponsor_long"],
        key=lambda s: s if s.name != "note" else
            s.map({"asymmetric": 0, "symmetric": 1, "symmetric-mixed": 2,
                   "symmetric-alt": 3, "alt": 4, "long-only": 9}).fillna(5))
    best = pairs_sorted.drop_duplicates("underlying", keep="first").reset_index(drop=True)
    print(f"  predictions: {len(preds):,} rows, "
          f"{preds['ticker'].nunique()} underlyings, "
          f"{preds['end_date'].min().date()} → {preds['end_date'].max().date()}")
    print(f"  wrappers:    {wr['Ticker'].nunique()} tickers, "
          f"{wr['Date'].min().date()} → {wr['Date'].max().date()}")
    print(f"  pair-table:  {len(best)} underlyings  "
          f"({best['short_etf_valid'].fillna('').str.len().gt(0).sum()} with short ETF)")

    # ── Build per-(end_date, underlying) wrapper returns ─────────────────────
    print("\nComputing wrapper forward returns at every month-end …")
    rows = []
    preds_w = preds.merge(best, left_on="ticker", right_on="underlying",
                           how="inner")
    for (date, ticker), g in preds_w.groupby(["end_date", "ticker"]):
        row = g.iloc[0]
        long_w  = row.get("long_etf", "")
        short_w = row.get("short_etf_valid", "")
        if not isinstance(short_w, str):
            short_w = ""
        r_long = month_end_to_fwd_return(wr, long_w, date, args.horizon_days) \
                    if isinstance(long_w, str) and long_w else np.nan
        r_inv  = month_end_to_fwd_return(wr, short_w, date, args.horizon_days) \
                    if short_w else np.nan
        rows.append({
            "end_date": date,
            "underlying": ticker,
            "p_up_mean": row["p_up_mean"],
            "long_etf": long_w,
            "short_etf": short_w,
            "long_lev": row.get("long_lev", 2.0),
            "short_lev": row.get("short_lev", np.nan),
            "r_long_wrapper": r_long,
            "r_inv_wrapper":  r_inv,
            "r_underlying_fwd": row.get("forward_return", np.nan),
        })
    mon = pd.DataFrame(rows).sort_values(["end_date", "underlying"])
    mon = mon[mon["end_date"] >= "2022-08-01"]   # before first wrapper inception
    mon.to_parquet(args.out_dir / "sse_monthly.parquet", index=False)
    print(f"  wrote {args.out_dir / 'sse_monthly.parquet'}  rows={len(mon):,}")

    # ── Build monthly LS portfolios under several rules ──────────────────────
    print("\nBuilding monthly portfolios …")
    holding_yr_frac = args.horizon_days / 252.0
    exp_drag_per_month = args.expense_ratio_pa * holding_yr_frac  # ~0.083%
    spread_per_leg = args.half_spread_bps / 1e4                    # half-spread

    summaries = []
    cum_data = {}

    for variant in ["complete-pairs", "long-only", "full"]:
        # complete-pairs: only underlyings with valid short_etf → true L/S
        # long-only:      only long leg (top decile half), no short hedge
        # full:           include long-only names on long side, skip short
        port_rows = []
        for date, g in mon.groupby("end_date"):
            if variant == "complete-pairs":
                cell = g.dropna(subset=["r_long_wrapper", "r_inv_wrapper"])
            elif variant == "long-only":
                cell = g.dropna(subset=["r_long_wrapper"])
            else:  # full
                cell = g.dropna(subset=["r_long_wrapper"]).copy()
            if len(cell) < 4:
                continue
            n = len(cell)
            rank = cell["p_up_mean"].rank(method="first") / n
            top = cell[rank > 0.5]
            bot = cell[rank <= 0.5]
            if variant == "long-only":
                # Just go long top-half via long-ETF, no short
                r_long_mean = top["r_long_wrapper"].mean()
                r_port = r_long_mean
                r_long_only_top = r_long_mean
                r_inv_only_bot = np.nan
                r_top_under, r_bot_under = top["r_underlying_fwd"].mean(), bot["r_underlying_fwd"].mean()
            else:
                # Equal capital: 50% long-ETF on top half, 50% inverse-ETF on bot half
                # For 'full' variant, drop bot names without short ETF
                bot_paired = bot.dropna(subset=["r_inv_wrapper"])
                if len(bot_paired) < 2:
                    continue
                r_long_only_top = top["r_long_wrapper"].mean()
                r_inv_only_bot  = bot_paired["r_inv_wrapper"].mean()
                r_port = 0.5 * (r_long_only_top + r_inv_only_bot)
                r_top_under, r_bot_under = top["r_underlying_fwd"].mean(), bot["r_underlying_fwd"].mean()
            port_rows.append({
                "end_date": date,
                "n": n, "n_top": len(top), "n_bot": len(bot),
                "r_long_etf_top": r_long_only_top,
                "r_inv_etf_bot":  r_inv_only_bot,
                "r_portfolio_gross": r_port,
                "r_underlying_top": r_top_under,
                "r_underlying_bot": r_bot_under,
                "r_underlying_LS":  r_top_under - r_bot_under,
            })
        pf = pd.DataFrame(port_rows).sort_values("end_date").reset_index(drop=True)
        if not len(pf):
            print(f"  [skip] variant={variant}: no portfolios")
            continue

        # Approximate full-rebalance round-trip cost per month:
        #   2 sides × full turnover × half-spread  (round-trip)
        # Direxion's published OER ~0.95-1.15 %; we drag both legs.
        cost_per_month = exp_drag_per_month + 2 * 2 * spread_per_leg  # rough
        pf["r_portfolio_net"] = pf["r_portfolio_gross"] - cost_per_month

        # Annualise from monthly
        rg = pf["r_portfolio_gross"].dropna(); rn = pf["r_portfolio_net"].dropna()
        gross_cagr = (1 + rg).prod() ** (12 / max(len(rg), 1)) - 1
        net_cagr   = (1 + rn).prod() ** (12 / max(len(rn), 1)) - 1
        sharpe_g   = (rg.mean() * 12) / (rg.std() * np.sqrt(12)) if rg.std() > 0 else np.nan
        sharpe_n   = (rn.mean() * 12) / (rn.std() * np.sqrt(12)) if rn.std() > 0 else np.nan
        nw_t_g     = newey_west_t(rg.to_numpy())
        nw_t_n     = newey_west_t(rn.to_numpy())

        # Save
        pf.to_parquet(args.out_dir / f"sse_portfolio_{variant}.parquet", index=False)
        cum_data[variant] = pf
        summaries.append({
            "variant": variant,
            "months":     len(pf),
            "mean_univ":  pf["n"].mean(),
            "first_date": pf["end_date"].min().date(),
            "gross_cagr_pct":  gross_cagr * 100,
            "net_cagr_pct":    net_cagr * 100,
            "monthly_mean_pct": rg.mean() * 100,
            "monthly_std_pct":  rg.std() * 100,
            "sharpe_gross":    sharpe_g,
            "sharpe_net":      sharpe_n,
            "nw_t_gross":      nw_t_g,
            "nw_t_net":        nw_t_n,
            "cost_per_month_bps": cost_per_month * 1e4,
            "expense_drag_pct_pa": args.expense_ratio_pa * 100,
            "spread_bps_per_leg":  args.half_spread_bps,
        })

    summary = pd.DataFrame(summaries)
    print()
    print(summary.round(2).to_string(index=False))

    with pd.ExcelWriter(args.out_dir / "sse_summary.xlsx", engine="openpyxl") as xw:
        summary.round(3).to_excel(xw, sheet_name="summary", index=False)
        for v, df in cum_data.items():
            df.round(4).to_excel(xw, sheet_name=v[:31], index=False)
    print(f"\nwrote {args.out_dir / 'sse_summary.xlsx'}")

    # ── Cumulative plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    for v, df in cum_data.items():
        cum_g = (1 + df["r_portfolio_gross"]).cumprod()
        cum_n = (1 + df["r_portfolio_net"]).cumprod()
        ax.plot(df["end_date"], cum_g, label=f"{v} (gross)", lw=1.3)
        ax.plot(df["end_date"], cum_n, label=f"{v} (net)", lw=1.0, ls="--")
    ax.set_title("Single-Stock ETF strategy — cumulative growth of $1")
    ax.set_ylabel("growth of $1"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    out_pdf = args.out_dir / "sse_cum.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
