"""
=============================================================================
SCRIPT NAME: backtest_generic.py
=============================================================================

INPUT FILES:
- --predictions PATH         predictions.parquet with:
                             ticker, end_date, forward_return, label, p_up_mean
- --universe-xlsx PATH       (optional) xlsx with 'Ticker' column to restrict
                             the backtest universe
- --top-pct / --bot-pct      percentile thresholds (default 20/20)
- --hold-days                forward horizon for non-overlap sampling (20)

OUTPUT FILES:
- {out_dir}/backtest_summary.xlsx     headline stats per portfolio
- {out_dir}/portfolios.parquet        per-date returns for top/bot/LS/EW
- {out_dir}/cumulative.pdf            log-scale cumulative curves
- {out_dir}/yearly_ls.xlsx            year-by-year long-short stats

DESCRIPTION:
Reusable backtest over any predictions.parquet + universe.  Ranks tickers
cross-sectionally by p_up_mean at each end_date, builds equal-weight
top X% and bottom X% portfolios, plus a universe-wide equal-weight
benchmark.  Non-overlap monthly (every hold_days-th date) compounding,
Newey-West t on the daily overlapping series.

USAGE:
  python scripts/backtest_generic.py \
      --predictions runs/etf_expanding/predictions.parquet \
      --universe-xlsx AssetList.xlsx \
      --out-dir     runs/etf_expanding/backtest \
      --top-pct 20 --bot-pct 20
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


def load_universe(xlsx_path: Path | None) -> set[str] | None:
    if xlsx_path is None:
        return None
    df = pd.read_excel(xlsx_path)
    col = "Ticker" if "Ticker" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.strip())


def build_portfolios(
    preds: pd.DataFrame,
    top_pct: float | None,
    bot_pct: float | None,
    top_n: int | None,
    bot_n: int | None,
) -> pd.DataFrame:
    """Return a DataFrame indexed by end_date with columns TOP, BOT, LS, EW.

    Selection:
      - if top_n/bot_n are set → pick that many tickers from each tail (absolute).
      - else use top_pct/bot_pct (percentile).
    """
    rows = []
    for date, g in preds.groupby("end_date", sort=True):
        if len(g) < 3:
            continue
        r = g["forward_return"].to_numpy()
        p = g["p_up_mean"].to_numpy()
        n = len(g)
        if top_n is not None or bot_n is not None:
            k_top = min(top_n or 0, n)
            k_bot = min(bot_n or 0, n)
            order = pd.Series(p).rank(method="first").to_numpy()  # 1..n ascending
            top_mask = order > (n - k_top)
            bot_mask = order <= k_bot
        else:
            rank = pd.Series(p).rank(method="first") / n
            top_mask = (rank >  1 - top_pct / 100).to_numpy()
            bot_mask = (rank <=     bot_pct / 100).to_numpy()
        top_ret = r[top_mask].mean() if top_mask.any() else np.nan
        bot_ret = r[bot_mask].mean() if bot_mask.any() else np.nan
        ew_ret  = r.mean()
        rows.append({
            "end_date":  date,
            "n":         len(g),
            "n_top":     int(top_mask.sum()),
            "n_bot":     int(bot_mask.sum()),
            "TOP":       top_ret,
            "BOT":       bot_ret,
            "LS":        top_ret - bot_ret,
            "EW":        ew_ret,
        })
    out = pd.DataFrame(rows).set_index("end_date").sort_index()
    return out


def ann_stats(r: pd.Series, hold_days: int = 20) -> dict:
    """Stats for a non-overlap sampled LOG-return series.

    forward_return in the parquet is log(p_{t+H}/p_t), so the cumulative
    simple gross return is exp(sum(r)), not prod(1+r).  For the LS column
    (difference of two log returns) the same formula approximates a
    dollar-neutral long/short rebalanced every H days.
    """
    r = r.dropna()
    if len(r) == 0:
        return {k: np.nan for k in ("n", "cum_x", "ann_comp", "ann_arith",
                                    "ann_vol", "sharpe", "t")}
    m = r.mean()               # mean log-return per H-day period
    s = r.std()                # std  log-return per H-day period
    n = len(r)
    years   = n * hold_days / 252
    cum     = float(np.exp(r.sum()))                       # exp(Σ log r)
    ann_comp = float(np.exp(m * 252 / hold_days) - 1)      # CAGR
    ann_ar  = m * (252 / hold_days)                        # annualised mean log r
    ann_vol = s * np.sqrt(252 / hold_days)
    sharpe  = ann_ar / ann_vol if ann_vol > 0 else np.nan
    t       = m / (s / np.sqrt(n)) if s > 0 else np.nan
    return dict(n=n, cum_x=cum, ann_comp=ann_comp * 100,
                ann_arith=ann_ar * 100, ann_vol=ann_vol * 100,
                sharpe=sharpe, t=t)


def newey_west_t(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan
    mu = x.mean()
    xc = x - mu
    s = (xc ** 2).mean()
    for k in range(1, lag + 1):
        if k >= n:
            break
        gk = (xc[k:] * xc[:-k]).mean()
        w = 1 - k / (lag + 1)
        s += 2 * w * gk
    se = np.sqrt(max(s, 0) / n)
    return float(mu / se) if se > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--universe-xlsx", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top-pct", type=float, default=20.0)
    ap.add_argument("--bot-pct", type=float, default=20.0)
    ap.add_argument("--top-n", type=int, default=None,
                    help="absolute # of tickers in top bucket (overrides --top-pct)")
    ap.add_argument("--bot-n", type=int, default=None,
                    help="absolute # of tickers in bottom bucket (overrides --bot-pct)")
    ap.add_argument("--hold-days", type=int, default=20)
    ap.add_argument("--min-date", type=str, default=None,
                    help="e.g. 2000-01-01")
    args = ap.parse_args()

    preds = pd.read_parquet(args.predictions)
    preds["end_date"] = pd.to_datetime(preds["end_date"])

    universe = load_universe(args.universe_xlsx)
    if universe is not None:
        preds = preds[preds["ticker"].isin(universe)].copy()
        missing = universe - set(preds["ticker"].unique())
        print(f"Universe filter: {len(universe)} tickers requested, "
              f"{preds['ticker'].nunique()} present")
        if missing:
            print(f"  missing: {sorted(missing)}")

    if args.min_date is not None:
        preds = preds[preds["end_date"] >= pd.Timestamp(args.min_date)].copy()

    print(f"Predictions: {len(preds):,} rows  "
          f"{preds['end_date'].min().date()} → {preds['end_date'].max().date()}  "
          f"tickers={preds['ticker'].nunique()}")

    port = build_portfolios(preds, args.top_pct, args.bot_pct,
                             args.top_n, args.bot_n)
    if args.top_n is not None or args.bot_n is not None:
        sel_label = f"Top {args.top_n} / Bot {args.bot_n}"
    else:
        sel_label = f"Top {args.top_pct:.0f}% / Bot {args.bot_pct:.0f}%"
    print(f"Portfolio dates: {len(port):,}  "
          f"mean universe={port['n'].mean():.1f}  "
          f"mean n_top={port['n_top'].mean():.1f}  "
          f"mean n_bot={port['n_bot'].mean():.1f}  "
          f"({sel_label})")

    # Non-overlap monthly sample (every hold_days-th date).
    monthly = port.iloc[::args.hold_days].copy()

    rows = {}
    for col in ("TOP", "BOT", "LS", "EW"):
        rows[col] = ann_stats(monthly[col], hold_days=args.hold_days)

    # Newey-West t on the full daily (overlapping) LS series — lag = hold_days - 1.
    nw_t_ls = newey_west_t(port["LS"].to_numpy(), lag=args.hold_days - 1)
    nw_t_top = newey_west_t(port["TOP"].to_numpy(), lag=args.hold_days - 1)
    nw_t_bot = newey_west_t(port["BOT"].to_numpy(), lag=args.hold_days - 1)
    nw_t_ew  = newey_west_t(port["EW"].to_numpy(),  lag=args.hold_days - 1)

    summary = pd.DataFrame(rows).T
    summary["nw_t_daily"] = [nw_t_top, nw_t_bot, nw_t_ls, nw_t_ew]
    summary.index.name = "portfolio"

    print()
    print(f"{sel_label}  —  hold={args.hold_days}d non-overlap")
    print(summary.round(3).to_string())
    print()

    # Year-by-year LS and EW
    port["year"] = port.index.year
    yearly = port.groupby("year").agg(
        n=("LS", "count"),
        ls_mean=("LS", "mean"),
        ls_std=("LS", "std"),
        ew_mean=("EW", "mean"),
        ew_std=("EW", "std"),
    )
    # forward_return is log-return → annualised compound = exp(m * 252/H) - 1
    yearly["ls_ann_pct"] = (np.exp(yearly["ls_mean"] * 252 / args.hold_days) - 1) * 100
    yearly["ew_ann_pct"] = (np.exp(yearly["ew_mean"] * 252 / args.hold_days) - 1) * 100
    yearly["ls_pct_pos"] = (port.groupby("year")["LS"].apply(lambda x: (x > 0).mean()) * 100)
    print("Year-by-year:")
    print(yearly[["n", "ls_ann_pct", "ew_ann_pct", "ls_pct_pos"]].round(2).to_string())
    print()

    # ── write outputs ──────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    port.drop(columns="year").to_parquet(args.out_dir / "portfolios.parquet")
    with pd.ExcelWriter(args.out_dir / "backtest_summary.xlsx",
                         engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="summary")
        yearly.to_excel(xw, sheet_name="yearly")

    # ── plot ───────────────────────────────────────────────────────────────
    # forward_return is a log-return → cumulative simple = exp(cumsum).
    cum = np.exp(monthly[["TOP", "BOT", "LS", "EW"]].cumsum())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    if args.top_n is not None:
        top_label = f"TOP {args.top_n}"
        bot_label = f"BOT {args.bot_n}"
    else:
        top_label = f"TOP {args.top_pct:.0f}%"
        bot_label = f"BOT {args.bot_pct:.0f}%"
    ax1.plot(cum.index, cum["TOP"], color="#2e7d32", lw=1.8, label=top_label)
    ax1.plot(cum.index, cum["BOT"], color="#c62828", lw=1.8, label=bot_label)
    ax1.plot(cum.index, cum["EW"],  color="black",  lw=1.5, ls="--",
             label="EW (universe)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Cumulative value (log, $1 start)")
    ax1.set_title(f"Top/Bot/EW portfolios — {preds['ticker'].nunique()} tickers, "
                  f"{cum.index[0].date()} → {cum.index[-1].date()}")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.plot(cum.index, cum["LS"], color="tab:blue", lw=1.8,
             label=f"LS (Top−Bot)  NW t={nw_t_ls:+.2f}")
    ax2.axhline(1.0, color="grey", lw=0.7, ls="--")
    ax2.set_yscale("log")
    ax2.set_ylabel("LS cumulative (log)")
    ax2.set_xlabel("End date")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper left")

    last = cum.iloc[-1]
    for col, color in [("TOP", "#2e7d32"), ("BOT", "#c62828"), ("EW", "black")]:
        ax1.annotate(f"{last[col]:.2f}×", (cum.index[-1], last[col]),
                     fontsize=9, ha="left", va="center", color=color)
    ax2.annotate(f"{last['LS']:.2f}×", (cum.index[-1], last["LS"]),
                 fontsize=9, ha="left", va="center", color="tab:blue")

    plt.tight_layout()
    out_pdf = args.out_dir / "cumulative.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.with_suffix(".png"), dpi=140, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {args.out_dir / 'backtest_summary.xlsx'}")
    print(f"wrote {args.out_dir / 'portfolios.parquet'}")


if __name__ == "__main__":
    main()
