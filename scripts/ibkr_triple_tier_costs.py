"""
=============================================================================
SCRIPT NAME: ibkr_triple_tier_costs.py
=============================================================================

INPUT FILES:
- --predictions PATH    predictions_monthly.parquet (ticker, end_date, p_up_mean, ...)
- --ohlcv PATH          data/r1000_ohlcv_database.parquet

OUTPUT FILES:
- {out-dir}/ibkr_spreads_YYYYMMDD.xlsx     per-ticker bid/ask, quoted spread,
                                            shortability, 30d historical spread
- {out-dir}/ibkr_repriced_summary.txt       net-return estimate using real spreads

DESCRIPTION:
Pulls live bid/ask and short-availability data from Interactive Brokers for the
current-month triple-tier-filter universe (small-cap ∩ high-vol ∩ recent-loser),
then re-runs the per-month cost model with IBKR-observed spreads instead of the
Corwin-Schultz proxy.  Two data sources per ticker:

    (a) Real-time snapshot (reqMktData snapshot=True) → current bid/ask + shortable
    (b) 30 trading days of daily BID_ASK bars (reqHistoricalData whatToShow='BID_ASK')
        → robust per-name average spread

The script combines these with Almgren-style impact (σ_daily × √(Q/ADV))
and the same turnover series from the backtest to estimate net CAGR.

PREREQUISITES:
- TWS or IB Gateway running and logged in
- API enabled in TWS/Gateway Configure → API → Settings
- Port 7496 (TWS live) / 7497 (TWS paper) / 4001 (GW live) / 4002 (GW paper)
- ib_insync installed:  pip install ib_insync

USAGE:
  python scripts/ibkr_triple_tier_costs.py \
      --predictions runs/expanding/20260419_174908_cdef6809/predictions_monthly.parquet \
      --out-dir     runs/expanding/20260419_174908_cdef6809/ibkr_costs \
      --ib-port 7497

  # For a smaller sample while testing:
  python scripts/ibkr_triple_tier_costs.py ... --max-names 20

NOTES ON SHORT BORROW:
IBKR does not expose the exact short-rebate fee via the standard TWS API.
Closest proxies: genericTick 236 ("Shortable"; ≥3 = easy, 2-3 = available,
<2 = hard-to-borrow) and 258 ("Shortable Shares"; raw count).  The per-ticker
fee rate itself is only available in TWS UI or via the IBKR Stock Loan
availability file (ftp2.interactivebrokers.com or the Client Portal download).
This script flags hard-to-borrow names so they can be priced manually.
=============================================================================
"""
from __future__ import annotations

# ── Python-3.14 compatibility shim for ib_insync (event-loop autocreate) ──
import asyncio as _asyncio
try:
    _asyncio.get_event_loop()
except RuntimeError:
    _asyncio.set_event_loop(_asyncio.new_event_loop())

import argparse
import math
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ib_insync import IB, Stock, util

warnings.filterwarnings("ignore")


# ───────────────────── feature construction (matches backtest) ─────────────────────
def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv[["Ticker", "Date", "High", "Low", "Close", "Volume", "AdjClose"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])
    g = df.groupby("Ticker", group_keys=False)
    # dollar-volume proxy
    df["dv_60d"] = g.apply(
        lambda x: (x["Close"] * x["Volume"]).rolling(60, min_periods=15).mean()
    ).reset_index(level=0, drop=True)
    # 12-1 momentum
    df["logp"] = np.log(df["AdjClose"].replace(0, np.nan))
    df["mom_12_1"] = g["logp"].shift(21) - g["logp"].shift(252 + 21)
    # 60d realized vol (annualised)
    df["ret"] = g["AdjClose"].pct_change()
    df["vol_60d"] = g["ret"].transform(
        lambda s: s.rolling(60, min_periods=20).std() * np.sqrt(252)
    )
    return df[["Ticker", "Date", "dv_60d", "mom_12_1", "vol_60d"]].rename(
        columns={"Ticker": "ticker", "Date": "end_date"}
    )


def qtile(g: pd.DataFrame, col: str) -> pd.Series:
    if g[col].notna().sum() < 3:
        return pd.Series([pd.NA] * len(g), index=g.index)
    return pd.qcut(
        g[col].rank(method="first"), 3,
        labels=["Low", "Mid", "High"], duplicates="drop",
    )


# ───────────────────── ticker-format conversion ─────────────────────
def bbg_to_ibkr_symbol(bbg_ticker: str) -> str | None:
    """'AAPL US Equity' → 'AAPL';  'BRK/B US Equity' → 'BRK B'.

    Returns None for obviously-unparseable tickers (e.g., Bloomberg-internal
    transient identifiers ending in 'D' which typically correspond to
    delisted/acquired names with no current IBKR contract).
    """
    tokens = bbg_ticker.strip().split()
    if len(tokens) < 3 or tokens[-1] != "Equity":
        return None
    sym = tokens[0].replace("/", " ")          # BRK/B → BRK B
    # Bloomberg-internal IDs look like '0111145D' or '2613148D' — skip
    if sym[0].isdigit():
        return None
    return sym


# ───────────────────── IBKR pull ─────────────────────
def fetch_ibkr_costs(symbols: list[str], ib: IB,
                     pause_snap: float = 0.3,
                     pause_hist: float = 1.0,
                     hist_lookback_days: int = 40) -> pd.DataFrame:
    """For each symbol pull live bid/ask + shortability + 30d BID_ASK history."""
    rows = []
    n = len(symbols)
    print(f"Fetching IBKR data for {n} tickers …")
    for i, sym in enumerate(symbols, 1):
        contract = Stock(sym, "SMART", "USD", primaryExchange="")
        try:
            [qc] = ib.qualifyContracts(contract)
        except Exception as e:
            print(f"  [{i:>3}/{n}] {sym:<10}  qualify FAILED ({e})")
            rows.append({"symbol": sym, "status": "unqualified"})
            continue

        # ── (1) snapshot market data ──
        #   generic tick 236 = Shortable (free).  258 = "Shortable Shares" but
        #   triggers a Fundamentals subscription error on some accounts; omit.
        #   In delayed mode (mktDataType=3), fields arrive as .delayedBid/.delayedAsk.
        try:
            tkr = ib.reqMktData(qc, genericTickList="236", snapshot=True)
            deadline = time.time() + 4.0
            def _pair():
                b = tkr.bid if not math.isnan(tkr.bid) else getattr(tkr, "delayedBid", float("nan"))
                a = tkr.ask if not math.isnan(tkr.ask) else getattr(tkr, "delayedAsk", float("nan"))
                return b, a
            while time.time() < deadline:
                b, a = _pair()
                if not (math.isnan(b) or math.isnan(a)):
                    break
                ib.sleep(0.2)
            bid, ask = _pair()
            shortable     = getattr(tkr, "shortableShares", float("nan"))
            shortable_ind = float("nan")
            for attr in ("shortable", "shortableSharesRequested"):
                v = getattr(tkr, attr, None)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    shortable_ind = v
                    break
        except Exception as e:
            print(f"  [{i:>3}/{n}] {sym:<10}  snapshot FAILED ({e})")
            bid = ask = shortable = shortable_ind = float("nan")

        # ── (2) historical BID_ASK bars ──
        hist_spread_bps = float("nan")
        hist_mid = float("nan")
        try:
            bars = ib.reqHistoricalData(
                qc, endDateTime="", durationStr=f"{hist_lookback_days} D",
                barSizeSetting="1 day", whatToShow="BID_ASK",
                useRTH=True, formatDate=1,
            )
            if bars:
                dfh = util.df(bars)
                # bar.open = avg bid,  bar.close = avg ask
                mid = (dfh["open"] + dfh["close"]) / 2.0
                spr = (dfh["close"] - dfh["open"]).clip(lower=0)
                prop = spr / mid
                hist_spread_bps = float(prop.median() * 1e4)
                hist_mid = float(mid.iloc[-1])
        except Exception as e:
            print(f"  [{i:>3}/{n}] {sym:<10}  hist FAILED ({e})")

        # snapshot spread
        if bid and ask and bid > 0 and ask > 0 and ask >= bid:
            mid_snap = (bid + ask) / 2
            snap_spread_bps = (ask - bid) / mid_snap * 1e4
        else:
            mid_snap = float("nan")
            snap_spread_bps = float("nan")

        print(f"  [{i:>3}/{n}] {sym:<10}  bid={bid:<8} ask={ask:<8} "
              f"spread_snap={snap_spread_bps:>6.1f}bps   "
              f"spread_hist30d={hist_spread_bps:>6.1f}bps   "
              f"shortable_shares={shortable}")

        rows.append({
            "symbol":           sym,
            "bid":              bid,
            "ask":              ask,
            "mid_snap":         mid_snap,
            "mid_hist_last":    hist_mid,
            "spread_snap_bps":  snap_spread_bps,
            "spread_hist_bps":  hist_spread_bps,
            "shortable_shares": shortable,
            "shortable_ind":    shortable_ind,
            "status":           "ok",
        })
        time.sleep(pause_snap)
        if (i % 20) == 0:
            time.sleep(pause_hist)      # extra breath every 20 hist calls

    return pd.DataFrame(rows)


# ───────────────────── net-return re-estimate ─────────────────────
def reprice(preds: pd.DataFrame, ibkr_df: pd.DataFrame,
            aum_per_side: float, min_adv: float, adv_cap: float,
            c_impact: float) -> dict:
    """Re-estimate triple-filter net return using IBKR spreads on the CURRENT
    triple-filter universe.  Historical LS returns & turnover come from the
    backtest; only the per-name spread changes."""
    # rebuild triple filter
    preds = preds.copy()
    preds["mcap_b"] = preds.groupby("end_date", group_keys=False).apply(qtile, col="dv_60d")
    preds["mom_b"]  = preds.groupby("end_date", group_keys=False).apply(qtile, col="mom_12_1")
    preds["vol_b"]  = preds.groupby("end_date", group_keys=False).apply(qtile, col="vol_60d")
    mask = (
        (preds["mcap_b"] == "Low")
        & (preds["vol_b"] == "High")
        & (preds["mom_b"] == "Low")
    )
    sub = preds.loc[mask & (preds["dv_60d"] >= min_adv)].copy()

    # attach IBKR spread by symbol
    sub["symbol"] = sub["ticker"].apply(bbg_to_ibkr_symbol)
    spread_map = ibkr_df.set_index("symbol")["spread_hist_bps"].to_dict()
    sub["ibkr_spread_bps"] = sub["symbol"].map(spread_map)
    # fall back to snapshot if historical missing
    spread_snap_map = ibkr_df.set_index("symbol")["spread_snap_bps"].to_dict()
    sub["ibkr_spread_bps"] = sub["ibkr_spread_bps"].fillna(sub["symbol"].map(spread_snap_map))
    print(f"\nIBKR spreads matched: {sub['ibkr_spread_bps'].notna().sum()}/{len(sub)}")

    # For each month, build TOP/BOT, cap at adv_cap of ADV, compute cost
    rows = []
    for date, gdf in sub.groupby("end_date"):
        if len(gdf) < 2:
            continue
        rk = gdf["p_up_mean"].rank(method="first") / len(gdf)
        top = gdf[rk > 0.5].copy(); bot = gdf[rk <= 0.5].copy()

        def size(leg):
            if len(leg) == 0:
                return leg.assign(pos=np.nan)
            target = aum_per_side / len(leg)
            cap = adv_cap * leg["dv_60d"]
            pos = np.minimum(target, cap)
            for _ in range(20):
                filled = pos.sum()
                if filled >= aum_per_side - 1:
                    break
                head = (cap - pos).clip(lower=0)
                if head.sum() < 1:
                    break
                add = np.minimum(head / head.sum() * (aum_per_side - filled), head)
                pos = pos + add
                if add.sum() < 1:
                    break
            return leg.assign(pos=pos)

        top = size(top); bot = size(bot)
        tT, tB = top["pos"].sum(), bot["pos"].sum()
        if tT <= 0 or tB <= 0:
            continue
        # use IBKR spread where available; fall back to CS proportional at 40% haircut
        # (we don't have spread_60d here because ibkr_df is a point-in-time current
        # snapshot; for historical dates we assume spread structure is stable)
        def pos_wavg(leg, col):
            return (leg[col].fillna(0) * leg["pos"]).sum() / leg["pos"].sum()

        sp_top_bps = pos_wavg(top, "ibkr_spread_bps")
        sp_bot_bps = pos_wavg(bot, "ibkr_spread_bps")
        hs_top = (sp_top_bps / 1e4) / 2.0
        hs_bot = (sp_bot_bps / 1e4) / 2.0

        imp_top = (
            (c_impact * (top["vol_60d"] / np.sqrt(252)) * np.sqrt(top["pos"] / top["dv_60d"]))
            * top["pos"]
        ).sum() / tT
        imp_bot = (
            (c_impact * (bot["vol_60d"] / np.sqrt(252)) * np.sqrt(bot["pos"] / bot["dv_60d"]))
            * bot["pos"]
        ).sum() / tB

        rows.append({
            "end_date": date,
            "n_top": len(top), "n_bot": len(bot),
            "sp_top_bps": sp_top_bps, "sp_bot_bps": sp_bot_bps,
            "imp_top": imp_top, "imp_bot": imp_bot,
            "ow_top": hs_top + imp_top, "ow_bot": hs_bot + imp_bot,
            "TOP": (top["forward_return"] * top["pos"]).sum() / tT,
            "BOT": (bot["forward_return"] * bot["pos"]).sum() / tB,
            "top_set": frozenset(top["ticker"]),
            "bot_set": frozenset(bot["ticker"]),
        })
    ls = pd.DataFrame(rows).sort_values("end_date").reset_index(drop=True)
    ls["LS"] = ls["TOP"] - ls["BOT"]
    tt = [np.nan]; tb = [np.nan]
    for i in range(1, len(ls)):
        tt.append(len(ls.loc[i, "top_set"] ^ ls.loc[i - 1, "top_set"]) /
                  max(len(ls.loc[i, "top_set"]) + len(ls.loc[i - 1, "top_set"]), 1))
        tb.append(len(ls.loc[i, "bot_set"] ^ ls.loc[i - 1, "bot_set"]) /
                  max(len(ls.loc[i, "bot_set"]) + len(ls.loc[i - 1, "bot_set"]), 1))
    ls["turn_top"] = tt; ls["turn_bot"] = tb
    ls["cost_m"] = 2 * ls["turn_top"] * ls["ow_top"] + 2 * ls["turn_bot"] * ls["ow_bot"]
    ls["LS_net"] = ls["LS"] - ls["cost_m"]

    H = 20
    def stats(r):
        r = r.dropna(); m, s = r.mean(), r.std()
        return {
            "cagr":   np.expm1(m * 252 / H) * 100,
            "vol":    s * np.sqrt(252 / H) * 100,
            "sharpe": (m * 252 / H) / (s * np.sqrt(252 / H)) if s > 0 else np.nan,
            "t":      m / (s / np.sqrt(len(r))) if s > 0 else np.nan,
            "worst_m": r.min() * 100,
        }
    return {
        "ls_gross": stats(ls["LS"]),
        "ls_net":   stats(ls["LS_net"]),
        "avg_spread_top_bps": ls["sp_top_bps"].mean(),
        "avg_spread_bot_bps": ls["sp_bot_bps"].mean(),
        "avg_imp_top_bps":    ls["imp_top"].mean() * 1e4,
        "avg_imp_bot_bps":    ls["imp_bot"].mean() * 1e4,
        "avg_monthly_cost_bps": ls["cost_m"].mean() * 1e4,
        "annual_cost_pct":     ls["cost_m"].mean() * 12 * 100,
        "avg_turnover_top":    ls["turn_top"].mean(),
        "avg_turnover_bot":    ls["turn_bot"].mean(),
        "months": len(ls),
    }


# ───────────────────── main ─────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ohlcv", type=Path,
                    default=Path("data/r1000_ohlcv_database.parquet"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ib-host", default="127.0.0.1")
    ap.add_argument("--ib-port", type=int, default=7497,
                    help="TWS live=7496, TWS paper=7497, GW live=4001, GW paper=4002")
    ap.add_argument("--ib-client-id", type=int, default=42)
    ap.add_argument("--max-names", type=int, default=None,
                    help="cap # of names to query (testing). Default: all.")
    ap.add_argument("--aum-per-side", type=float, default=5_000_000)
    ap.add_argument("--min-adv", type=float, default=5_000_000)
    ap.add_argument("--adv-cap", type=float, default=0.10)
    ap.add_argument("--c-impact", type=float, default=1.5)
    ap.add_argument("--market-data-type", type=int, default=3,
                    help="1=live (needs subscriptions), 2=frozen, 3=delayed (free), "
                         "4=delayed-frozen. Default 3.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data + build current triple-filter universe ──
    print("Loading predictions + OHLC …")
    preds = pd.read_parquet(args.predictions)
    preds["end_date"] = pd.to_datetime(preds["end_date"])
    ohlcv = pd.read_parquet(args.ohlcv)
    feats = build_features(ohlcv)
    preds = preds.merge(feats, on=["ticker", "end_date"], how="left")

    last_date = preds["end_date"].max()
    cur = preds[preds["end_date"] == last_date].copy()
    # bucket CURRENT month
    cur["mcap_b"] = qtile(cur, "dv_60d")
    cur["mom_b"]  = qtile(cur, "mom_12_1")
    cur["vol_b"]  = qtile(cur, "vol_60d")
    triple = cur[(cur["mcap_b"] == "Low") & (cur["vol_b"] == "High") & (cur["mom_b"] == "Low")]
    triple = triple[triple["dv_60d"] >= args.min_adv]

    print(f"Latest end_date: {last_date.date()}   "
          f"triple-filter universe (ADV≥${args.min_adv/1e6:.0f}M): {len(triple)} names")

    syms = []
    for t in triple["ticker"]:
        s = bbg_to_ibkr_symbol(t)
        if s:
            syms.append(s)
    syms = sorted(set(syms))
    if args.max_names:
        syms = syms[:args.max_names]
    print(f"IBKR-parseable symbols to query: {len(syms)}")

    # ── Connect to IBKR ──
    print(f"Connecting to IBKR {args.ib_host}:{args.ib_port} clientId={args.ib_client_id} …")
    ib = IB()
    try:
        ib.connect(args.ib_host, args.ib_port, clientId=args.ib_client_id, timeout=20)
        ib.reqMarketDataType(args.market_data_type)
        print(f"Market data type set to {args.market_data_type} "
              f"({'LIVE' if args.market_data_type==1 else 'FROZEN' if args.market_data_type==2 else 'DELAYED' if args.market_data_type==3 else 'DELAYED-FROZEN'})")
    except Exception as e:
        print(f"ERROR — could not connect to IBKR: {e}")
        print("Is TWS or IB Gateway running?  API enabled?  Port correct?")
        return

    try:
        ibkr_df = fetch_ibkr_costs(syms, ib)
    finally:
        ib.disconnect()

    # ── Save raw IBKR data ──
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    xlsx = args.out_dir / f"ibkr_spreads_{stamp}.xlsx"
    ibkr_df.to_excel(xlsx, index=False)
    print(f"\nwrote {xlsx}")
    print(f"\nStatus counts:\n{ibkr_df['status'].value_counts().to_string()}")
    ok = ibkr_df[ibkr_df["status"] == "ok"]
    if len(ok):
        print(f"\nIBKR spread stats over {len(ok)} names:")
        for col in ["spread_snap_bps", "spread_hist_bps"]:
            s = ok[col].dropna()
            if len(s):
                print(f"  {col:20s}  n={len(s):>3}  mean={s.mean():6.1f}  median={s.median():6.1f}  "
                      f"p25={s.quantile(.25):6.1f}  p75={s.quantile(.75):6.1f}  p90={s.quantile(.90):6.1f}")

    # ── Re-price the historical strategy using IBKR spreads ──
    print("\nRe-pricing historical triple-filter LS with IBKR spreads …")
    res = reprice(preds, ibkr_df, args.aum_per_side, args.min_adv,
                  args.adv_cap, args.c_impact)

    summary_txt = [
        "=" * 78,
        "IBKR-PRICED triple-tier cost & net-return estimate",
        "=" * 78,
        f"AUM per side          : ${args.aum_per_side:,.0f}",
        f"Min ADV               : ${args.min_adv:,.0f}",
        f"Per-name ADV cap      : {args.adv_cap*100:.0f}% of 1-day volume",
        f"Impact coeff (Almgren): c = {args.c_impact}",
        f"Months in backtest    : {res['months']}",
        "",
        f"avg IBKR spread TOP   : {res['avg_spread_top_bps']:6.1f} bps full",
        f"avg IBKR spread BOT   : {res['avg_spread_bot_bps']:6.1f} bps full",
        f"avg impact  TOP       : {res['avg_imp_top_bps']:6.1f} bps one-way",
        f"avg impact  BOT       : {res['avg_imp_bot_bps']:6.1f} bps one-way",
        f"avg monthly turnover  : TOP {res['avg_turnover_top']*100:.1f}%   BOT {res['avg_turnover_bot']*100:.1f}%",
        f"avg monthly cost LS   : {res['avg_monthly_cost_bps']:.0f} bps",
        f"ANNUAL  cost LS       : {res['annual_cost_pct']:5.2f}%",
        "",
        "                       CAGR       vol    Sharpe     t     worst mo",
        f"  LS gross             {res['ls_gross']['cagr']:+6.1f}%   "
        f"{res['ls_gross']['vol']:5.1f}%   "
        f"{res['ls_gross']['sharpe']:+5.2f}   "
        f"{res['ls_gross']['t']:+5.2f}    "
        f"{res['ls_gross']['worst_m']:+6.1f}%",
        f"  LS net (IBKR spr)    {res['ls_net']['cagr']:+6.1f}%   "
        f"{res['ls_net']['vol']:5.1f}%   "
        f"{res['ls_net']['sharpe']:+5.2f}   "
        f"{res['ls_net']['t']:+5.2f}    "
        f"{res['ls_net']['worst_m']:+6.1f}%",
        "",
        "(Short-borrow cost NOT included; subtract borrow rate × notional",
        " annualised — typical small-cap loser: 100-500 bps/yr, some 'specials'",
        " 1000-5000 bps.  Pull from IBKR Stock Loan list for precise numbers.)",
        "=" * 78,
    ]
    summary = "\n".join(summary_txt)
    print("\n" + summary)
    (args.out_dir / "ibkr_repriced_summary.txt").write_text(summary + "\n")
    print(f"\nwrote {args.out_dir/'ibkr_repriced_summary.txt'}")


if __name__ == "__main__":
    main()
