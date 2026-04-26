"""
=============================================================================
SCRIPT NAME: backtest_liquid_grid.py
=============================================================================

INPUT FILES:
- --predictions PATH    predictions_monthly.parquet
- --ohlcv PATH          data/r1000_ohlcv_database.parquet

OUTPUT FILES:
- {out_dir}/liquid_grid_summary.xlsx   per-cell stats + cost drag
- {out_dir}/liquid_grid_portfolios.parquet

DESCRIPTION:
Searches the LIQUID end of R1000 for a tradable cell of the CNN signal:
a place where LS is positive & significant AND the names are cheap
enough to trade (spread ~3-10 bps, easy borrow).

At each month-end we bucket by:
  - dv_60d (small/mid/large $-volume)
  - vol_60d (low/mid/high realized vol)   OR
  - mom_12_1 (loser/neutral/winner)

Inside each 3×3 cell we form 50/50 top-half vs bot-half LS by p_up_mean.
Per cell we report:
  - mean universe, mean #top, months
  - TOP / BOT / LS / EW CAGR
  - LS Sharpe and Newey-West t
  - one-sided monthly turnover (ticker symmetric-diff / (|S|+|S_prev|))
  - realistic half-spread estimate (bps) keyed off $-volume tertile
  - cost drag per year and net LS CAGR

USAGE:
  python scripts/backtest_liquid_grid.py \
      --predictions runs/expanding/.../predictions_monthly.parquet \
      --out-dir runs/expanding/.../backtest_liquid_grid
=============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── feature builders ─────────────────────────────────────────────────────────

def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv[["Ticker", "Date", "Close", "Volume", "AdjClose"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])
    g = df.groupby("Ticker")
    df["dv"] = df["Close"] * df["Volume"]
    df["dv_60d"] = g["dv"].transform(lambda s: s.rolling(60, min_periods=15).mean())
    df["logp"] = np.log(df["AdjClose"].replace(0, np.nan))
    df["mom_12_1"] = g["logp"].shift(21) - g["logp"].shift(252 + 21)
    df["ret"] = g["AdjClose"].pct_change()
    df["vol_60d"] = g["ret"].transform(
        lambda s: s.rolling(60, min_periods=20).std() * np.sqrt(252))
    return df[["Ticker", "Date", "dv_60d", "mom_12_1", "vol_60d"]].rename(
        columns={"Ticker": "ticker", "Date": "end_date"})


def bucket3(df: pd.DataFrame, col: str) -> pd.Series:
    labels = ["Low", "Mid", "High"]

    def assign(g):
        if g[col].notna().sum() < 3:
            return pd.Series([pd.NA] * len(g), index=g.index)
        return pd.qcut(g[col].rank(method="first"), 3,
                       labels=labels, duplicates="drop")
    return df.groupby("end_date", group_keys=False).apply(assign)


# ── portfolio builder ────────────────────────────────────────────────────────

def build_5050(preds: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    rows = []
    sub = preds.loc[mask]
    for date, g in sub.groupby("end_date"):
        if len(g) < 4:
            continue
        n = len(g)
        rank = g["p_up_mean"].rank(method="first") / n
        top = g[rank > 0.5]
        bot = g[rank <= 0.5]
        rows.append({
            "end_date": date,
            "n": n,
            "n_top": len(top),
            "TOP": top["forward_return"].mean(),
            "BOT": bot["forward_return"].mean(),
            "EW":  g["forward_return"].mean(),
            "top_set": frozenset(top["ticker"]),
            "bot_set": frozenset(bot["ticker"]),
        })
    df = pd.DataFrame(rows).sort_values("end_date").reset_index(drop=True)
    if not len(df):
        return df
    df["LS"] = df["TOP"] - df["BOT"]
    top_turn = [np.nan]
    bot_turn = [np.nan]
    for i in range(1, len(df)):
        pt, pb = df.loc[i - 1, "top_set"], df.loc[i - 1, "bot_set"]
        ct, cb = df.loc[i,     "top_set"], df.loc[i,     "bot_set"]
        top_turn.append(len(ct ^ pt) / max(len(ct) + len(pt), 1))
        bot_turn.append(len(cb ^ pb) / max(len(cb) + len(pb), 1))
    df["turn_top"] = top_turn
    df["turn_bot"] = bot_turn
    return df.drop(columns=["top_set", "bot_set"])


# ── stats helpers ────────────────────────────────────────────────────────────

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


# Realistic half-spread by $-volume tertile (bps, one-sided)
# Low-dv (≈ smallest liquid R1000): ~25 bps half-spread
# Mid-dv: ~8 bps
# High-dv (mega-caps): ~2 bps
SPREAD_BPS_BY_DV = {"Low": 25.0, "Mid": 8.0, "High": 2.5}


def cell_stats(name: str, cell: pd.DataFrame, dv_tier: str) -> dict:
    if len(cell) == 0:
        return {}
    r = cell["LS"].dropna()
    top = cell["TOP"].dropna()
    bot = cell["BOT"].dropna()
    ew = cell["EW"].dropna()
    m = r.mean(); v = r.std()
    ls_cagr = (np.exp(m * 12) - 1) * 100
    ls_vol = v * np.sqrt(12) * 100
    ls_sh = (m * 12) / (v * np.sqrt(12)) if v > 0 else np.nan
    ls_t = newey_west_t(r.to_numpy())
    top_cagr = (np.exp(top.mean() * 12) - 1) * 100
    bot_cagr = (np.exp(bot.mean() * 12) - 1) * 100
    ew_cagr = (np.exp(ew.mean() * 12) - 1) * 100
    turn_top = cell["turn_top"].mean()
    turn_bot = cell["turn_bot"].mean()
    spread_bps = SPREAD_BPS_BY_DV.get(dv_tier, 15.0)
    # two sides × 12 months × turnover × half-spread
    cost_drag_pct = 12 * 2 * (turn_top + turn_bot) / 2 * (spread_bps / 10000) * 100
    net_ls_cagr = ls_cagr - cost_drag_pct
    return dict(
        cell=name,
        dv_tier=dv_tier,
        mean_univ=cell["n"].mean(),
        mean_top=cell["n_top"].mean(),
        months=len(cell),
        top_cagr=top_cagr,
        bot_cagr=bot_cagr,
        ls_cagr=ls_cagr,
        ls_vol=ls_vol,
        ls_sharpe=ls_sh,
        nw_t=ls_t,
        ew_cagr=ew_cagr,
        turn_top=turn_top,
        turn_bot=turn_bot,
        spread_bps=spread_bps,
        cost_drag_pct=cost_drag_pct,
        net_ls_cagr=net_ls_cagr,
    )


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ohlcv", type=Path,
                    default=Path("data/r1000_ohlcv_database.parquet"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OHLCV + predictions …")
    ohlcv = pd.read_parquet(args.ohlcv)
    feats = compute_features(ohlcv)
    preds = pd.read_parquet(args.predictions)
    preds["end_date"] = pd.to_datetime(preds["end_date"])
    preds = preds.merge(feats, on=["ticker", "end_date"], how="left")

    print("Bucketing …")
    preds["dv_b"]  = bucket3(preds, "dv_60d")
    preds["mom_b"] = bucket3(preds, "mom_12_1")
    preds["vol_b"] = bucket3(preds, "vol_60d")
    univ = preds[["dv_b", "mom_b", "vol_b"]].notna().all(axis=1)
    p = preds[univ].copy()
    print(f"Rows with all 3 features: {len(p):,}")

    # Grid A: dv × vol
    grid_dv_vol = []
    for dv in ["Low", "Mid", "High"]:
        for vb in ["Low", "Mid", "High"]:
            m = (p["dv_b"] == dv) & (p["vol_b"] == vb)
            port = build_5050(p, m.to_numpy())
            if len(port) == 0: continue
            grid_dv_vol.append(cell_stats(f"dv={dv}/vol={vb}", port, dv))
    grid_dv_vol = pd.DataFrame(grid_dv_vol)

    # Grid B: dv × mom
    grid_dv_mom = []
    for dv in ["Low", "Mid", "High"]:
        for mb in ["Low", "Mid", "High"]:
            m = (p["dv_b"] == dv) & (p["mom_b"] == mb)
            port = build_5050(p, m.to_numpy())
            if len(port) == 0: continue
            grid_dv_mom.append(cell_stats(f"dv={dv}/mom={mb}", port, dv))
    grid_dv_mom = pd.DataFrame(grid_dv_mom)

    # Grid C: dv × vol × mom  (27 cells)
    grid_triple = []
    for dv in ["Low", "Mid", "High"]:
        for vb in ["Low", "Mid", "High"]:
            for mb in ["Low", "Mid", "High"]:
                m = ((p["dv_b"] == dv) & (p["vol_b"] == vb) & (p["mom_b"] == mb))
                port = build_5050(p, m.to_numpy())
                if len(port) == 0:
                    continue
                grid_triple.append(cell_stats(
                    f"dv={dv}/vol={vb}/mom={mb}", port, dv))
    grid_triple = pd.DataFrame(grid_triple)

    # Marginal dv tertiles + full universe
    marginals = []
    for dv in ["Low", "Mid", "High"]:
        m = (p["dv_b"] == dv)
        port = build_5050(p, m.to_numpy())
        marginals.append(cell_stats(f"dv={dv} (all)", port, dv))
    port_all = build_5050(p, np.ones(len(p), dtype=bool))
    marginals.append(cell_stats("all R1000", port_all, "Mid"))
    marginals = pd.DataFrame(marginals)

    # Print
    cols_show = ["cell", "mean_univ", "mean_top", "months",
                 "ls_cagr", "ls_sharpe", "nw_t",
                 "turn_top", "spread_bps", "cost_drag_pct", "net_ls_cagr"]
    print("\n=== Marginal dv tertiles ===")
    print(marginals[cols_show].round(2).to_string(index=False))
    print("\n=== Grid A: dv × vol ===")
    print(grid_dv_vol[cols_show].round(2).to_string(index=False))
    print("\n=== Grid B: dv × mom ===")
    print(grid_dv_mom[cols_show].round(2).to_string(index=False))
    print("\n=== Grid C: dv × vol × mom (27 cells) ===")
    print(grid_triple[cols_show].round(2).to_string(index=False))

    # Rank best tradable cells: net_ls_cagr > 0 AND nw_t >= 2
    all_cells = pd.concat(
        [grid_dv_vol, grid_dv_mom, grid_triple, marginals], ignore_index=True)
    good = all_cells[(all_cells["net_ls_cagr"] > 0) & (all_cells["nw_t"] >= 2.0)]
    good = good.sort_values("net_ls_cagr", ascending=False)
    print("\n=== Tradable cells (net LS > 0 and t ≥ 2), sorted by net LS ===")
    print(good[cols_show].round(2).to_string(index=False))

    # Save
    with pd.ExcelWriter(args.out_dir / "liquid_grid_summary.xlsx",
                         engine="openpyxl") as xw:
        marginals.round(3).to_excel(xw, sheet_name="marginals", index=False)
        grid_dv_vol.round(3).to_excel(xw, sheet_name="dv_x_vol", index=False)
        grid_dv_mom.round(3).to_excel(xw, sheet_name="dv_x_mom", index=False)
        grid_triple.round(3).to_excel(xw, sheet_name="dv_x_vol_x_mom", index=False)
        good.round(3).to_excel(xw, sheet_name="tradable", index=False)
    print(f"\nwrote {args.out_dir / 'liquid_grid_summary.xlsx'}")


if __name__ == "__main__":
    main()
