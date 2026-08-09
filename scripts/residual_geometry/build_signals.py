"""
Build residual / Dixon geometry / FOMC-gated signals for Pattern monthly panel.

INPUT:
  - runs/expanding/.../predictions_monthly.parquet
  - data/r1000_ohlcv_spliced.parquet
  - data/fomc_dates.csv

OUTPUT:
  - {out_dir}/residual_signals.parquet
  - {out_dir}/feature_ic.csv
  - {out_dir}/build_meta.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("residual_geometry")

DEFAULT_PRED = (
    ROOT / "runs" / "expanding" / "20260419_174908_cdef6809" / "predictions_monthly.parquet"
)
DEFAULT_OHLCV = ROOT / "data" / "r1000_ohlcv_spliced.parquet"
DEFAULT_FOMC = ROOT / "data" / "fomc_dates.csv"
MIN_PRICE = 5.0
WINDOW = 20  # I20

# higher = buy after orientation
SIGN_BASE = {
    "rev_1m": -1,
    "rev_5d": -1,
    "mom_12_1": +1,
    "vol_60d": -1,
    "combo": +1,
}
BASELINES = ["rev_1m", "rev_5d", "mom_12_1", "vol_60d", "combo"]


def newey_west_t(x: np.ndarray, lag: int = 3) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean()
    xc = x - mu
    s = (xc ** 2).mean()
    for k in range(1, lag + 1):
        if k >= n:
            break
        s += 2 * (1 - k / (lag + 1)) * (xc[k:] * xc[:-k]).mean()
    se = np.sqrt(max(s, 0) / n)
    return float(mu / se) if se > 0 else float("nan")


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    if not sd or not np.isfinite(sd) or sd <= 0:
        return s * 0.0
    return (s - s.mean()) / sd


def cs_residual(y: pd.Series, X_cols: list[pd.Series]) -> np.ndarray:
    """Residual of z(y) on z(X) with intercept; NaN-safe."""
    n = len(y)
    yz = zscore(y).to_numpy(dtype=float)
    cols = [zscore(c).to_numpy(dtype=float) for c in X_cols]
    X = np.column_stack([np.ones(n)] + cols)
    resid = np.full(n, np.nan)
    ok = np.isfinite(X).all(1) & np.isfinite(yz)
    if ok.sum() <= X.shape[1] + 3:
        return resid
    beta, *_ = np.linalg.lstsq(X[ok], yz[ok], rcond=None)
    resid[ok] = yz[ok] - X[ok] @ beta
    return resid


def build_baseline_features(ohlcv_path: Path) -> pd.DataFrame:
    o = pd.read_parquet(
        ohlcv_path, columns=["Ticker", "Date", "Close", "Volume", "AdjClose"]
    )
    o["Date"] = pd.to_datetime(o["Date"])
    o = o.sort_values(["Ticker", "Date"])
    g = o.groupby("Ticker", sort=False)
    o["logp"] = np.log(o["AdjClose"].replace(0, np.nan))
    o["rev_1m"] = o["logp"] - g["logp"].shift(21)
    o["rev_5d"] = o["logp"] - g["logp"].shift(5)
    o["mom_12_1"] = g["logp"].shift(21) - g["logp"].shift(252 + 21)
    o["ret"] = g["AdjClose"].pct_change(fill_method=None)
    o["vol_60d"] = g["ret"].transform(
        lambda s: s.rolling(60, min_periods=20).std() * np.sqrt(252)
    )
    o["dv"] = o["Close"] * o["Volume"]
    o["dv_60d"] = g["dv"].transform(lambda s: s.rolling(60, min_periods=15).mean())
    return o.rename(columns={"Ticker": "ticker", "Date": "end_date"})[
        ["ticker", "end_date", "Close", "rev_1m", "rev_5d", "mom_12_1", "vol_60d", "dv_60d"]
    ]


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full_like(a, 0.5, dtype=float)
    return (a - lo) / (hi - lo)


def geometry_for_window(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray
) -> dict[str, float]:
    """
    Dixon-style geometry on one I20 window.
    o,h,l,c,v length WINDOW; prices should be split-adjusted coherent.
    """
    # return-reconstruct close path from first close (JKX spirit)
    if not np.isfinite(c).all() or c[0] <= 0:
        return {k: np.nan for k in (
            "level_c", "slope_c", "curve_c", "level_rng", "slope_rng",
            "level_v", "slope_v", "rlc", "rlv", "rlcc", "form_ret")}
    rets = np.diff(c) / c[:-1]
    p = np.empty(WINDOW, dtype=float)
    p[0] = 1.0
    for i in range(1, WINDOW):
        p[i] = p[i - 1] * (1.0 + rets[i - 1]) if np.isfinite(rets[i - 1]) else p[i - 1]
    # scale ohl to same path
    scale = p / c
    oo, hh, ll, cc = o * scale, h * scale, l * scale, p
    xc = _norm01(cc)
    rng = _norm01(hh - ll)
    vv = _norm01(v.astype(float))
    mid = (WINDOW - 1) // 2  # day index 0..19
    last, first = WINDOW - 1, 0

    def lev(x):
        return float(np.nanmean(x))

    def slope(x):
        return float(x[last] - x[first])

    def curve(x):
        return float((x[last] - x[mid]) - (x[mid] - x[first]))

    # relative last measures vs mean of prior days
    prior = slice(0, last)
    rlc = float(xc[last] - np.nanmean(xc[prior]))
    rlv = float(vv[last] - np.nanmean(vv[prior]))
    # close vs mid-range of bar, relative to prior mean of same
    cl_mid = (cc - 0.5 * (hh + ll)) / np.maximum(hh - ll, 1e-8)
    cl_mid_n = _norm01(cl_mid)
    rlcc = float(cl_mid_n[last] - np.nanmean(cl_mid_n[prior]))
    form_ret = float(np.log(cc[last] / cc[first])) if cc[first] > 0 else np.nan

    return {
        "level_c": lev(xc),
        "slope_c": slope(xc),
        "curve_c": curve(xc),
        "level_rng": lev(rng),
        "slope_rng": slope(rng),
        "level_v": lev(vv),
        "slope_v": slope(vv),
        "rlc": rlc,
        "rlv": rlv,
        "rlcc": rlcc,
        "form_ret": form_ret,
    }


def build_geometry(ohlcv_path: Path, panel: pd.DataFrame) -> pd.DataFrame:
    """Compute I20 Dixon geometry for each (ticker, end_date) in panel."""
    o = pd.read_parquet(
        ohlcv_path,
        columns=["Ticker", "Date", "Open", "High", "Low", "Close", "Volume", "AdjClose"],
    )
    o["Date"] = pd.to_datetime(o["Date"])
    o = o.sort_values(["Ticker", "Date"])
    # use AdjClose to back-adjust OHLC ratios
    o["factor"] = o["AdjClose"] / o["Close"].replace(0, np.nan)
    for col in ["Open", "High", "Low", "Close"]:
        o[col] = o[col] * o["factor"]

    need = panel[["ticker", "end_date"]].drop_duplicates()
    need["end_date"] = pd.to_datetime(need["end_date"])
    rows = []

    for ticker, g in o.groupby("Ticker", sort=False):
        sub_need = need[need["ticker"] == ticker]
        if sub_need.empty:
            continue
        dates = g["Date"].to_numpy()
        arr = g[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float)
        # map end_date -> row index
        date_to_i = {pd.Timestamp(d).to_datetime64(): i for i, d in enumerate(dates)}
        for ed in sub_need["end_date"].to_numpy():
            ed64 = pd.Timestamp(ed).to_datetime64()
            i = date_to_i.get(ed64)
            if i is None or i + 1 < WINDOW:
                continue
            w = arr[i - WINDOW + 1 : i + 1]
            if w.shape[0] != WINDOW or not np.isfinite(w[:, 3]).all():
                continue
            feat = geometry_for_window(w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4])
            feat["ticker"] = ticker
            feat["end_date"] = pd.Timestamp(ed)
            rows.append(feat)

    if not rows:
        raise RuntimeError("No geometry features computed")
    geo = pd.DataFrame(rows)
    log.info("Geometry rows: %s", f"{len(geo):,}")
    return geo


def fomc_mask(end_dates: pd.Series, ohlcv_path: Path, fomc_path: Path) -> pd.Series:
    """True if end_date within [-2, +5] trading days of an FOMC decision date."""
    fomc = pd.read_csv(fomc_path)
    fomc_d = pd.to_datetime(fomc["date"]).sort_values().to_numpy(dtype="datetime64[ns]")
    cal = (
        pd.read_parquet(ohlcv_path, columns=["Date"])["Date"]
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype="datetime64[ns]")
    )
    # precompute set of attention formation dates
    attention = set()
    for fd in fomc_d:
        pos = np.searchsorted(cal, fd, side="left")
        # if FOMC not on calendar, use next trading day as anchor
        if pos >= len(cal):
            continue
        # [-2, +5] trading days around FOMC date position
        for off in range(-2, 6):
            j = pos + off
            if 0 <= j < len(cal):
                attention.add(pd.Timestamp(cal[j]))

    eds = pd.to_datetime(end_dates)
    return eds.map(lambda d: pd.Timestamp(d) in attention)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    ap.add_argument("--fomc", type=Path, default=DEFAULT_FOMC)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-price", type=float, default=MIN_PRICE)
    ap.add_argument("--min-names", type=int, default=50)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Loading predictions …")
    p = pd.read_parquet(args.predictions)
    p["end_date"] = pd.to_datetime(p["end_date"])
    log.info("Baseline features …")
    feats = build_baseline_features(args.ohlcv)
    df = p.merge(feats, on=["ticker", "end_date"], how="left")
    df = df[df["Close"] >= args.min_price]
    df = df.dropna(
        subset=["forward_return", "p_up_mean", "rev_1m", "rev_5d", "mom_12_1", "vol_60d"]
    )

    log.info("Dixon geometry (I20) …")
    geo = build_geometry(args.ohlcv, df)
    df = df.merge(geo, on=["ticker", "end_date"], how="left")

    log.info("FOMC attention mask …")
    df["fomc_window"] = fomc_mask(df["end_date"], args.ohlcv, args.fomc)

    # per-date signals
    parts = []
    ic_rows = []
    for dt, g in df.groupby("end_date"):
        if len(g) < args.min_names:
            continue
        g = g.copy()
        g["combo"] = zscore(-g["rev_1m"]) + zscore(-g["vol_60d"])

        # oriented baselines
        for s in ["rev_1m", "rev_5d", "mom_12_1", "vol_60d"]:
            g[f"{s}_or"] = SIGN_BASE[s] * g[s]
        g["combo_or"] = g["combo"]

        # geometry: Dixon finds high last-bar close (high rlc/rlcc) → lower P(up)
        # so oriented buy signal flips: lower rlc/rlcc = buy (reversal)
        g["rlcc_or"] = -g["rlcc"]
        g["rlc_or"] = -g["rlc"]
        g["form_ret_or"] = -g["form_ret"]  # short formation winners (reversal)
        g["slope_c_or"] = -g["slope_c"]
        g["geom_score"] = (
            zscore(g["rlcc_or"].fillna(0))
            + zscore(g["rlc_or"].fillna(0))
            + zscore(g["form_ret_or"].fillna(0))
            + 0.5 * zscore(g["slope_c_or"].fillna(0))
        )

        base_cols = [g[f"{s}_or"] for s in ["rev_1m", "rev_5d", "mom_12_1", "vol_60d"]] + [
            g["combo_or"]
        ]
        geom_cols = [
            g["rlcc"].fillna(g["rlcc"].median()),
            g["rlc"].fillna(g["rlc"].median()),
            g["rlv"].fillna(g["rlv"].median()),
            g["form_ret"].fillna(g["form_ret"].median()),
            g["slope_c"].fillna(g["slope_c"].median()),
            g["curve_c"].fillna(g["curve_c"].median()),
            g["level_c"].fillna(g["level_c"].median()),
            g["slope_v"].fillna(g["slope_v"].median()),
        ]

        g["p_up_resid"] = cs_residual(g["p_up_mean"], base_cols)
        g["p_up_resid_geom"] = cs_residual(g["p_up_mean"], geom_cols)
        g["p_up_resid_full"] = cs_residual(g["p_up_mean"], base_cols + geom_cols)
        g["p_up_resid_nofomc"] = g["p_up_resid"].where(~g["fomc_window"], np.nan)
        g["geom_score_nofomc"] = g["geom_score"].where(~g["fomc_window"], np.nan)

        parts.append(g)

        r = g["forward_return"]
        rec = {"end_date": dt, "n": len(g), "n_fomc": int(g["fomc_window"].sum())}
        for col, orient in [
            ("p_up_mean", 1),
            ("p_up_resid", 1),
            ("p_up_resid_geom", 1),
            ("p_up_resid_full", 1),
            ("geom_score", 1),
            ("rlcc_or", 1),
            ("rev_5d_or", 1),
        ]:
            if col not in g.columns:
                continue
            rec[col] = orient * g[col].corr(r, method="spearman")
        # nofomc IC only among non-FOMC months: compute later globally
        ic_rows.append(rec)

    out = pd.concat(parts, ignore_index=True)
    keep = [
        "ticker", "end_date", "forward_return", "p_up_mean",
        "rev_1m", "rev_5d", "mom_12_1", "vol_60d", "combo",
        "rlc", "rlv", "rlcc", "form_ret", "slope_c", "curve_c", "level_c",
        "geom_score", "p_up_resid", "p_up_resid_geom", "p_up_resid_full",
        "p_up_resid_nofomc", "geom_score_nofomc", "fomc_window",
        "rlcc_or", "rev_5d_or",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep]

    sig_path = args.out_dir / "residual_signals.parquet"
    out.to_parquet(sig_path, index=False)

    ic = pd.DataFrame(ic_rows).sort_values("end_date")
    # IC on non-FOMC months only for gated residual (use all rows with non-null score)
    nofomc_ics = []
    for dt, g in out.groupby("end_date"):
        if g["fomc_window"].all():
            continue
        sub = g.loc[~g["fomc_window"]]
        if len(sub) < args.min_names:
            continue
        icv = sub["p_up_resid"].corr(sub["forward_return"], method="spearman")
        if np.isfinite(icv):
            nofomc_ics.append({"end_date": dt, "p_up_resid_nofomc": icv})
    nofomc_ic = pd.DataFrame(nofomc_ics)

    score_cols = [
        c for c in [
            "p_up_mean", "p_up_resid", "p_up_resid_geom", "p_up_resid_full",
            "geom_score", "rlcc_or", "rev_5d_or",
        ]
        if c in ic.columns
    ]
    ic_tab = []
    for s in score_cols:
        series = ic[s].dropna()
        ic_tab.append({
            "signal": s,
            "mean_IC": float(series.mean()),
            "IC_t_NW": newey_west_t(series.to_numpy()),
            "pct_months_pos": float((series > 0).mean()),
            "n_months": int(len(series)),
        })
    if len(nofomc_ic):
        s = nofomc_ic["p_up_resid_nofomc"]
        ic_tab.append({
            "signal": "p_up_resid_nofomc_months",
            "mean_IC": float(s.mean()),
            "IC_t_NW": newey_west_t(s.to_numpy()),
            "pct_months_pos": float((s > 0).mean()),
            "n_months": int(len(s)),
        })
    ic_tab = pd.DataFrame(ic_tab).sort_values("IC_t_NW", ascending=False)
    ic_tab.to_csv(args.out_dir / "feature_ic.csv", index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(out)),
        "n_months": int(out["end_date"].nunique()),
        "frac_fomc_rows": float(out["fomc_window"].mean()),
        "n_fomc_dates": int(pd.read_csv(args.fomc).shape[0]),
        "ic_table": ic_tab.to_dict("records"),
    }
    (args.out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Wrote %s\n%s", sig_path, ic_tab.to_string(index=False))
    return sig_path


if __name__ == "__main__":
    main()
