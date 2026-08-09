"""
=============================================================================
SCRIPT NAME: gbdt_label_bakeoff.py
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_ohlcv_spliced.parquet
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/expanding/20260419_174908_cdef6809/predictions_monthly.parquet
      used ONLY for its (ticker, end_date) formation grid and `window` column,
      so the GBDT is trained and tested on exactly the CNN's schedule
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_classifications.parquet
      BICS sectors, for the L3 factor-neutral label

OUTPUT FILES:
- {out_dir}/gbdt_predictions.parquet   one score column per (label, seed) plus
                                       the 5-seed ensemble mean per label
- {out_dir}/bakeoff_meta.json          features, params, purge audit, timings

DESCRIPTION:
STAGE 2 of PREREGISTRATION.md — the pre-registered label bake-off (§7, L0-L3).

Stage 0/1 established that the CNN's binary sign label trains it to rank by
mu/sigma rather than mu: the score predicts forward realized VOLATILITY at
IC 0.097 (t 19.1) and forward RETURN at only 0.010 (t 2.60). This asks whether
a corrected objective recovers content the hijacked one never learned.

A gradient-boosted tree is used as the platform rather than the CNN because
each fit takes seconds instead of hours, so all four labels can be tested
properly. It also answers a question never asked here: whether the image
representation earns its keep at all. NOTE the asymmetry when reading the
result -- GBDT >= CNN is strong evidence the images are unnecessary, but
GBDT < CNN is NOT proof they are necessary, since it may only mean these
hand-built features are worse than what the CNN learns.

THE FOUR LABELS (§7, fixed in advance):
  L0  binary sign of the 20-day forward return           (current production)
  L1  binary sign of the market-excess forward return
  L2  cross-sectional rank of the forward return
  L3  cross-sectional rank of the FACTOR-NEUTRAL forward return
      (residual on vol, size, sector, momentum, reversal)

Labels are constructed from data available only after the fact, which is
legitimate -- a training target is always known in hindsight. No label
information reaches the features.

EARLY STOPPING ON VALIDATION IC, for every label. The production CNN selects on
validation cross-entropy, which is not the objective and which Stage 0 showed
barely moves (median 4 epochs, 25% of models peaking at epoch 1). Ranking IC is
what the signal is actually used for, and it makes the four labels comparable
despite their different natural losses.

PURGE: replicates the production guard -- 39 trading days (window + horizon - 1)
on the DAILY calendar between train/val and val/test. Counting the purge in
monthly observations instead of trading days would silently drop 39 MONTHS, so
the boundary is computed on the daily index and audited in the output.

USAGE:
  python3 scripts/gbdt_label_bakeoff.py --out-dir runs/experiments/stage2_gbdt
  python3 scripts/gbdt_label_bakeoff.py --out-dir ... --seeds 5 --labels L0,L1,L2,L3
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
OHLCV = ROOT / "data" / "r1000_ohlcv_spliced.parquet"
GRID = ROOT / "runs" / "expanding" / "20260419_174908_cdef6809" / "predictions_monthly.parquet"
CLASSES = ROOT / "data" / "r1000_classifications.parquet"

WINDOW, HORIZON = 20, 20
PURGE_TD = WINDOW + HORIZON - 1          # 39 trading days, as pattern/cli.py:132
MIN_PRICE = 5.0
VAL_FRACTION = 0.30
START_YEAR = 1996


# ── features ─────────────────────────────────────────────────────────────────

def build_features(px_path: Path) -> pd.DataFrame:
    """Everything visible in the CNN's 20-day image, as columns.

    Price-path shape, daily ranges, close-location, volume texture and the
    moving-average relationship -- the same information content, scale-free.
    """
    o = pd.read_parquet(px_path, columns=["Ticker", "Date", "Open", "High",
                                          "Low", "Close", "Volume", "AdjClose"])
    o["Date"] = pd.to_datetime(o["Date"])
    o = o.sort_values(["Ticker", "Date"])
    g = o.groupby("Ticker")

    o["lp"] = np.log(o["AdjClose"].where(o["AdjClose"] > 0))
    o["r1"] = g["lp"].diff()
    f = pd.DataFrame(index=o.index)

    # short-horizon path: the last five daily returns, individually
    for k in range(1, 6):
        f[f"r_lag{k}"] = g["r1"].shift(k - 1)
    # cumulative shape
    for k in (5, 10, 20):
        f[f"ret_{k}"] = o["lp"] - g["lp"].shift(k)
    # dispersion / texture
    f["vol_20"] = g["r1"].transform(lambda s: s.rolling(20, min_periods=12).std())
    f["vol_5"] = g["r1"].transform(lambda s: s.rolling(5, min_periods=4).std())
    f["vol_ratio"] = f["vol_5"] / f["vol_20"].replace(0, np.nan)
    f["skew_20"] = g["r1"].transform(lambda s: s.rolling(20, min_periods=12).skew())
    f["updays_20"] = g["r1"].transform(lambda s: (s > 0).rolling(20, min_periods=12).mean())
    # intraday bar geometry
    rng = (o["High"] - o["Low"]).where(lambda s: s > 0)
    f["rng_rel"] = (rng / o["Close"]).groupby(o["Ticker"]).transform(
        lambda s: s.rolling(20, min_periods=12).mean())
    clv = (o["Close"] - 0.5 * (o["High"] + o["Low"])) / rng
    f["clv_last"] = clv
    f["clv_mean20"] = clv.groupby(o["Ticker"]).transform(
        lambda s: s.rolling(20, min_periods=12).mean())
    f["gap_last"] = np.log(o["Open"].where(o["Open"] > 0)) - g["lp"].shift(1)
    # volume texture
    lv = np.log(o["Volume"].where(o["Volume"] > 0))
    f["vlz_last"] = lv - lv.groupby(o["Ticker"]).transform(
        lambda s: s.rolling(20, min_periods=12).mean())
    f["vl_trend"] = (lv.groupby(o["Ticker"]).transform(lambda s: s.rolling(5, min_periods=3).mean())
                     - lv.groupby(o["Ticker"]).transform(lambda s: s.rolling(20, min_periods=12).mean()))
    # moving-average relationship (the MA line drawn in the image)
    ma = g["AdjClose"].transform(lambda s: s.rolling(20, min_periods=12).mean())
    f["dist_ma"] = (o["AdjClose"] - ma) / ma.replace(0, np.nan)
    # drawdown within the window
    roll_max = g["lp"].transform(lambda s: s.rolling(20, min_periods=12).max())
    f["dd_20"] = o["lp"] - roll_max

    f["ticker"], f["end_date"] = o["Ticker"], o["Date"]
    f["Close"], f["_lp"] = o["Close"], o["lp"]
    return f.reset_index(drop=True)


FEATURES = ["r_lag1", "r_lag2", "r_lag3", "r_lag4", "r_lag5",
            "ret_5", "ret_10", "ret_20", "vol_20", "vol_5", "vol_ratio",
            "skew_20", "updays_20", "rng_rel", "clv_last", "clv_mean20",
            "gap_last", "vlz_last", "vl_trend", "dist_ma", "dd_20"]


# ── labels ───────────────────────────────────────────────────────────────────

def _xs_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) - 0.5


def build_labels(p: pd.DataFrame) -> pd.DataFrame:
    """L0-L3 exactly as pre-registered."""
    p = p.copy()
    p["L0"] = (p["forward_return"] > 0).astype(float)

    mkt = p.groupby("end_date")["forward_return"].transform("mean")
    p["L1"] = (p["forward_return"] - mkt > 0).astype(float)

    p["L2"] = p.groupby("end_date", group_keys=False)["forward_return"].apply(_xs_rank)

    # L3: residual of forward return on the free factors, then ranked
    ctrl = ["vol_20", "ret_20", "ret_5", "log_dv"]
    sec = [c for c in p.columns if c.startswith("sec_")]
    res = np.full(len(p), np.nan)
    for dt, g in p.groupby("end_date"):
        gg = g.dropna(subset=["forward_return"] + ctrl)
        if len(gg) < 50:
            continue
        X = np.column_stack([np.ones(len(gg))]
                            + [_xs_rank(gg[c]).to_numpy() for c in ctrl]
                            + [gg[c].to_numpy() for c in sec])
        y = gg["forward_return"].to_numpy()
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        if ok.sum() < X.shape[1] + 20:
            continue
        b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        r = np.full(len(gg), np.nan)
        r[ok] = y[ok] - X[ok] @ b
        res[gg.index.to_numpy()] = r
    p["_l3res"] = res
    p["L3"] = p.groupby("end_date", group_keys=False)["_l3res"].apply(_xs_rank)
    return p


# ── splits: mirror the production purge, on the DAILY calendar ───────────────

def window_splits(dates: np.ndarray, cal: np.ndarray, test_year: int):
    """(train_dates, val_dates, test_dates) for one expanding window."""
    yr = pd.DatetimeIndex(dates).year
    tv = np.sort(dates[(yr >= START_YEAR) & (yr <= test_year - 1)])
    te = np.sort(dates[yr == test_year])
    if len(tv) < 12 or len(te) == 0:
        return None
    cut = max(1, min(int(np.floor(len(tv) * (1 - VAL_FRACTION))), len(tv) - 1))
    val_start = tv[cut]
    val = tv[tv >= val_start]
    train = tv[tv < val_start]
    # purge on the DAILY calendar -- counting in monthly rows would drop 39 months
    vi = int(np.searchsorted(cal, val_start))
    if vi - PURGE_TD <= 0:
        return None
    train = train[train <= cal[vi - PURGE_TD - 1]]
    ei = int(np.searchsorted(cal, val.max(), side="right"))
    if ei + PURGE_TD >= len(cal):
        return None
    te = te[te >= cal[ei + PURGE_TD]]
    if len(train) < 6 or len(val) < 3 or len(te) == 0:
        return None
    return train, val, te


def ic_eval(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
    """Spearman IC as the early-stopping metric -- the actual objective."""
    y = dtrain.get_label()
    if len(y) < 30:
        return "ic", 0.0
    a = pd.Series(preds).rank().to_numpy()
    b = pd.Series(y).rank().to_numpy()
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return "ic", (float((a * b).sum() / d) if d > 0 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--labels", default="L0,L1,L2,L3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--min-price", type=float, default=MIN_PRICE)
    ap.add_argument("--exclude-features", default="",
                    help="comma-separated features to drop. Used to isolate "
                         "genuine forecasting from bid-ask-bounce exploitation: "
                         "clv_last and r_lag1 are same-bar microstructure and "
                         "die on a one-day implementation lag (README L.8/L.9).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = args.labels.split(",")
    global FEATURES
    if args.exclude_features:
        drop = [x.strip() for x in args.exclude_features.split(",") if x.strip()]
        FEATURES = [f for f in FEATURES if f not in drop]
        print(f"Excluding {drop} -> {len(FEATURES)} features")
    t0 = time.time()

    print("Building features …")
    f = build_features(OHLCV)

    grid = pd.read_parquet(GRID, columns=["ticker", "end_date", "forward_return", "window"])
    grid["end_date"] = pd.to_datetime(grid["end_date"])
    p = grid.merge(f, on=["ticker", "end_date"], how="left")
    p = p[p["Close"] >= args.min_price]

    # controls needed by L3
    o = pd.read_parquet(OHLCV, columns=["Ticker", "Date", "Close", "Volume"])
    o["Date"] = pd.to_datetime(o["Date"]); o = o.sort_values(["Ticker", "Date"])
    o["dv60"] = (o["Close"] * o["Volume"]).groupby(o["Ticker"]).transform(
        lambda s: s.rolling(60, min_periods=15).mean())
    p = p.merge(o[["Ticker", "Date", "dv60"]].rename(
        columns={"Ticker": "ticker", "Date": "end_date"}), on=["ticker", "end_date"], how="left")
    p["log_dv"] = np.log(p["dv60"].where(p["dv60"] > 0))
    if CLASSES.exists():
        cl = pd.read_parquet(CLASSES, columns=["ticker", "bics_level_1"])
        p = p.merge(cl, on="ticker", how="left")
        for s in sorted(p["bics_level_1"].dropna().unique())[1:]:
            p[f"sec_{s.replace(' ', '_').replace('&', 'and')}"] = (p["bics_level_1"] == s).astype(float)

    p = p.dropna(subset=["forward_return"]).reset_index(drop=True)
    print(f"  panel {len(p):,} rows  {p.end_date.nunique()} dates  {p.ticker.nunique()} tickers")

    print("Building labels …")
    p = build_labels(p)
    for L in labels:
        print(f"  {L}: {p[L].notna().sum():,} non-null")

    cal = np.sort(pd.read_parquet(OHLCV, columns=["Date"])["Date"].unique())
    dates = np.sort(p["end_date"].unique())
    windows = sorted(p["window"].dropna().unique())

    params_base = dict(max_depth=5, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, min_child_weight=50,
                       reg_lambda=1.0, nthread=8)

    audit, out_cols = [], {}
    for L in labels:
        for seed in range(args.seeds):
            col = f"{L}_s{seed}"
            preds = np.full(len(p), np.nan)
            for w in windows:
                test_year = 1999 + int(w)
                sp = window_splits(dates, cal, test_year)
                if sp is None:
                    continue
                tr_d, va_d, te_d = sp
                if seed == 0 and L == labels[0]:
                    gap = int(np.searchsorted(cal, va_d.min()) - np.searchsorted(cal, tr_d.max()))
                    audit.append({"window": int(w), "test_year": test_year,
                                  "train_val_gap_td": gap, "n_train_dates": len(tr_d)})
                tr = p[p.end_date.isin(tr_d) & p[L].notna()]
                va = p[p.end_date.isin(va_d) & p[L].notna()]
                te = p[p.end_date.isin(te_d)]
                if len(tr) < 500 or len(va) < 200 or len(te) == 0:
                    continue
                obj = "binary:logistic" if L in ("L0", "L1") else "reg:squarederror"
                prm = {**params_base, "objective": obj, "seed": seed}
                dtr = xgb.DMatrix(tr[FEATURES], label=tr[L])
                dva = xgb.DMatrix(va[FEATURES], label=va[L])
                bst = xgb.train(prm, dtr, num_boost_round=400,
                                evals=[(dva, "val")], custom_metric=ic_eval,
                                maximize=True, early_stopping_rounds=30,
                                verbose_eval=False)
                preds[te.index.to_numpy()] = bst.predict(
                    xgb.DMatrix(te[FEATURES]),
                    iteration_range=(0, bst.best_iteration + 1))
            out_cols[col] = preds
            n = int(np.isfinite(preds).sum())
            print(f"  {col}: {n:,} predictions   [{time.time()-t0:.0f}s]")

    res = p[["ticker", "end_date", "forward_return"]].copy()
    for c, v in out_cols.items():
        res[c] = v
    for L in labels:                      # 5-seed ensemble, as the CNN does
        cols = [f"{L}_s{s}" for s in range(args.seeds) if f"{L}_s{s}" in res]
        res[f"{L}_ens"] = res[cols].mean(axis=1)
    res.to_parquet(args.out_dir / "gbdt_predictions.parquet", index=False)

    ad = pd.DataFrame(audit)
    meta = {"features": FEATURES, "n_features": len(FEATURES), "labels": labels,
            "seeds": args.seeds, "params": params_base,
            "early_stopping_metric": "validation Spearman IC",
            "purge_trading_days": PURGE_TD,
            "purge_audit_min_gap": int(ad.train_val_gap_td.min()) if len(ad) else None,
            "purge_audit_ok": bool(len(ad) and ad.train_val_gap_td.min() >= PURGE_TD + 1),
            "rows": int(len(res)), "runtime_s": round(time.time() - t0, 1)}
    (args.out_dir / "bakeoff_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\npurge audit: min train->val gap {meta['purge_audit_min_gap']} trading days "
          f"(need >= {PURGE_TD+1})  OK={meta['purge_audit_ok']}")
    print(f"wrote {args.out_dir / 'gbdt_predictions.parquet'}  [{meta['runtime_s']}s]")


if __name__ == "__main__":
    main()
