"""
=============================================================================
SCRIPT NAME: score_live.py
=============================================================================

INPUT FILES:
- --data-csv            OHLCV CSV in r1000 schema (e.g. data/etf_ohlcv.csv)
- --src-run             expanding run dir with w{NN}_ensemble_{0..4}.pt
                        (default: runs/expanding/20260419_174908_cdef6809)
- --tickers-xlsx        (optional) xlsx with 'Ticker' column to restrict output
                        (default: AssetList.xlsx)
- --cache-dir           historical cache with pixel-stat reference
                        (default: cache/etf_I20)

OUTPUT FILES:
- --out-parquet         ticker, end_date, p_up_0..4, p_up_mean, p_up_std

DESCRIPTION:
Produces p_up forecasts for the tail of trading days that don't yet have a
20-day forward return (and so are excluded from the labeled cache).  Uses
the most recent expanding-window checkpoint (w27) because its training
span matches "everything up to 2025".  Pixel stats use the pre-existing
cache rows as a reference so normalisation matches the labeled pathway.

USAGE:
  python scripts/score_live.py \
      --data-csv data/etf_ohlcv.csv \
      --out-parquet runs/etf_expanding/live_predictions.parquet
=============================================================================
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pattern.config import Config
from pattern.data.loader import build_ticker_index, get_window, load_data
from pattern.imaging.cache import load_cache
from pattern.imaging.renderer import render_window
from pattern.models.cnn import ChartCNN

DEFAULT_SRC_RUN = ROOT / "runs" / "expanding" / "20260419_174908_cdef6809"
LIVE_WINDOW = 27  # last expanding window — trained on 1996-2025, tests 2026

log = logging.getLogger("score_live")


def pick_device(pref: str) -> torch.device:
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(pref)


def ref_pixel_stats(cache_dir: Path) -> tuple[float, float]:
    """Mean/std over the historical cache — used for normalisation of live
    images.  We use the whole cache since the live model is window 27 trained
    on 1996–2025 data."""
    images, _ = load_cache(cache_dir)
    n = len(images)
    chunk = 5000
    sum_ = 0.0
    sum_sq = 0.0
    n_px = 0
    for start in range(0, n, chunk):
        batch = images[start:start + chunk].astype(np.float64)
        sum_ += batch.sum()
        sum_sq += (batch ** 2).sum()
        n_px += batch.size
    mean = sum_ / n_px
    std = float(np.sqrt(max(sum_sq / n_px - mean ** 2, 0.0))) or 1.0
    return float(mean), std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", type=Path, default=ROOT / "data" / "etf_ohlcv.csv")
    ap.add_argument("--src-run", type=Path, default=DEFAULT_SRC_RUN)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "etf_I20")
    ap.add_argument("--tickers-xlsx", type=Path, default=ROOT / "AssetList.xlsx")
    ap.add_argument("--out-parquet", type=Path,
                    default=ROOT / "runs" / "etf_expanding" / "live_predictions.parquet")
    ap.add_argument("--tail-days", type=int, default=30,
                    help="number of trailing trading days per ticker to consider")
    ap.add_argument("--cadence", choices=["monthly", "daily"], default="monthly",
                    help="monthly = score only the last trading day of each "
                         "calendar month in the tail window (default).  "
                         "daily = score every day in the tail window.")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    device = pick_device(args.device)
    log.info(f"Device: {device}")

    cfg = Config.from_yaml(args.src_run / "config.yaml")
    img_cfg = cfg.image
    window = img_cfg.window
    lookback = window - 1  # for MA

    # ── load models ────────────────────────────────────────────────────────
    models = []
    for k in range(5):
        ckpt = args.src_run / f"w{LIVE_WINDOW:02d}_ensemble_{k}.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m = ChartCNN(cfg.model, img_cfg)
        m.load_state_dict(state)
        m.to(device).eval()
        models.append(m)
    log.info(f"Loaded w{LIVE_WINDOW:02d} ensemble (5 seeds)")

    pix_mean, pix_std = ref_pixel_stats(args.cache_dir)
    log.info(f"Reference pixel stats: mean={pix_mean:.2f}  std={pix_std:.2f}")

    # ── load OHLCV + render recent unlabeled tail ──────────────────────────
    df = load_data(args.data_csv, min_history_days=cfg.data.min_history_days)
    tickers_xlsx = pd.read_excel(args.tickers_xlsx)["Ticker"].astype(str).str.strip().tolist()
    df = df[df["Ticker"].isin(tickers_xlsx)].copy()
    log.info(f"Loaded {len(df):,} rows across {df['Ticker'].nunique()} tickers  "
             f"{df['Date'].min().date()} → {df['Date'].max().date()}")

    tix = build_ticker_index(df)

    records = []
    imgs = []
    for ticker, tdf in tix.items():
        dates = tdf["Date"].to_numpy()
        if len(dates) < window + lookback:
            continue
        # Score the last `tail_days` rows for this ticker — these include
        # the rows whose 20-day forward return isn't yet available.
        tail = tdf.iloc[-args.tail_days:]
        if args.cadence == "monthly":
            # Keep only the CANONICAL last trading day of each calendar month
            # in the tail — the latest date for that month in the full
            # universe (`df`), not just this ticker.  Live scoring runs at
            # ≤ 1 row/ticker/month and all tickers align on the same dates.
            ym = tail["Date"].dt.to_period("M")
            canonical_by_month = df.groupby(df["Date"].dt.to_period("M"))["Date"].max()
            mask = tail["Date"].values == ym.map(canonical_by_month).values
            tail = tail.loc[mask]
        for _, row in tail.iterrows():
            end_date = row["Date"]
            res = get_window(tdf, end_date, window, lookback)
            if res is None:
                continue
            ohlcv, tri = res
            img = render_window(
                ohlcv, tri, window, img_cfg.height, img_cfg.width,
                img_cfg.ohlc_height_ratio,
                img_cfg.include_ma, img_cfg.include_volume,
            )
            if img is None:
                continue
            records.append({"ticker": ticker, "end_date": end_date})
            imgs.append(img)

    if not imgs:
        raise RuntimeError("No live images generated.")

    X = np.stack(imgs, axis=0).astype(np.float32)   # (N, 1, H, W)
    X = (X - pix_mean) / pix_std
    log.info(f"Rendered {len(X):,} live images")

    probs = np.empty((len(X), 5), dtype=np.float32)
    with torch.inference_mode():
        x = torch.from_numpy(X).to(device)
        for k, m in enumerate(models):
            p = F.softmax(m(x), dim=-1)[:, 1].cpu().numpy()
            probs[:, k] = p

    out = pd.DataFrame(records)
    for k in range(5):
        out[f"p_up_{k}"] = probs[:, k]
    out["p_up_mean"] = probs.mean(axis=1)
    out["p_up_std"] = probs.std(axis=1)
    out = out.sort_values(["ticker", "end_date"]).reset_index(drop=True)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)
    log.info(f"wrote {args.out_parquet}  rows={len(out):,}  "
             f"tickers={out['ticker'].nunique()}  "
             f"{out['end_date'].min().date()} → {out['end_date'].max().date()}")


if __name__ == "__main__":
    main()
