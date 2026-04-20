"""
=============================================================================
SCRIPT NAME: render_etf_cache.py
=============================================================================

INPUT FILES:
- data/etf_ohlcv.csv                 output of fetch_etf_data.py
- runs/expanding/20260419_174908_cdef6809/config.yaml
                                     source of ImageConfig / LabelConfig
                                     (must match training to keep pixel
                                     geometry identical)

OUTPUT FILES:
- cache/etf_I20/images.npy           memmap uint8 (N, 1, 64, 60)
- cache/etf_I20/index.parquet        ticker, end_date, forward_return, label

DESCRIPTION:
Builds the image cache for the 34 ETFs using the exact same image geometry
(I20, 64×60, include_ma=True, include_volume=True) as the expanding-pathway
training run, so the saved CNN weights can be applied cleanly.

USAGE:
  python scripts/render_etf_cache.py
=============================================================================
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pattern.config import Config
from pattern.data.loader import compute_labels, load_data
from pattern.imaging.cache import build_cache

DEFAULT_TRAIN_CONFIG = ROOT / "runs" / "expanding" / "20260419_174908_cdef6809" / "config.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", type=Path, default=ROOT / "data" / "etf_ohlcv.csv",
                    help="OHLCV CSV in r1000 schema")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "etf_I20",
                    help="where to write images.npy + index.parquet")
    ap.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG,
                    help="source of ImageConfig/LabelConfig (must match training)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    cfg = Config.from_yaml(args.train_config)
    img_cfg = cfg.image.model_copy(update={"cache_dir": args.cache_dir})
    lbl_cfg = cfg.label

    print(f"Image config: {img_cfg.height}x{img_cfg.width}  window={img_cfg.window}  "
          f"MA={img_cfg.include_ma}  volume={img_cfg.include_volume}")
    print(f"Label horizon: {lbl_cfg.horizon}")

    df = load_data(args.data_csv, min_history_days=cfg.data.min_history_days)
    print(f"Loaded {len(df):,} rows across {df['Ticker'].nunique()} tickers "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")

    labelled = compute_labels(df, horizon=lbl_cfg.horizon)

    build_cache(
        labelled_df=labelled,
        img_cfg=img_cfg,
        lbl_cfg=lbl_cfg,
        cache_dir=args.cache_dir,
        n_workers=0,
    )

    # Report
    idx = args.cache_dir / "index.parquet"
    if idx.exists():
        import pandas as pd
        index_df = pd.read_parquet(idx)
        print(f"\nCache index: {len(index_df):,} rows  "
              f"{index_df['ticker'].nunique()} tickers  "
              f"{index_df['end_date'].min().date()} → "
              f"{index_df['end_date'].max().date()}")
        per_ticker = (index_df.groupby("ticker")["end_date"]
                      .agg(["count", "min", "max"])
                      .rename(columns={"count": "n", "min": "first", "max": "last"}))
        print(per_ticker.to_string())


if __name__ == "__main__":
    main()
