"""
=============================================================================
SCRIPT NAME: train_extra_seeds.py
=============================================================================
INPUT FILES:
- configs/debug.yaml         : config used for the original run
- runs/<existing>/predictions.parquet : seed-0 predictions to merge with

OUTPUT FILES:
- runs/<existing>/w00_ensemble_<k>.pt           : checkpoints for seeds k=1..4
- runs/<existing>/w00_ensemble_<k>_training_log.csv
- runs/<existing>/predictions_ensemble.parquet  : merged predictions with
                                                   p_up_0..p_up_4, p_up_mean, p_up_std

DESCRIPTION:
Train ensemble members 1..4 on the SAME debug split that produced the existing
seed-0 run, then merge their test-set probabilities into a 5-model ensemble.
Avoids ~40 min of redundant seed-0 retraining.

USAGE:
python -m scripts.train_extra_seeds --run-dir runs/20260418_023201_35c3ccf6
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
from torch.utils.data import DataLoader

from pattern.cli import _seed_everything
from pattern.config import Config
from pattern.data.loader import compute_labels, load_data
from pattern.data.splits import balance_labels, get_splits
from pattern.imaging.cache import compute_pixel_stats, load_cache
from pattern.models.cnn import ChartCNN
from pattern.train.dataset import CachedDataset
from pattern.train.loop import predict, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="Existing run directory containing seed-0 outputs.")
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    cfg = Config.from_yaml(args.config)

    # Reproduce the same split that the original run used.
    df       = load_data(cfg.data.csv_path, cfg.data.date_format, cfg.data.min_history_days)
    labelled = compute_labels(df, cfg.label.horizon)
    purge    = cfg.image.window + cfg.label.horizon - 1
    splits   = get_splits(labelled, cfg.split, purge_days=purge)
    assert len(splits) == 1, "this script targets debug mode (single window)"
    split = splits[0]

    train_df = split["train"]
    val_df   = split["val"]
    test_df  = split["test"]
    if cfg.label.balance_train:
        train_df = balance_labels(train_df, seed=0)   # window_idx = 0

    images, index_df = load_cache(Path(cfg.image.cache_dir))
    index_df["_key"] = index_df["ticker"] + "|" + index_df["end_date"].astype(str)
    index_df["_pos"] = np.arange(len(index_df), dtype=np.int64)

    def to_idx(s: pd.DataFrame) -> np.ndarray:
        keys    = (s["Ticker"] + "|" + s["Date"].astype(str)).values
        matched = index_df[index_df["_key"].isin(pd.Index(keys))]
        return matched["_pos"].to_numpy(dtype=np.int64)

    train_idx = to_idx(train_df)
    val_idx   = to_idx(val_df)
    test_idx  = to_idx(test_df)
    log.info(f"Cache indices — train:{len(train_idx):,}  val:{len(val_idx):,}  test:{len(test_idx):,}")

    pix_mean, pix_std = compute_pixel_stats(Path(cfg.image.cache_dir), train_idx)

    pin = cfg.train.device == "cuda" or (
        cfg.train.device == "auto" and torch.cuda.is_available()
    )

    test_meta = index_df.iloc[test_idx][["ticker", "end_date", "forward_return", "label"]].reset_index(drop=True)

    # Seed-0 probs from the existing run
    seed0 = pd.read_parquet(run_dir / "predictions.parquet")
    if "p_up_0" not in seed0.columns:
        raise RuntimeError(f"existing predictions missing p_up_0: {seed0.columns.tolist()}")
    test_meta = test_meta.merge(
        seed0[["ticker", "end_date", "p_up_0"]],
        on=["ticker", "end_date"], how="left",
    )
    if test_meta["p_up_0"].isna().any():
        raise RuntimeError("seed-0 merge left NaNs — split mismatch?")

    for k in args.seeds:
        seed = cfg.train.seeds[k] if k < len(cfg.train.seeds) else k
        _seed_everything(seed)

        train_ds = CachedDataset(images, index_df, train_idx, pix_mean, pix_std)
        val_ds   = CachedDataset(images, index_df, val_idx,   pix_mean, pix_std)
        test_ds  = CachedDataset(images, index_df, test_idx,  pix_mean, pix_std)

        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                                  shuffle=True,  num_workers=cfg.train.num_workers, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size,
                                  shuffle=False, num_workers=cfg.train.num_workers, pin_memory=pin)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.train.batch_size,
                                  shuffle=False, num_workers=cfg.train.num_workers, pin_memory=pin)

        model_name = f"w00_ensemble_{k}"
        log.info(f"=== Training {model_name} (seed={seed}) ===")
        model = ChartCNN(cfg.model, cfg.image)
        model, _ = train_model(model, train_loader, val_loader,
                               cfg.train, run_dir, model_name, seed)

        probs, _ = predict(model, test_loader, cfg.train.device)
        test_meta[f"p_up_{k}"] = probs[:, 1].numpy()

        # Persist after every seed so we can recover from interruption.
        p_cols = [c for c in test_meta.columns if c.startswith("p_up_")]
        test_meta["p_up_mean"] = test_meta[p_cols].mean(axis=1)
        test_meta["p_up_std"]  = test_meta[p_cols].std(axis=1)
        out_path = run_dir / "predictions_ensemble.parquet"
        tmp_path = out_path.with_suffix(".parquet.tmp")
        test_meta.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)
        log.info(f"  Saved {out_path}  (cols={p_cols})")

    # Final AUC on the ensemble mean
    try:
        from sklearn.metrics import roc_auc_score
        auc_mean = roc_auc_score(test_meta["label"], test_meta["p_up_mean"])
        per_seed = {c: roc_auc_score(test_meta["label"], test_meta[c])
                    for c in test_meta.columns if c.startswith("p_up_") and c != "p_up_mean"}
        log.info(f"Ensemble Test AUC: {auc_mean:.4f}")
        for c, a in per_seed.items():
            log.info(f"  {c}: AUC={a:.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
