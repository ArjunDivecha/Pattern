"""
=============================================================================
SCRIPT NAME: train_full_for_live.py
=============================================================================

INPUT:
- --config configs/prod_expanding.yaml  (for data/image/model/train sections)
- --weight-mode {equal, ewma}
- --half-life-years 5.0                  (only used when weight-mode=ewma)
- --n-epochs 10                          (fixed budget, no validation)
- --n-seeds 5
- --output-dir /data/Pattern/runs/final
- --tag                                  (e.g. "equal_20260420" or "ewma_5yr")

OUTPUT:
- {output_dir}/<tag>_<ts>/ewma_ensemble_{k}.pt            x5
- {output_dir}/<tag>_<ts>/ewma_ensemble_{k}_training_log.csv
- {output_dir}/<tag>_<ts>/predictions.parquet            full-universe scores
- {output_dir}/<tag>_<ts>/config.yaml, ewma_params.json

DESCRIPTION:
Trains one 5-seed ensemble on the ENTIRE image cache (1996→today) with
either equal-weight or exponentially-decayed sample weights.  No validation,
no test: this is the "live deployment" model the user will apply to fresh
data going forward.  Class balance via balance_labels (undersampling).
Weights are applied in the loss, not the sampler.

USAGE:
  # Equal-weighted (baseline "today" model)
  python scripts/train_full_for_live.py --config configs/prod_expanding.yaml \
      --weight-mode equal --tag equal --n-epochs 10

  # EWMA with 5-year half-life
  python scripts/train_full_for_live.py --config configs/prod_expanding.yaml \
      --weight-mode ewma --half-life-years 5 --tag ewma_5yr --n-epochs 10

NOTES:
- Pin CUDA_VISIBLE_DEVICES at the shell level to run on a specific GPU.
- predictions.parquet is generated over every labelled sample in the cache so
  you can decile-sort on any subset afterwards.
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pattern.config import Config
from pattern.data.splits import balance_labels
from pattern.imaging.cache import compute_pixel_stats, load_cache
from pattern.models.cnn import ChartCNN
from pattern.train.dataset import CachedDataset
from pattern.train.loop import predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("live")


class WeightedCachedDataset(Dataset):
    """CachedDataset that also returns a per-sample scalar weight."""

    def __init__(self, images, index_df, indices, weights,
                 pixel_mean=0.0, pixel_std=1.0, preload=True):
        self.index_df = index_df.iloc[indices].reset_index(drop=True)
        self.weights = np.asarray(weights, dtype=np.float32)
        assert len(self.weights) == len(self.index_df)
        self.pixel_mean = pixel_mean
        self.pixel_std = max(pixel_std, 1e-6)
        if preload:
            self.images = np.ascontiguousarray(images[indices])
            self._preloaded = True
        else:
            self.images = images
            self.indices = indices
            self._preloaded = False

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        if self._preloaded:
            img = self.images[idx].astype(np.float32)
        else:
            img = self.images[self.indices[idx]].astype(np.float32)
        x = (img - self.pixel_mean) / self.pixel_std
        label = int(self.index_df.iloc[idx]["label"])
        w = float(self.weights[idx])
        return torch.from_numpy(x), label, w


def train_one_seed(model, loader, cfg_train, n_epochs, seed, device,
                   ckpt_path: Path, log_path: Path, tag: str):
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,wall_s\n")

    t_start = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        loss_sum = 0.0
        wsum = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        for x, y, w in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            logits = model(x)
            per = criterion(logits, y)
            loss = (per * w).sum() / w.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += (per * w).sum().item()
            wsum += w.sum().item()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        mean_loss = loss_sum / wsum
        acc = correct / total
        t_epoch = time.time() - t0
        with open(log_path, "a") as f:
            f.write(f"{epoch},{mean_loss:.5f},{acc:.4f},{t_epoch:.1f}\n")
        log.info(f"  [{tag}] seed={seed} epoch {epoch:2d}/{n_epochs}  "
                 f"loss={mean_loss:.5f}  acc={acc:.4f}  wall={t_epoch:.0f}s")

    torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, ckpt_path)
    log.info(f"  [{tag}] saved {ckpt_path.name}  total={time.time()-t_start:.0f}s")
    return model.to("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--weight-mode", choices=["equal", "ewma"], required=True)
    ap.add_argument("--half-life-years", type=float, default=5.0)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--reference-date", default=None,
                    help="YYYY-MM-DD; defaults to today (used for EWMA only)")
    ap.add_argument("--output-dir", type=Path, default=Path("/data/Pattern/runs/final"))
    ap.add_argument("--tag", required=True, help="subdir prefix, e.g. 'equal' or 'ewma_5yr'")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    cfg_hash = hashlib.md5(json.dumps(yaml.safe_load(args.config.read_text()),
                                      sort_keys=True, default=str).encode()).hexdigest()[:8]
    run_dir = args.output_dir / f"{args.tag}_{stamp}_{cfg_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[{args.tag}] run_dir = {run_dir}")

    # Load cache
    images, index_df = load_cache(Path(cfg.image.cache_dir))
    index_df["end_date"] = pd.to_datetime(index_df["end_date"])
    ok = index_df["label"].isin([0, 1]) & index_df["forward_return"].notna()
    idx_all = np.where(ok)[0]
    log.info(f"[{args.tag}] valid samples: {len(idx_all):,}")

    # Class balance undersampling (track original cache_idx through the reshuffle)
    sub = index_df.iloc[idx_all].copy()
    sub["cache_idx"] = idx_all
    bal = balance_labels(sub, seed=0)
    cache_idx = bal["cache_idx"].to_numpy()
    log.info(f"[{args.tag}] balanced samples: {len(cache_idx):,}")

    # Sample weights
    if args.weight_mode == "equal":
        weights = np.ones(len(cache_idx), dtype=np.float32)
        ref_str = "n/a"
        lam = 0.0
    else:
        ref = (pd.to_datetime(args.reference_date)
               if args.reference_date else pd.Timestamp.today().normalize())
        ref_str = str(ref.date())
        ages_days = (ref - bal["end_date"]).dt.days.to_numpy().astype(np.float64)
        lam = np.log(2) / (args.half_life_years * 365.25)
        w = np.exp(-lam * ages_days)
        # Normalize so mean weight == 1 (preserves loss magnitude).
        w = w * (len(w) / w.sum())
        weights = w.astype(np.float32)
        log.info(f"[{args.tag}] EWMA ref={ref_str}  half_life={args.half_life_years}yr  "
                 f"age_days min/max={ages_days.min():.0f}/{ages_days.max():.0f}  "
                 f"w min/max={weights.min():.4f}/{weights.max():.4f}  "
                 f"top:bot={weights.max()/weights.min():.1f}x")

    # Save config + params
    (run_dir / "config.yaml").write_text(args.config.read_text())
    (run_dir / "params.json").write_text(json.dumps({
        "weight_mode": args.weight_mode,
        "half_life_years": args.half_life_years if args.weight_mode == "ewma" else None,
        "reference_date": ref_str,
        "n_epochs": args.n_epochs,
        "n_seeds": args.n_seeds,
        "n_train_balanced": int(len(cache_idx)),
        "n_valid_total": int(len(idx_all)),
        "lambda": float(lam),
    }, indent=2))

    # Pixel stats from training subset
    pix_mean, pix_std = compute_pixel_stats(Path(cfg.image.cache_dir), cache_idx)
    log.info(f"[{args.tag}] pixel_mean={pix_mean:.3f}  pixel_std={pix_std:.3f}")

    # Device (CUDA_VISIBLE_DEVICES pinning happens at shell level)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[{args.tag}] device={device}")

    # Train 5 seeds
    models = []
    t0 = time.time()
    for k in range(args.n_seeds):
        seed = cfg.train.seeds[k] if k < len(cfg.train.seeds) else k
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_ds = WeightedCachedDataset(
            images, index_df, cache_idx, weights, pix_mean, pix_std, preload=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.train.num_workers, pin_memory=(device.type == "cuda"),
        )
        log.info(f"[{args.tag}] Training seed {k+1}/{args.n_seeds}  (seed={seed})")
        model = ChartCNN(cfg.model, cfg.image)
        ckpt = run_dir / f"ensemble_{k}.pt"
        lg = run_dir / f"ensemble_{k}_training_log.csv"
        model = train_one_seed(model, train_loader, cfg.train, args.n_epochs,
                               seed, device, ckpt, lg, args.tag)
        models.append(model)
    log.info(f"[{args.tag}] training done in {(time.time()-t0)/60:.1f} min")

    # Full-universe inference
    log.info(f"[{args.tag}] scoring full universe ({len(idx_all):,} samples)")
    infer_ds = CachedDataset(images, index_df, idx_all, pix_mean, pix_std)
    infer_loader = DataLoader(
        infer_ds, batch_size=cfg.train.batch_size * 2, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=(device.type == "cuda"),
    )
    all_probs = []
    for k, m in enumerate(models):
        log.info(f"[{args.tag}]   scoring seed {k+1}/{args.n_seeds}")
        probs, _ = predict(m, infer_loader, "auto", return_features=False)
        all_probs.append(probs[:, 1].numpy().astype(np.float32))

    out = index_df.iloc[idx_all][["ticker", "end_date", "forward_return", "label"]].reset_index(drop=True)
    for k, p in enumerate(all_probs):
        out[f"p_up_{k}"] = p
    p_cols = [f"p_up_{k}" for k in range(args.n_seeds)]
    out["p_up_mean"] = out[p_cols].mean(axis=1)
    out["p_up_std"]  = out[p_cols].std(axis=1)
    out["rank_pct"] = out.groupby("end_date")["p_up_mean"].rank(method="average", pct=True)
    out["decile"] = out.groupby("end_date")["p_up_mean"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop")
    )
    out_path = run_dir / "predictions.parquet"
    out.to_parquet(out_path, index=False)
    log.info(f"[{args.tag}] wrote {out_path}  rows={len(out):,}")

    # Quick sanity AUC on full period (training signal leakage expected — this
    # is not a performance estimate, just a sanity check).
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(out["label"], out["p_up_mean"])
        log.info(f"[{args.tag}] full-period in-sample AUC = {auc:.4f}  "
                 f"(not out-of-sample — sanity check only)")
    except Exception as e:
        log.warning(f"[{args.tag}] AUC skipped: {e}")


if __name__ == "__main__":
    main()
