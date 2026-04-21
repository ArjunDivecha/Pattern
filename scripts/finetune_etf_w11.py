"""
=============================================================================
SCRIPT NAME: finetune_etf_w11.py
=============================================================================

INPUT FILES:
- cache/etf_I20/{images.npy,index.parquet}
- runs/expanding/20260419_174908_cdef6809/config.yaml
- runs/expanding/20260419_174908_cdef6809/w11_ensemble_0.pt
- AssetList.xlsx   (34-country ETF universe)

OUTPUT FILES:
- runs/etf_finetune_w11/baseline_predictions.parquet
- runs/etf_finetune_w11/finetuned_predictions.parquet
- runs/etf_finetune_w11/finetuned_w11.pt
- runs/etf_finetune_w11/training_log.csv
- runs/etf_finetune_w11/comparison.txt

DESCRIPTION:
Single-seed fine-tune experiment on the 34-country ETF universe.

  Baseline    : US-trained expanding w11 (train 1996-2009, test 2010),
                seed 0.  Has never seen 2010+ labels.
  Fine-tune on: ETF end_dates 2000-01-01..2010-12-31 (class-balanced, last
                10% of dates held out for validation / early stopping).
  Test on     : ETF end_dates 2011-01-01..2026-03-20 — score with both
                baseline and fine-tuned models and compare AUC + LS
                backtest so we can see if fine-tuning actually helped.

USAGE:
  python scripts/finetune_etf_w11.py
=============================================================================
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pattern.config import Config
from pattern.imaging.cache import load_cache
from pattern.models.cnn import ChartCNN

DEFAULT_SRC_RUN = ROOT / "runs" / "expanding" / "20260419_174908_cdef6809"
DEFAULT_SEED = 0
WINDOW = 11  # expanding w11 = train 1996-2009, test 2010

log = logging.getLogger("finetune_etf_w11")


class CacheDataset(Dataset):
    """Normalised uint8 images keyed by integer indices into the memmap."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray,
                 pix_mean: float, pix_std: float):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.pix_mean = float(pix_mean)
        self.pix_std = float(pix_std)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img = self.images[idx].astype(np.float32)
        img = (img - self.pix_mean) / self.pix_std
        return torch.from_numpy(img), int(self.labels[idx])


def pixel_stats(images: np.ndarray, indices: np.ndarray,
                chunk: int = 5000) -> tuple[float, float]:
    s = 0.0
    s2 = 0.0
    n = 0
    for start in range(0, len(indices), chunk):
        batch = images[indices[start:start + chunk]].astype(np.float64)
        s += batch.sum()
        s2 += (batch ** 2).sum()
        n += batch.size
    mean = s / n
    std = float(np.sqrt(max(s2 / n - mean ** 2, 0.0))) or 1.0
    return float(mean), std


def class_balance(indices: np.ndarray, labels: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """50/50 undersample for training."""
    y = labels[indices]
    pos = indices[y == 1]
    neg = indices[y == 0]
    k = min(len(pos), len(neg))
    pos = rng.choice(pos, size=k, replace=False)
    neg = rng.choice(neg, size=k, replace=False)
    out = np.concatenate([pos, neg])
    rng.shuffle(out)
    return out


def score(model: nn.Module, images: np.ndarray, indices: np.ndarray,
          pix_mean: float, pix_std: float, device: torch.device,
          chunk: int = 1024) -> np.ndarray:
    """Return p_up for the given indices."""
    model = model.to(device).eval()
    probs = np.empty(len(indices), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), chunk):
            idx = indices[start:start + chunk]
            batch = images[idx].astype(np.float32)
            batch = (batch - pix_mean) / pix_std
            x = torch.from_numpy(batch).to(device)
            p = F.softmax(model(x), dim=-1)[:, 1].cpu().numpy()
            probs[start:start + len(idx)] = p
    return probs


def ls_backtest(preds: pd.DataFrame, top_pct: float = 20.0, bot_pct: float = 20.0,
                hold_days: int = 20) -> dict:
    """Very lightweight decile backtest for reporting alongside AUC."""
    rows = []
    for date, g in preds.groupby("end_date", sort=True):
        if len(g) < 3:
            continue
        r = g["forward_return"].to_numpy()
        p = g["p_up_mean"].to_numpy()
        n = len(g)
        rank = pd.Series(p).rank(method="first") / n
        top = (rank > 1 - top_pct / 100).to_numpy()
        bot = (rank <= bot_pct / 100).to_numpy()
        rows.append({
            "end_date": date,
            "TOP": r[top].mean() if top.any() else np.nan,
            "BOT": r[bot].mean() if bot.any() else np.nan,
            "EW":  r.mean(),
        })
    df = pd.DataFrame(rows).dropna().set_index("end_date").sort_index()
    df["LS"] = df["TOP"] - df["BOT"]
    # non-overlap monthly sampling
    mask = np.zeros(len(df), dtype=bool)
    mask[::hold_days] = True
    monthly = df[mask]
    m = monthly["LS"].mean()
    s = monthly["LS"].std(ddof=1)
    cagr = float(np.exp(m * 252 / hold_days) - 1)
    sharpe = m / s * np.sqrt(252 / hold_days) if s else np.nan
    ann_top = float(np.exp(monthly["TOP"].mean() * 252 / hold_days) - 1)
    ann_bot = float(np.exp(monthly["BOT"].mean() * 252 / hold_days) - 1)
    ann_ew  = float(np.exp(monthly["EW" ].mean() * 252 / hold_days) - 1)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(preds["label"], preds["p_up_mean"])
    except Exception:
        auc = float("nan")
    return {"auc": auc, "ls_cagr": cagr * 100, "ls_sharpe": sharpe,
            "top_cagr": ann_top * 100, "bot_cagr": ann_bot * 100,
            "ew_cagr": ann_ew * 100, "n_months": int(mask.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-run", type=Path, default=DEFAULT_SRC_RUN)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "etf_I20")
    ap.add_argument("--tickers-xlsx", type=Path, default=ROOT / "AssetList.xlsx")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to runs/etf_finetune_w11 or runs/etf_scratch_w11")
    ap.add_argument("--from-scratch", action="store_true",
                    help="train from random init (not baseline) — still uses baseline for comparison")
    ap.add_argument("--lr", type=float, default=None,
                    help="default 1e-6 for fine-tune, 1e-5 for from-scratch")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="default 20 for fine-tune, 60 for from-scratch")
    ap.add_argument("--patience", type=int, default=None,
                    help="default 2 for fine-tune, 4 for from-scratch")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    # Defaults depend on mode
    if args.out_dir is None:
        args.out_dir = ROOT / ("runs/etf_scratch_w11" if args.from_scratch
                               else "runs/etf_finetune_w11")
    if args.lr is None:
        args.lr = 1e-5 if args.from_scratch else 1e-6
    if args.max_epochs is None:
        args.max_epochs = 60 if args.from_scratch else 20
    if args.patience is None:
        args.patience = 4 if args.from_scratch else 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Mode: {'FROM-SCRATCH' if args.from_scratch else 'FINE-TUNE'}  "
             f"lr={args.lr}  max_epochs={args.max_epochs}  patience={args.patience}")

    # ── device ─────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    cfg = Config.from_yaml(args.src_run / "config.yaml")

    # ── load cache + universe filter ───────────────────────────────────────
    images, index_df = load_cache(args.cache_dir)
    log.info(f"Cache: {len(index_df):,} rows  "
             f"{index_df['end_date'].min()} → {index_df['end_date'].max()}")

    tickers = (pd.read_excel(args.tickers_xlsx)["Ticker"]
               .astype(str).str.strip().tolist())
    idx_mask = index_df["ticker"].isin(tickers)
    log.info(f"Universe filter: {idx_mask.sum():,} / {len(index_df):,} rows "
             f"across {len(tickers)} tickers in AssetList.xlsx")

    end_date = pd.to_datetime(index_df["end_date"])
    labels = index_df["label"].to_numpy().astype(np.int64)

    train_mask = idx_mask & (end_date >= "2000-01-01") & (end_date <= "2010-12-31")
    test_mask  = idx_mask & (end_date >= "2011-01-01") & (end_date <= "2026-12-31")

    train_all_idx = np.flatnonzero(train_mask.to_numpy()).astype(np.int64)
    test_idx      = np.flatnonzero(test_mask.to_numpy()).astype(np.int64)

    log.info(f"Train rows: {len(train_all_idx):,}   Test rows: {len(test_idx):,}")

    # Use last 10% of *training dates* as val (chronological, not random)
    train_dates = end_date.iloc[train_all_idx].to_numpy()
    order = np.argsort(train_dates)
    cutoff = int(len(order) * 0.90)
    train_idx = train_all_idx[order[:cutoff]]
    val_idx   = train_all_idx[order[cutoff:]]
    log.info(f"  train(90%)={len(train_idx):,}  val(10%)={len(val_idx):,}")

    # Pixel stats on training indices only
    pix_mean, pix_std = pixel_stats(images, train_idx)
    log.info(f"Pixel stats: mean={pix_mean:.2f}  std={pix_std:.2f}")

    # ── load w11 seed 0 as baseline ────────────────────────────────────────
    ckpt_path = args.src_run / f"w{WINDOW:02d}_ensemble_{args.seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    log.info(f"Baseline checkpoint: {ckpt_path.name}")

    baseline = ChartCNN(cfg.model, cfg.image)
    baseline.load_state_dict(state)

    # Score test set with baseline BEFORE touching it
    log.info("Scoring test set with BASELINE model …")
    baseline_probs = score(baseline, images, test_idx, pix_mean, pix_std, device)

    baseline_df = index_df.iloc[test_idx][["ticker", "end_date",
                                           "forward_return", "label"]].copy()
    baseline_df["p_up_mean"] = baseline_probs
    baseline_df = baseline_df.sort_values(["ticker", "end_date"]).reset_index(drop=True)
    baseline_df.to_parquet(args.out_dir / "baseline_predictions.parquet", index=False)

    # ── train ──────────────────────────────────────────────────────────────
    if args.from_scratch:
        log.info("Training FROM SCRATCH (random init) on 2000-2010 ETF data …")
    else:
        log.info("Fine-tuning on 2000-2010 ETF data …")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model = ChartCNN(cfg.model, cfg.image)
    if not args.from_scratch:
        model.load_state_dict(state)
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    val_ds = CacheDataset(images, labels, val_idx, pix_mean, pix_std)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    log_path = args.out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(1, args.max_epochs + 1):
        # Resample class-balanced training indices each epoch
        bal_idx = class_balance(train_idx, labels, rng)
        train_ds = CacheDataset(images, labels, bal_idx, pix_mean, pix_std)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)

        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(y)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_n += len(y)
        tr_loss /= tr_n
        tr_acc = tr_correct / tr_n

        # val
        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_n = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += loss.item() * len(y)
                va_correct += (logits.argmax(1) == y).sum().item()
                va_n += len(y)
        va_loss /= va_n
        va_acc = va_correct / va_n

        log.info(f"  Epoch {epoch:2d}/{args.max_epochs}  "
                 f"train_loss={tr_loss:.5f} acc={tr_acc:.3f}  |  "
                 f"val_loss={va_loss:.5f} acc={va_acc:.3f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(tr_loss, 6), round(tr_acc, 4),
                             round(va_loss, 6), round(va_acc, 4)])

        if va_loss < best_val:
            best_val = va_loss
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            ckpt_name = "scratch_w11.pt" if args.from_scratch else "finetuned_w11.pt"
            torch.save(best_state, args.out_dir / ckpt_name)
            log.info(f"    ✓ new best val_loss={best_val:.5f}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                log.info(f"  Early stop at epoch {epoch} (patience={args.patience})")
                break

    # Load best and score test
    model.load_state_dict(best_state)
    label = "FROM-SCRATCH" if args.from_scratch else "FINE-TUNED"
    log.info(f"Scoring test set with {label} model …")
    ft_probs = score(model, images, test_idx, pix_mean, pix_std, device)

    ft_df = index_df.iloc[test_idx][["ticker", "end_date",
                                     "forward_return", "label"]].copy()
    ft_df["p_up_mean"] = ft_probs
    ft_df = ft_df.sort_values(["ticker", "end_date"]).reset_index(drop=True)
    pred_name = "scratch_predictions.parquet" if args.from_scratch else "finetuned_predictions.parquet"
    ft_df.to_parquet(args.out_dir / pred_name, index=False)

    # ── compare ────────────────────────────────────────────────────────────
    log.info("")
    log.info("Comparison (test window 2011-01-01 → 2026-03-20):")
    b = ls_backtest(baseline_df)
    t = ls_backtest(ft_df)

    col = "From-scratch" if args.from_scratch else "Fine-tuned"
    lines = [
        f"Metric                  Baseline (w11)     {col}",
        "-" * 65,
        f"AUC                     {b['auc']:.4f}              {t['auc']:.4f}",
        f"TOP 20% CAGR (%)        {b['top_cagr']:+.2f}              {t['top_cagr']:+.2f}",
        f"BOT 20% CAGR (%)        {b['bot_cagr']:+.2f}              {t['bot_cagr']:+.2f}",
        f"LS  CAGR     (%)        {b['ls_cagr']:+.2f}              {t['ls_cagr']:+.2f}",
        f"LS  Sharpe              {b['ls_sharpe']:+.3f}             {t['ls_sharpe']:+.3f}",
        f"EW  CAGR     (%)        {b['ew_cagr']:+.2f}              {t['ew_cagr']:+.2f}",
        f"(non-overlap months)    {b['n_months']}                 {t['n_months']}",
    ]
    report = "\n".join(lines)
    log.info("\n" + report)
    (args.out_dir / "comparison.txt").write_text(report + "\n")

    log.info(f"Artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
