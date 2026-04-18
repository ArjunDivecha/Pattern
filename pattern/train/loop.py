"""
=============================================================================
SCRIPT NAME: loop.py
=============================================================================
INPUT:
- train/val DataLoaders
- ChartCNN model
OUTPUT:
- Best model checkpoint (.pt)
- training_log.csv  (per-epoch train/val loss + accuracy)

DESCRIPTION:
Training loop with early stopping.  Supports single-model training (Phase 1)
and is called once per ensemble member (Phase 2).

Device priority: CUDA → MPS (Apple Silicon) → CPU.
=============================================================================
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pattern.config import TrainConfig
from pattern.models.cnn import ChartCNN

log = logging.getLogger(__name__)


def _init_wandb(cfg: TrainConfig, run_dir: Path, model_name: str, seed: int):
    """Initialise a W&B run if wandb is installed and cfg.use_wandb is True."""
    try:
        import wandb
        run = wandb.init(
            project = cfg.wandb_project,
            name    = f"{run_dir.name}/{model_name}",
            group   = run_dir.name,
            config  = {
                "model_name": model_name,
                "seed":       seed,
                "lr":         cfg.lr,
                "batch_size": cfg.batch_size,
                "max_epochs": cfg.max_epochs,
                "patience":   cfg.early_stop_patience,
                "device":     cfg.device,
            },
            dir     = str(run_dir),
            reinit  = True,
            settings = wandb.Settings(init_timeout=300),
        )
        return run
    except Exception as e:
        log.warning(f"W&B init failed ({e}) — continuing without it.")
        return None


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[float, float]:
    """Run one epoch (train if optimizer given, else eval). Returns (loss, acc)."""
    training = optimizer is not None
    model.train(training)

    # non_blocking H2D copies are only meaningful for pinned CUDA memory.
    # On MPS/CPU they are silently a no-op at best and a source of sporadic
    # garbage reads at worst (the latter is what first introduced NaN logits
    # into our loss accumulator on MPS).
    non_blocking = device.type == "cuda"

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(training):
        for step, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)

            logits = model(x)
            loss   = criterion(logits, y)

            # FAIL LOUD on non-finite logits or loss before we ever accumulate
            # or backprop — silent NaN poisons both metrics and parameters.
            if not torch.isfinite(logits).all():
                lmin = float(logits.min().item())
                lmax = float(logits.max().item())
                raise RuntimeError(
                    f"Non-finite logits at step {step}/{len(loader)}  "
                    f"batch_size={len(y)}  x.shape={tuple(x.shape)}  "
                    f"y_unique={y.unique().tolist()}  "
                    f"logit_has_nan={bool(torch.isnan(logits).any().item())}  "
                    f"logit_has_inf={bool(torch.isinf(logits).any().item())}  "
                    f"logit_range=({lmin:.3e},{lmax:.3e})"
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss={loss.item()} at step {step}/{len(loader)}  "
                    f"batch_size={len(y)}  x.shape={tuple(x.shape)}"
                )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


def train_model(
    model: ChartCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    run_dir: Path,
    model_name: str = "model",
    seed: int = 0,
) -> Tuple[ChartCNN, dict]:
    """
    Train a single model with early stopping.

    Args:
        model:        Initialised ChartCNN.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        cfg:          Training config.
        run_dir:      Directory to save checkpoint and log.
        model_name:   File stem for checkpoint (e.g. "ensemble_0").
        seed:         For reproducibility logging.

    Returns:
        (best_model, history_dict)
    """
    torch.manual_seed(seed)
    device = _get_device(cfg.device)
    model  = model.to(device)

    log.info(f"Training {model_name}  |  device={device}  |  seed={seed}")

    if cfg.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None
    history       = []

    log_path  = run_dir / f"{model_name}_training_log.csv"
    ckpt_path = run_dir / f"{model_name}.pt"

    wb_run = _init_wandb(cfg, run_dir, model_name, seed) if cfg.use_wandb else None

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _run_epoch(model, val_loader,   criterion, None,      device)

        row = {
            "epoch":      epoch,
            "train_loss": round(tr_loss, 6),
            "train_acc":  round(tr_acc,  4),
            "val_loss":   round(va_loss, 6),
            "val_acc":    round(va_acc,  4),
        }
        history.append(row)

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row.values())

        if wb_run is not None:
            wb_run.log({
                "train/loss": tr_loss,
                "train/acc":  tr_acc,
                "val/loss":   va_loss,
                "val/acc":    va_acc,
                "best_val_loss": best_val_loss,
            }, step=epoch)

        log.info(
            f"  Epoch {epoch:3d}/{cfg.max_epochs}  "
            f"train_loss={tr_loss:.5f}  acc={tr_acc:.3f}  |  "
            f"val_loss={va_loss:.5f}  acc={va_acc:.3f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            log.info(f"    ✓ New best val_loss={best_val_loss:.5f} — saved {ckpt_path.name}")
            if wb_run is not None:
                wb_run.summary["best_val_loss"] = best_val_loss
                wb_run.summary["best_epoch"]    = epoch
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.early_stop_patience:
                log.info(f"  Early stop at epoch {epoch} (patience={cfg.early_stop_patience})")
                break

    if wb_run is not None:
        wb_run.finish()

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to("cpu")

    return model, history


def predict(
    model: ChartCNN,
    loader: DataLoader,
    device_str: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference on all batches.

    Returns:
        probs:  (N, 2) float tensor [P(down), P(up)] for each sample.
        labels: (N,) int tensor.
    """
    device = _get_device(device_str)
    model  = model.to(device).eval()
    non_blocking = device.type == "cuda"

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=non_blocking)
            p = model.predict_proba(x).cpu()
            all_probs.append(p)
            all_labels.append(y)

    return torch.cat(all_probs), torch.cat(all_labels)
