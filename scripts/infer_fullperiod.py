"""
=============================================================================
SCRIPT NAME: infer_fullperiod.py
=============================================================================
INPUT FILES:
- runs/<src_run>/config.yaml + w00_ensemble_{0..K-1}.pt
- cache/debug_I20/images.npy + index.parquet

OUTPUT FILES:
- runs/<dst_run>/predictions_ensemble.parquet (+ predictions.parquet mirror)
- runs/<dst_run>/config.yaml
- runs/<dst_run>/report.md + backtest_*.{pdf,xlsx,parquet}

DESCRIPTION:
Ensemble inference across the full OOS period using the 5 checkpoints
at runs/20260418_023201_35c3ccf6. Those were trained with the OLD CNN
(every block using asymmetric outer conv params) so inner params are
overridden to match before loading.

Path:
  1. Load all 5 state_dicts once.
  2. Stack each layer's weights across models into a GROUPED convolution
     so a single forward pass computes all 5 ensemble logits.
  3. Run on MPS with modest batch size (good neighbour to concurrent
     training). Falls back to CPU if --device cpu.
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from pattern.config import Config
from pattern.imaging.cache import compute_pixel_stats, load_cache
from pattern.models.cnn import ChartCNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _load_cfg_with_old_arch(src_cfg_path: Path) -> Config:
    with open(src_cfg_path) as f:
        raw = yaml.safe_load(f)
    m = raw.setdefault("model", {})
    m["conv_stride_inner"]   = list(m.get("conv_stride",   [3, 1]))
    m["conv_padding_inner"]  = list(m.get("conv_padding",  [12, 1]))
    m["conv_dilation_inner"] = list(m.get("conv_dilation", [2, 1]))
    return Config.model_validate(raw)


def _make_run_dir(output_dir: Path, cfg: Config, tag: str) -> Path:
    cfg_str  = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, default=str)
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = output_dir / f"{stamp}_{cfg_hash}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ── Ensemble-stacked (grouped-conv) model ────────────────────────────────────

def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Fold BatchNorm into preceding Conv weights (eval mode).

    Returns (weight, bias) tensors for a pure Conv2d that produces the same
    output as Conv → BN in eval mode.
    """
    w = conv.weight.clone()
    b = conv.bias.clone() if conv.bias is not None else torch.zeros(w.shape[0])
    rv = bn.running_var
    rm = bn.running_mean
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias
    scale = gamma / torch.sqrt(rv + eps)
    w = w * scale.reshape(-1, 1, 1, 1)
    b = (b - rm) * scale + beta
    return w, b


class StackedEnsemble(nn.Module):
    """All K ensemble members evaluated in ONE forward pass via grouped convs.

    For each conv layer, stack the K model weights along dim 0 to get
    (K*out_ch, in_ch, kH, kW) and use groups=K. The input is repeated
    K times on the channel dim (B, K*1, H, W). Between blocks, MaxPool
    still applies per-channel so grouping is preserved.

    Final FC: each model has its own FC weight (2, fc_in). We build a
    block-diagonal weight (K*2, K*fc_in) so that one matmul produces
    all K pairs of logits.
    """

    def __init__(self, models: list, img_cfg):
        super().__init__()
        assert all(len(m.conv_blocks) == len(models[0].conv_blocks) for m in models)
        K = len(models)
        self.K = K

        # Per-block: (K*out, in, kH, kW) weight with groups=K.
        # Treat the 5 independent models as 5 groups; input must be duplicated
        # K times along the channel dim and each model sees its own slice.
        self.block_params = nn.ParameterList()
        self.block_biases = nn.ParameterList()
        self.block_meta   = []  # list of dicts with stride/padding/dilation/pool

        n_blocks = len(models[0].conv_blocks)
        for bi in range(n_blocks):
            weights, biases = [], []
            meta = None
            for m in models:
                blk = m.conv_blocks[bi]
                conv = blk.block[0]
                bn   = blk.block[1]
                w, b = _fuse_conv_bn(conv, bn)
                weights.append(w)
                biases.append(b)
                if meta is None:
                    pool = blk.block[3]
                    meta = {
                        "stride":   conv.stride,
                        "padding":  conv.padding,
                        "dilation": conv.dilation,
                        "leaky":    blk.block[2].negative_slope,
                        "pool_k":   pool.kernel_size if isinstance(pool.kernel_size, tuple)
                                    else (pool.kernel_size, pool.kernel_size),
                        "pool_s":   pool.stride if isinstance(pool.stride, tuple)
                                    else (pool.stride, pool.stride),
                    }
            W = torch.cat(weights, dim=0)  # (K*out, in, kH, kW)
            B = torch.cat(biases,  dim=0)  # (K*out,)
            self.block_params.append(nn.Parameter(W, requires_grad=False))
            self.block_biases.append(nn.Parameter(B, requires_grad=False))
            self.block_meta.append(meta)

        # FC: stack K copies into a block-diagonal weight.
        fc_weights = torch.stack([m.fc.weight for m in models], dim=0)  # (K, 2, fc_in)
        fc_biases  = torch.stack([m.fc.bias   for m in models], dim=0)  # (K, 2)
        self.fc_weight = nn.Parameter(fc_weights, requires_grad=False)  # (K, 2, fc_in)
        self.fc_bias   = nn.Parameter(fc_biases,  requires_grad=False)  # (K, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) → (B, K, 2) logits."""
        B = x.size(0)
        K = self.K
        # Replicate input across K groups: (B, K, H, W) with each group=1 channel per model.
        x = x.expand(B, K, x.size(2), x.size(3)).contiguous()

        for bi, (W, b, meta) in enumerate(zip(self.block_params, self.block_biases, self.block_meta)):
            x = F.conv2d(x, W, b,
                         stride=meta["stride"],
                         padding=meta["padding"],
                         dilation=meta["dilation"],
                         groups=K)
            x = F.leaky_relu(x, negative_slope=meta["leaky"])
            x = F.max_pool2d(x, kernel_size=meta["pool_k"], stride=meta["pool_s"])

        # x is (B, K*C, Hf, Wf). Reshape to (B, K, C*Hf*Wf)
        x = x.view(B, K, -1)
        # Per-group linear: (B, K, fc_in) × (K, 2, fc_in)^T → (B, K, 2)
        #   einsum 'bki,koi->bko'
        logits = torch.einsum("bki,koi->bko", x, self.fc_weight) + self.fc_bias
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-run", default="runs/20260418_023201_35c3ccf6")
    parser.add_argument("--exclude-years", type=int, nargs="*", default=[1996, 1997, 1998])
    parser.add_argument("--min-year", type=int, default=1999)
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="Rows per forward pass. Smaller = gentler on concurrent training.")
    parser.add_argument("--device", default="mps",
                        choices=["mps", "cpu"],
                        help="mps recommended; CPU is ~10× slower here due to lack of MKL-DNN on arm64.")
    parser.add_argument("--tag", default="oos")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    src_run = Path(args.src_run)
    cfg = _load_cfg_with_old_arch(src_run / "config.yaml")

    device = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── 1. Load cache ────────────────────────────────────────────────────────
    cache_dir = Path(cfg.image.cache_dir)
    images, index_df = load_cache(cache_dir)

    # ── 2. OOS mask ─────────────────────────────────────────────────────────
    dates = pd.to_datetime(index_df["end_date"])
    years = dates.dt.year
    mask = (years >= args.min_year) & (~years.isin(args.exclude_years))
    oos_pos = np.flatnonzero(mask.values).astype(np.int64)
    if args.limit > 0:
        oos_pos = oos_pos[:args.limit]
    log.info(f"OOS cache rows: {len(oos_pos):,}  "
             f"({dates[mask].min().date()} → {dates[mask].max().date()})")

    # ── 3. Pixel stats ──────────────────────────────────────────────────────
    orig_train_mask = (years >= 1996) & (years <= 1998)
    orig_train_pos = np.flatnonzero(orig_train_mask.values).astype(np.int64)
    pix_mean, pix_std = compute_pixel_stats(cache_dir, orig_train_pos)
    log.info(f"Pixel stats: mean={pix_mean:.4f} std={pix_std:.4f}")

    # ── 4. Load all K ensemble members and stack into one grouped model ─────
    # The saved config may say ensemble_size=1 even when multiple seed
    # checkpoints exist on disk — discover them by file.
    ckpt_paths = sorted(src_run.glob("w00_ensemble_*.pt"),
                        key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not ckpt_paths:
        raise FileNotFoundError(f"No w00_ensemble_*.pt checkpoints in {src_run}")
    K = len(ckpt_paths)
    models = []
    for ck in ckpt_paths:
        m = ChartCNN(cfg.model, cfg.image)
        state = torch.load(ck, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    log.info(f"Loaded {K} ensemble members: {[p.name for p in ckpt_paths]}")

    stacked = StackedEnsemble(models, cfg.image).to(device).eval()

    # Validate correctness on a small batch: stacked vs. one-model-at-a-time.
    with torch.inference_mode():
        xv = torch.randn(8, 1, cfg.image.height, cfg.image.width, device=device)
        sv = torch.softmax(stacked(xv), dim=-1)[..., 1]  # (B, K)
        refs = []
        for m in models:
            m_dev = m.to(device)
            refs.append(torch.softmax(m_dev(xv), dim=-1)[:, 1])
        ref = torch.stack(refs, dim=1)  # (B, K)
        mad = (sv - ref).abs().max().item()
        log.info(f"Stacked vs. per-model max abs diff: {mad:.2e}")
        assert mad < 1e-3, f"Stacked model mismatch ({mad:.2e})"

    # ── 5. Run inference chunk-by-chunk ─────────────────────────────────────
    N = len(oos_pos)
    chunk = args.chunk_size
    probs_matrix = np.empty((N, K), dtype=np.float32)
    t_start = datetime.now()

    with torch.inference_mode():
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            idx = oos_pos[start:end]

            batch_u8 = np.ascontiguousarray(images[idx])
            batch_f32 = (batch_u8.astype(np.float32) - pix_mean) / pix_std
            x = torch.from_numpy(batch_f32).to(device, non_blocking=False)

            logits = stacked(x)                                 # (bs, K, 2)
            p_up = torch.sigmoid(logits[..., 1] - logits[..., 0]).cpu().numpy()
            probs_matrix[start:end] = p_up

            if (start // chunk) % 50 == 0:
                elapsed = (datetime.now() - t_start).total_seconds()
                rate = end / elapsed if elapsed > 0 else 0.0
                eta = (N - end) / rate / 60 if rate > 0 else float("inf")
                log.info(f"  {end:,}/{N:,}  {rate:,.0f} samp/s  ETA={eta:.1f} min")

    log.info(f"Inference done in {(datetime.now()-t_start).total_seconds()/60:.1f} min")

    # ── 6. Assemble predictions parquet ──────────────────────────────────────
    meta = index_df.iloc[oos_pos][["ticker", "end_date", "forward_return", "label"]].reset_index(drop=True)
    for k in range(K):
        meta[f"p_up_{k}"] = probs_matrix[:, k]
    meta["p_up_mean"] = probs_matrix.mean(axis=1)
    meta["p_up_std"]  = probs_matrix.std(axis=1)
    meta["window"]    = 0

    # ── 7. Save to new run dir ───────────────────────────────────────────────
    run_dir = _make_run_dir(Path(cfg.output_dir), cfg, args.tag)
    log.info(f"Saving to: {run_dir}")
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg.model_dump(mode="json"), f, default_flow_style=False)
    with open(run_dir / "source_run.txt", "w") as f:
        f.write(str(src_run.resolve()) + "\n")
    meta.to_parquet(run_dir / "predictions.parquet", index=False)
    meta.to_parquet(run_dir / "predictions_ensemble.parquet", index=False)

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(meta["label"], meta["p_up_mean"])
        log.info(f"Full-OOS AUC: {auc:.4f}")
    except Exception as e:
        log.warning(f"AUC failed: {e}")

    # ── 8. Backtest ──────────────────────────────────────────────────────────
    log.info("Running backtest…")
    from pattern.backtest.report import run_backtest
    holding = cfg.backtest.holding_period_days or cfg.label.horizon
    run_backtest(run_dir, cfg.backtest, holding_period_days=holding)

    log.info(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
