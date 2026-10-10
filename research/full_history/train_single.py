"""Train and evaluate the single full-history CNN fit.

This is intentionally a one-fit run.  It uses the same CNN, optimizer,
learning rate, batch size, seed, no-validation setup, and 20-epoch budget as
the full rolling5 study.  WandB is online by default so the run URL is saved
in wandb.json and can be given to the user.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/full_history_single_2019cutoff_seed101"
sys.path.insert(0, str(ROOT))
from pattern.config import ImageConfig, ModelConfig  # noqa: E402
from pattern.models.cnn import ChartCNN  # noqa: E402
from pattern.research.checkpoint import atomic_save  # noqa: E402
from pattern.research.development import evaluate  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def source_maps(sample: pd.DataFrame):
    maps = {}
    for shard in sample.cache_shard.unique():
        state = json.loads((ROOT / "cache" / shard / "state.json").read_text())
        maps[shard] = np.memmap(
            ROOT / "cache" / shard / "images.bin",
            mode="r",
            dtype="uint8",
            shape=(state["rows"], 1, 64, 60),
        )
    return maps


class SourceImages:
    def __init__(self, sample: pd.DataFrame):
        self.shards = sample.cache_shard.to_numpy()
        self.rows = sample.row_id.to_numpy(dtype=np.int64)
        self.maps = source_maps(sample)

    def get(self, ids: np.ndarray) -> np.ndarray:
        out = np.empty((len(ids), 1, 64, 60), dtype=np.uint8)
        for shard in np.unique(self.shards[ids]):
            where = np.flatnonzero(self.shards[ids] == shard)
            selected = ids[where]
            out[where] = self.maps[shard][self.rows[selected]]
        return out


def main() -> None:
    cfg = json.loads((OUT / "config.json").read_text())
    sample = pd.read_parquet(OUT / "sample.parquet")
    ret = sample.forward_return_5.to_numpy(dtype=np.float64)
    train = np.flatnonzero(sample.partition.eq("train") & np.isfinite(ret))
    test = np.flatnonzero(sample.partition.eq("comparison"))
    if len(train) == 0 or len(test) == 0:
        raise ValueError("empty train or test")

    images = SourceImages(sample)
    rng = np.random.default_rng(101)
    stat_ids = rng.choice(train, size=min(100_000, len(train)), replace=False)
    fraction = np.count_nonzero(images.get(stat_ids)) / images.get(stat_ids).size
    mean = np.float32(255 * fraction)
    std = np.float32(255 * np.sqrt(fraction * (1 - fraction)))
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"invalid image normalization std={std}")

    torch.manual_seed(101)
    device = torch.device("mps")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this run")
    torch.set_num_threads(3)
    model = ChartCNN(ModelConfig(), ImageConfig(window=20, width=60)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    batch_size = 256
    run_id = hashlib.sha256(str(OUT).encode()).hexdigest()[:16]
    import wandb
    wb = wandb.init(
        project="pattern-cnn",
        entity="arjun-divecha-dancing-elephant",
        id=run_id,
        resume="allow",
        name=cfg["name"],
        group="full_history_single_period",
        dir=str(OUT),
        config={**cfg, "batch_size": batch_size, "normalization_sample": len(stat_ids)},
        settings=wandb.Settings(disable_git=True, save_code=False),
    )
    atomic_json(OUT / "wandb.json", {
        "url": wb.url,
        "run_id": run_id,
        "project": "pattern-cnn",
        "entity": "arjun-divecha-dancing-elephant",
        "mode": os.getenv("WANDB_MODE", "online"),
    })

    history = []
    start_epoch = 1
    if (OUT / "last.pt").exists():
        cp = torch.load(OUT / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(cp["model"])
        opt.load_state_dict(cp["optimizer"])
        history = cp["history"]
        start_epoch = cp["epoch"] + 1

    def inputs(ids: np.ndarray):
        arr = images.get(ids).astype("float32")
        return torch.from_numpy((arr - mean) / std).to(device)

    for epoch in range(start_epoch, 21):
        started = time.monotonic()
        torch.manual_seed(101 + epoch * 10000)
        order = train.copy()
        np.random.default_rng(101 + epoch).shuffle(order)
        model.train()
        loss_sum = 0.0
        correct = 0
        for n in range(0, len(order), batch_size):
            ids = order[n:n + batch_size]
            y = torch.from_numpy((ret[ids] > 0).astype("int64")).to(device)
            logits = model(inputs(ids))
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {n}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(ids)
            correct += int((logits.argmax(1) == y).sum().item())
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train),
            "train_accuracy": correct / len(train),
            "seconds": time.monotonic() - started,
        }
        history.append(row)
        atomic_save(OUT / "last.pt", {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "epoch": epoch, "history": history,
        })
        pd.DataFrame(history).to_csv(OUT / "epochs.csv", index=False)
        atomic_json(OUT / "status.json", {"state": "training", **row})
        wb.log(row)
        print(row, flush=True)

    atomic_save(OUT / "final.pt", model.state_dict())
    model.eval()
    scores = []
    with torch.no_grad():
        for n in range(0, len(test), batch_size):
            scores.extend(torch.softmax(model(inputs(test[n:n + batch_size])), 1)[:, 1].cpu().numpy())
    result = sample.iloc[test][["security", "date"]].copy()
    result["score"] = scores
    result["forward_return"] = ret[test]
    cal = pd.DatetimeIndex(pd.read_parquet(OUT / "candidate_sessions.parquet").date)
    expected = pd.DatetimeIndex(result.date.unique()).sort_values()
    metrics, tables = evaluate(result, cal, 5, expected, 10)
    metrics.update({
        "method": cfg["method"],
        "train_rows": len(train),
        "test_rows": len(test),
        "normalization_sample": len(stat_ids),
        "return_basis": "raw next-open to fifth-session close outcomes",
    })
    result.to_parquet(OUT / "scores.parquet", index=False)
    atomic_json(OUT / "metrics.json", metrics)
    for name, table in tables.items():
        table.to_parquet(OUT / f"{name}.parquet", index=True)
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            wb.summary[k] = v
    wb.finish()
    atomic_json(OUT / "status.json", {"state": "complete", "epochs": 20, "known_train": len(train)})
    print(json.dumps({"metrics": metrics, "wandb_url": wb.url}, indent=2, default=str), flush=True)
    del model, opt, images
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
