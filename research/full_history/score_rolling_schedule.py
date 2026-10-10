"""Score the completed single-period model on the rolling study's dates.

The fit itself is unchanged.  This inference-only pass uses the annual
rolling5 formation schedule so the full-history and rolling models can be
compared on exactly the same stock-date observations.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/full_history_single_2019cutoff_seed101"
sys.path.insert(0, str(ROOT))
from pattern.config import ImageConfig, ModelConfig  # noqa: E402
from pattern.models.cnn import ChartCNN  # noqa: E402
from research.full_history.train_single import SourceImages  # noqa: E402


def main() -> None:
    cal = pd.DatetimeIndex(pd.read_parquet(OUT / "candidate_sessions.parquet").date)
    d = pd.read_parquet(
        ROOT / "data/research_universe/clean_I20_R5_v1/eligibility.parquet",
        columns=["security", "date", "row_id", "cache_shard", "formation_eligible", "next_open_return_5_raw"],
    )
    d["pos"] = cal.get_indexer(d.date)
    rows = []
    for year in range(2020, 2027):
        first = int(cal.searchsorted(f"{year}-01-01"))
        end = min(int(cal.searchsorted(f"{year + 1}-01-01")) - 1, len(cal) - 6)
        x = d[
            d.formation_eligible
            & d.pos.between(first, end)
            & ((d.pos - first) % 5 == 0)
        ].copy()
        rows.append(x)
    score_sample = pd.concat(rows, ignore_index=True)
    score_sample["forward_return"] = score_sample.next_open_return_5_raw
    score_sample["partition"] = "comparison"
    score_sample["sample_id"] = np.arange(len(score_sample), dtype=np.int64)

    original = pd.read_parquet(OUT / "sample.parquet")
    train = original.loc[original.partition.eq("train")].index.to_numpy()
    stat_ids = np.random.default_rng(101).choice(train, size=min(100_000, len(train)), replace=False)
    image_source_for_stats = SourceImages(original)
    stats_images = image_source_for_stats.get(stat_ids)
    fraction = np.count_nonzero(stats_images) / stats_images.size
    mean = np.float32(255 * fraction)
    std = np.float32(255 * np.sqrt(fraction * (1 - fraction)))

    source = SourceImages(score_sample)
    device = torch.device("mps")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this inference pass")
    model = ChartCNN(ModelConfig(), ImageConfig(window=20, width=60)).to(device)
    state = torch.load(OUT / "final.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    scores = []
    with torch.no_grad():
        for n in range(0, len(score_sample), 256):
            ids = np.arange(n, min(n + 256, len(score_sample)), dtype=np.int64)
            arr = source.get(ids).astype("float32")
            x = torch.from_numpy((arr - mean) / std).to(device)
            scores.extend(torch.softmax(model(x), 1)[:, 1].cpu().numpy())
    result = score_sample[["security", "date", "forward_return"]].copy()
    result["score"] = scores
    result = result[["security", "date", "score", "forward_return"]]
    result.to_parquet(OUT / "scores_rolling_schedule.parquet", index=False)
    manifest = {
        "rows": len(result), "dates": int(result.date.nunique()),
        "date_min": str(result.date.min().date()), "date_max": str(result.date.max().date()),
        "normalization_mean": float(mean), "normalization_std": float(std),
        "source_model": str(OUT / "final.pt"),
    }
    (OUT / "scores_rolling_schedule.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
