"""Prepare one full-history training period and a 2020--2026 OOS period.

The study deliberately creates one fit, rather than an expanding sequence of
fits.  Training uses every eligible stock on the non-overlapping five-session
formation schedule available before the 2020 test period.  A 44-session gap
before the first test formation protects the 20-session image history and the
five-session forward label, matching the existing rolling5 study.

Images are not copied into a second 80GB+ array.  The sample keeps references
to the frozen cache shard and row id; train_single.py reads those source
memmaps in batches.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/full_history_single_2019cutoff_seed101"
DATA = ROOT / "data/research_universe/clean_I20_R5_v1/eligibility.parquet"
FROZEN = ROOT / "data/frozen/CURRENT.json"


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref = json.loads(FROZEN.read_text())
    frozen = ROOT / ref["path"]
    cal = pd.DatetimeIndex(pd.read_parquet(frozen / "candidate_sessions.parquet").date)
    d = pd.read_parquet(DATA)
    d["pos"] = cal.get_indexer(d.date)
    if (d.pos < 0).any():
        raise ValueError("eligibility rows outside the frozen candidate calendar")

    first_test = int(cal.searchsorted("2020-01-01"))
    train_cutoff = first_test - 44
    first_formation = int(cal.searchsorted("1996-01-01"))
    last_test = min(int(cal.searchsorted("2027-01-01")) - 1, len(cal) - 6)

    # Keep the same non-overlapping formation schedule as the rolling5 study.
    train = d[
        d.training_eligible
        & d.pos.between(first_formation, train_cutoff)
        & ((d.pos - first_formation) % 5 == 0)
    ].copy()
    test = d[
        d.formation_eligible
        & d.pos.between(first_test, last_test)
        & ((d.pos - first_test) % 5 == 0)
    ].copy()
    if train.empty or test.empty:
        raise ValueError("empty train or test partition")
    if train.pos.max() + 5 >= test.pos.min() - 39:
        raise ValueError("train/test information intervals are not purged")

    keep = [
        "security", "date", "pos", "row_id", "cache_shard", "close",
        "below_price_floor", "input_jump", "formation_eligible",
        "entry_date", "exit_date", "next_open_return_5_raw",
        "return_outlier", "label_valid", "training_eligible",
        "next_open_return_5_clean",
    ]
    train = train[keep].copy()
    test = test[keep].copy()
    train["partition"] = "train"
    test["partition"] = "comparison"
    train["forward_return_5"] = train.next_open_return_5_clean
    test["forward_return_5"] = test.next_open_return_5_raw
    sample = pd.concat([train, test], ignore_index=True)
    sample["sample_id"] = np.arange(len(sample), dtype=np.int64)
    sample.to_parquet(OUT / "sample.parquet", index=False)
    pd.DataFrame({"date": cal}).to_parquet(OUT / "candidate_sessions.parquet", index=False)

    config = {
        "name": "full_history_single_2019cutoff_seed101",
        "method": "single_full_history",
        "seed": 101,
        "train_start": str(cal[train.pos.min()].date()),
        "train_end": str(cal[train.pos.max()].date()),
        "test_start": str(cal[test.pos.min()].date()),
        "test_end": str(cal[test.pos.max()].date()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_known_labels": int(train.forward_return_5.notna().sum()),
        "test_known_outcomes": int(test.forward_return_5.notna().sum()),
        "test_years": sorted(test.date.dt.year.unique().astype(int).tolist()),
        "formation_stride": 5,
        "purge_sessions": 44,
        "epochs": 20,
        "validation": False,
        "clean_overlay": "clean_I20_R5_v1",
        "evaluation": "raw next-open outcomes; flagged outcomes retained and disclosed",
        "dataset": ref,
    }
    atomic_json(OUT / "config.json", config)
    atomic_json(OUT / "preflight.json", {
        "state": "prepared",
        "all_eligible_stocks": True,
        "sampled": False,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": str(train.date.min().date()),
        "train_end": str(train.date.max().date()),
        "test_start": str(test.date.min().date()),
        "test_end": str(test.date.max().date()),
    })
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
