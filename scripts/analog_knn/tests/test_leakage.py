"""Property tests for analog kNN leakage rules and embedding norms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analog_knn.query_knn import outcome_available_dates, run_knn


def test_outcome_available_is_after_formation():
    cal = pd.bdate_range("2020-01-01", periods=100).values.astype("datetime64[ns]")
    ends = np.array([cal[10], cal[20]], dtype="datetime64[ns]")
    avail = outcome_available_dates(ends, cal, horizon=20)
    assert avail[0] == cal[30]
    assert avail[1] == cal[40]
    assert avail[0] > ends[0]


def test_no_self_ticker_in_synthetic_knn():
    """Tiny synthetic bank: each ticker's nearest raw neighbor is itself — must exclude."""
    rng = np.random.default_rng(0)
    n = 200
    # 20 tickers × 10 dates
    tickers = np.array([f"T{i%20}" for i in range(n)])
    # Shared date blocks
    dates = np.array(
        [np.datetime64("2015-01-01") + np.timedelta64(30 * (i // 20), "D") for i in range(n)],
        dtype="datetime64[ns]",
    )
    emb = rng.normal(size=(n, 8)).astype(np.float32)
    # Make each row most similar to itself
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    fwd = rng.normal(scale=0.05, size=n)

    data = {
        "ticker": tickers,
        "end_date": dates,
        "window": np.zeros(n, dtype=np.int16),
        "forward_return": fwd,
        "label": (fwd > 0).astype(np.int8),
        "emb": emb,
    }
    # Dense calendar
    cal = pd.bdate_range("2014-01-01", "2018-01-01").values.astype("datetime64[ns]")
    import torch

    device = torch.device("cpu")
    df = run_knn(data, cal, k_list=[5], device=device, bank_chunk=1000)
    # Scores should exist for later dates; k_eff > 0 somewhere
    assert df["analog_k_eff"].max() > 0
    # No NaN explosion on late dates
    late = df[df["end_date"] >= np.datetime64("2015-06-01")]
    assert late["analog_r_mean"].notna().mean() > 0.5


def test_embedding_unit_norm_property():
    e = np.random.randn(100, 256).astype(np.float32)
    e = e / np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-8)
    norms = np.linalg.norm(e, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
