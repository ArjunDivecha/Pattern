"""
=============================================================================
SCRIPT NAME: test_splits.py
=============================================================================

DESCRIPTION:
    Tests for the data splitting and label-balancing functions in the Pattern
    library.  Generates synthetic monthly time-series DataFrames in memory
    (no external data) and verifies that:

      - debug_split() produces train/val/test splits with no date overlap.
      - Test dates are strictly after the latest train+val date.
      - The validation fraction matches the configured ratio within tolerance.
      - balance_labels() returns equal counts of each label class while
        preserving the minority class size.

INPUT FILES:
    (none — this script generates its own synthetic data in memory)

OUTPUT FILES:
    (none — this script only prints assertion results to stdout via pytest)

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - pandas
    - pytest
    - pattern.config (SplitConfig)
    - pattern.data.splits (debug_split, balance_labels)

USAGE:
    pytest tests/test_splits.py          (from project root)
    python -m pytest test_splits.py      (from tests/ directory)

NOTES:
    - All data is synthetic and deterministic; no fixtures or external
      dependencies beyond the packages listed above.
    - The _make_df helper produces 12 rows per year (one per month) for a
      single ticker with a fixed label of 1.
=============================================================================
"""

import pandas as pd
import pytest

from pattern.config import SplitConfig
from pattern.data.splits import debug_split, balance_labels


def _make_df(start_year=2000, n_years=6):
    rows = []
    for y in range(start_year, start_year + n_years):
        for m in range(1, 13):
            rows.append({"Ticker": "A", "Date": pd.Timestamp(y, m, 1), "label": 1})
    return pd.DataFrame(rows)


def test_debug_split_no_overlap():
    df  = _make_df(2000, 6)
    cfg = SplitConfig(mode="debug", train_years=3, val_fraction=0.3, test_years=2)
    out = debug_split(df, cfg)

    tr_dates = set(out["train"]["Date"])
    va_dates = set(out["val"]["Date"])
    te_dates = set(out["test"]["Date"])

    assert len(tr_dates & va_dates) == 0, "Train and val overlap"
    assert len(tr_dates & te_dates) == 0, "Train and test overlap"
    assert len(va_dates & te_dates) == 0, "Val and test overlap"


def test_debug_split_test_dates_after_train():
    df  = _make_df(2000, 6)
    cfg = SplitConfig(mode="debug", train_years=3, val_fraction=0.3, test_years=2)
    out = debug_split(df, cfg)

    max_tv = max(out["train"]["Date"].max(), out["val"]["Date"].max())
    min_te = out["test"]["Date"].min()
    assert min_te > max_tv, "Test must come strictly after train+val"


def test_debug_split_val_fraction():
    df  = _make_df(2000, 6)
    cfg = SplitConfig(mode="debug", train_years=3, val_fraction=0.3, test_years=2)
    out = debug_split(df, cfg)

    tv_total = len(out["train"]) + len(out["val"])
    val_frac = len(out["val"]) / tv_total
    assert abs(val_frac - 0.3) < 0.05, f"Val fraction {val_frac:.2f} not near 0.30"


def test_balance_labels_equal_counts():
    df = pd.DataFrame({"label": [1] * 70 + [0] * 30})
    b  = balance_labels(df)
    assert (b["label"] == 0).sum() == (b["label"] == 1).sum()


def test_balance_labels_preserves_minority():
    df = pd.DataFrame({"label": [1] * 70 + [0] * 30})
    b  = balance_labels(df)
    assert len(b) == 60   # 2 × 30
