"""Tests for data splitting logic in splits.py."""

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
