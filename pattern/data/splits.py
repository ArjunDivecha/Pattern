"""
=============================================================================
SCRIPT NAME: splits.py
=============================================================================
DESCRIPTION:
Implements the three split modes from PRD §7:
  debug     — first train_years for train+val (70/30 time-based), next test_years for test
  expanding — growing training window, retrain every R years
  rolling   — fixed-length training window

All splits drop a `purge_days` trading-day gap on both sides of every
train/val and val/test boundary so no image window in the earlier set
can overlap the forecast horizon of any image window in the later set.
=============================================================================
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pattern.config import SplitConfig

log = logging.getLogger(__name__)


def _year(ts: pd.Timestamp) -> int:
    return ts.year


def _purge_tail(df: pd.DataFrame, boundary_date, sorted_dates: np.ndarray,
                purge_days: int) -> pd.DataFrame:
    """Keep only rows whose Date is ≤ `purge_days` trading days before `boundary_date`.

    `sorted_dates` is the global array of unique trading dates (ascending).
    A no-op when purge_days=0.
    """
    if purge_days <= 0:
        return df
    boundary_idx = int(np.searchsorted(sorted_dates, np.datetime64(boundary_date), side="left"))
    cut_idx = boundary_idx - purge_days
    if cut_idx <= 0:
        return df.iloc[0:0].copy()
    cut_date = sorted_dates[cut_idx - 1]   # latest date allowed in kept set
    return df[df["Date"] <= cut_date].copy()


def _time_split_trainval(
    trainval: pd.DataFrame,
    val_fraction: float,
    purge_days: int,
    sorted_dates: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split trainval by date with a trading-day purge between train and val.

    Train = earliest (1 - val_fraction) of unique dates within trainval.
    Val   = latest val_fraction of unique dates.
    Train rows within `purge_days` trading days before the val boundary
    are dropped so no image window in train overlaps the forecast horizon
    of any image window in val.
    """
    tv_dates  = np.sort(trainval["Date"].unique())
    split_idx = int(np.floor(len(tv_dates) * (1.0 - val_fraction)))
    split_idx = max(1, min(split_idx, len(tv_dates) - 1))
    val_start = pd.Timestamp(tv_dates[split_idx])

    val   = trainval[trainval["Date"] >= val_start].copy()
    train = trainval[trainval["Date"] < val_start]
    train = _purge_tail(train, val_start, sorted_dates, purge_days)
    return train, val


def _purge_test_head(test: pd.DataFrame, val: pd.DataFrame,
                     sorted_dates: np.ndarray, purge_days: int) -> pd.DataFrame:
    """Drop the first `purge_days` trading days of test that are adjacent to val's last date."""
    if purge_days <= 0 or len(val) == 0 or len(test) == 0:
        return test
    val_end_date = np.datetime64(val["Date"].max())
    end_idx = int(np.searchsorted(sorted_dates, val_end_date, side="right"))
    test_start_idx = end_idx + purge_days
    if test_start_idx >= len(sorted_dates):
        return test.iloc[0:0].copy()
    test_start_date = sorted_dates[test_start_idx]
    return test[test["Date"] >= test_start_date].copy()


def debug_split(
    df: pd.DataFrame,
    cfg: SplitConfig,
    seed: int = 0,
    purge_days: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Debug split: first train_years → train+val (time-based, with purge),
    next test_years → test.  Remaining data is ignored.

    Time-based split prevents same-ticker windows at adjacent dates ending up
    in both train and val (which leaks labels due to ~90% label persistence).

    Returns {"train": df_tr, "val": df_val, "test": df_te}
    """
    sorted_dates = np.sort(df["Date"].unique())
    start_year = cfg.start_year or pd.Timestamp(sorted_dates[0]).year

    train_end_year = start_year + cfg.train_years - 1
    test_end_year  = train_end_year + cfg.test_years

    trainval = df[df["Date"].dt.year.between(start_year, train_end_year)].copy()
    test     = df[df["Date"].dt.year.between(train_end_year + 1, test_end_year)].copy()

    train, val = _time_split_trainval(trainval, cfg.val_fraction, purge_days, sorted_dates)
    test       = _purge_test_head(test, val, sorted_dates, purge_days)

    log.info(
        f"Debug split  "
        f"train={len(train):,}  val={len(val):,}  test={len(test):,}  "
        f"purge_days={purge_days} (trading days)  "
        f"({start_year}–{train_end_year} tv | {train_end_year+1}–{test_end_year} test)"
    )
    return {"train": train, "val": val, "test": test}


def expanding_splits(
    df: pd.DataFrame,
    cfg: SplitConfig,
    seed: int = 0,
    purge_days: int = 0,
) -> List[Dict[str, pd.DataFrame]]:
    """
    Expanding-window retraining splits (PRD §7b).
    Returns a list of {"train": ..., "val": ..., "test": ...} dicts,
    one per retraining epoch.
    """
    sorted_dates = np.sort(df["Date"].unique())
    start_year = cfg.start_year or pd.Timestamp(sorted_dates[0]).year
    max_year   = pd.Timestamp(sorted_dates[-1]).year

    splits = []
    train_end_year = start_year + cfg.train_years - 1

    while train_end_year + 1 <= max_year:
        test_end_year = min(
            train_end_year + cfg.retrain_every_years, max_year
        )

        trainval = df[df["Date"].dt.year.between(start_year, train_end_year)].copy()
        test     = df[df["Date"].dt.year.between(
            train_end_year + 1, test_end_year)].copy()

        train, val = _time_split_trainval(trainval, cfg.val_fraction, purge_days, sorted_dates)
        test       = _purge_test_head(test, val, sorted_dates, purge_days)
        splits.append({"train": train, "val": val, "test": test})

        train_end_year = test_end_year   # expand

    log.info(f"Expanding splits: {len(splits)} retraining windows  "
             f"purge_days={purge_days} (trading days)")
    return splits


def rolling_splits(
    df: pd.DataFrame,
    cfg: SplitConfig,
    seed: int = 0,
    purge_days: int = 0,
) -> List[Dict[str, pd.DataFrame]]:
    """
    Rolling window splits.

    If `cfg.max_train_years` is set, the window expands from `train_years` up
    to `max_train_years` then keeps a trailing `max_train_years` window
    (expand-then-roll).  If None, the window stays fixed at `train_years`.
    """
    sorted_dates = np.sort(df["Date"].unique())
    start_year = cfg.start_year or pd.Timestamp(sorted_dates[0]).year
    max_year   = pd.Timestamp(sorted_dates[-1]).year
    cap        = cfg.max_train_years or cfg.train_years

    splits = []
    train_end_year = start_year + cfg.train_years - 1

    while train_end_year + 1 <= max_year:
        train_len   = min(cap, train_end_year - start_year + 1)
        win_start   = train_end_year - train_len + 1
        test_end_year = min(train_end_year + cfg.retrain_every_years, max_year)

        trainval = df[df["Date"].dt.year.between(win_start, train_end_year)].copy()
        test     = df[df["Date"].dt.year.between(
            train_end_year + 1, test_end_year)].copy()

        train, val = _time_split_trainval(trainval, cfg.val_fraction, purge_days, sorted_dates)
        test       = _purge_test_head(test, val, sorted_dates, purge_days)
        splits.append({"train": train, "val": val, "test": test})

        train_end_year = test_end_year

    log.info(f"Rolling splits: {len(splits)} windows  cap={cap}yr  "
             f"purge_days={purge_days} (trading days)")
    return splits


def get_splits(
    df: pd.DataFrame,
    cfg: SplitConfig,
    seed: int = 0,
    purge_days: int = 0,
) -> List[Dict[str, pd.DataFrame]]:
    """Dispatch to the correct split mode."""
    if cfg.mode == "debug":
        return [debug_split(df, cfg, seed, purge_days)]
    elif cfg.mode == "expanding":
        return expanding_splits(df, cfg, seed, purge_days)
    elif cfg.mode == "rolling":
        return rolling_splits(df, cfg, seed, purge_days)
    else:
        raise ValueError(f"Unknown split mode: {cfg.mode}")


def balance_labels(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """
    Undersample the majority class to give 50/50 label balance (PRD §5).
    Applied to train only — val and test are left at natural class distribution
    so their metrics reflect the real-world class imbalance.
    """
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n   = min(len(pos), len(neg))
    rng = np.random.default_rng(seed)

    pos_idx = rng.choice(len(pos), size=n, replace=False)
    neg_idx = rng.choice(len(neg), size=n, replace=False)

    balanced = pd.concat([
        pos.iloc[pos_idx],
        neg.iloc[neg_idx],
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    log.info(f"  Balanced: {len(df):,} → {len(balanced):,}  ({n} pos, {n} neg)")
    return balanced
