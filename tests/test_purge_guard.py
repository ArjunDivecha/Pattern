"""
=============================================================================
SCRIPT NAME: tests/test_purge_guard.py
=============================================================================

INPUT FILES:  none (synthetic fixture built in-memory)
OUTPUT FILES: none

DESCRIPTION:
Freezes the leakage guard that underwrites the entire out-of-sample claim.

`pattern/cli.py:132` sets `purge_days = image.window + label.horizon - 1`
(= 39 for the production I20/R20 config) and passes it to `get_splits`.
`_purge_tail` and `_purge_test_head` in `pattern/data/splits.py` then drop a
trading-day gap on both sides of every train->val and val->test boundary.

THE INVARIANT THIS PROTECTS (and why the gap has to be what it is):

A sample dated `t` carries an image covering [t-window+1, t] and a label
covering the forward return over (t, t+horizon]. So for a train sample at `t`
and a later val/test sample at `v`, the train label must finish strictly
before the val image begins:

    t + horizon  <  v - window + 1
    =>  v - t    >  window + horizon - 1
    =>  v - t    >= window + horizon          (in trading days)

which is exactly `purge_days + 1`. Anything less and the training label period
physically overlaps the evaluation image, i.e. the model is scored on data it
was fitted through.

`tests/test_splits.py` covers only `debug_split` at the DEFAULT `purge_days=0`
on a single-ticker monthly fixture, and asserts only naive set non-overlap --
it never checks the gap magnitude and never touches expanding/rolling. This
file closes that hole for all three modes at the production purge.

`test_zero_purge_is_detectably_unsafe` is the teeth check: it asserts the
invariant is VIOLATED at purge_days=0. If someone weakens the assertions
above, that test fails too, so the suite cannot silently go green.
=============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pattern.config import SplitConfig
from pattern.data.splits import get_splits

WINDOW = 20
HORIZON = 20
PROD_PURGE = WINDOW + HORIZON - 1          # 39, as pattern/cli.py:132 computes
REQUIRED_GAP = WINDOW + HORIZON            # 40 trading days between boundaries

N_TICKERS = 5


def _panel(start: str = "2000-01-03", end: str = "2007-12-31") -> pd.DataFrame:
    """Multi-ticker daily panel on a shared business-day calendar."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(0)
    frames = []
    for i in range(N_TICKERS):
        frames.append(pd.DataFrame({
            "Ticker": f"T{i}",
            "Date": dates,
            "label": rng.integers(0, 2, len(dates)),
            "forward_return": rng.normal(0, 0.05, len(dates)),
        }))
    return pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Date"])


def _cfg(mode: str) -> SplitConfig:
    return SplitConfig(
        mode=mode,
        start_year=2000,
        train_years=3,
        test_years=1,
        val_fraction=0.30,
        retrain_every_years=1,
        max_train_years=5 if mode == "rolling" else None,
    )


def _gap_trading_days(earlier: pd.DataFrame, later: pd.DataFrame,
                      cal: np.ndarray) -> int | None:
    """Trading-day distance from the last date of `earlier` to the first of `later`.

    Measured on the GLOBAL calendar, which is what the purge counts in -- a
    calendar-day gap would be wrong across weekends and holidays.
    """
    if len(earlier) == 0 or len(later) == 0:
        return None
    last = np.datetime64(earlier["Date"].max())
    first = np.datetime64(later["Date"].min())
    return int(np.searchsorted(cal, first) - np.searchsorted(cal, last))


@pytest.mark.parametrize("mode", ["debug", "expanding", "rolling"])
def test_boundary_gap_at_production_purge(mode):
    """INV1: every train->val and val->test boundary leaves >= window+horizon."""
    df = _panel()
    cal = np.sort(df["Date"].unique())
    splits = get_splits(df, _cfg(mode), purge_days=PROD_PURGE)
    assert splits, f"{mode}: produced no splits"

    checked = 0
    for i, sp in enumerate(splits):
        for a, b in (("train", "val"), ("val", "test")):
            gap = _gap_trading_days(sp[a], sp[b], cal)
            if gap is None:
                continue
            assert gap >= REQUIRED_GAP, (
                f"{mode} split {i}: {a}->{b} gap is {gap} trading days, "
                f"need >= {REQUIRED_GAP}. A train label spanning "
                f"{HORIZON} days would overlap a {WINDOW}-day evaluation image."
            )
            checked += 1
    assert checked > 0, f"{mode}: no boundaries were actually checked"


@pytest.mark.parametrize("mode", ["debug", "expanding", "rolling"])
def test_splits_are_pairwise_disjoint(mode):
    """INV2: no (ticker, date) appears in more than one of train/val/test."""
    df = _panel()
    splits = get_splits(df, _cfg(mode), purge_days=PROD_PURGE)
    for i, sp in enumerate(splits):
        keys = {k: set(zip(sp[k]["Ticker"], sp[k]["Date"])) for k in
                ("train", "val", "test")}
        for a, b in (("train", "val"), ("val", "test"), ("train", "test")):
            overlap = keys[a] & keys[b]
            assert not overlap, (
                f"{mode} split {i}: {len(overlap)} (ticker,date) rows shared "
                f"between {a} and {b}, e.g. {sorted(overlap)[:3]}"
            )


@pytest.mark.parametrize("mode", ["debug", "expanding", "rolling"])
def test_chronological_order(mode):
    """INV3: train strictly precedes val, val strictly precedes test."""
    df = _panel()
    splits = get_splits(df, _cfg(mode), purge_days=PROD_PURGE)
    for i, sp in enumerate(splits):
        if len(sp["train"]) and len(sp["val"]):
            assert sp["train"]["Date"].max() < sp["val"]["Date"].min(), \
                f"{mode} split {i}: train overruns val"
        if len(sp["val"]) and len(sp["test"]):
            assert sp["val"]["Date"].max() < sp["test"]["Date"].min(), \
                f"{mode} split {i}: val overruns test"


def test_purge_days_matches_cli_formula():
    """The guard is only correct if cli.py keeps deriving it from the config."""
    import inspect

    from pattern import cli
    src = inspect.getsource(cli)
    assert "cfg.image.window + cfg.label.horizon - 1" in src, (
        "pattern/cli.py no longer derives purge_days as window+horizon-1. "
        "REQUIRED_GAP in this file must be revisited before the change lands."
    )


def test_zero_purge_is_detectably_unsafe():
    """Teeth check: at purge_days=0 the invariant must FAIL.

    Guards against a future refactor that neuters the assertions above and
    leaves a green suite that proves nothing.
    """
    df = _panel()
    cal = np.sort(df["Date"].unique())
    splits = get_splits(df, _cfg("expanding"), purge_days=0)

    gaps = []
    for sp in splits:
        for a, b in (("train", "val"), ("val", "test")):
            g = _gap_trading_days(sp[a], sp[b], cal)
            if g is not None:
                gaps.append(g)

    assert gaps, "no boundaries produced at purge_days=0"
    assert min(gaps) < REQUIRED_GAP, (
        "purge_days=0 produced a safe gap -- the fixture or the split logic "
        "changed and this test no longer demonstrates the guard has teeth."
    )
