"""Tests for label computation in loader.py."""

import numpy as np
import pandas as pd
import pytest

from pattern.data.loader import compute_labels


def _make_df(tickers, n_days, base_tri=100.0, daily_return=0.01):
    """Synthetic DataFrame with monotonically increasing AdjClose."""
    rows = []
    for ticker in tickers:
        for i in range(n_days):
            tri = base_tri * (1 + daily_return) ** i
            rows.append({
                "Ticker": ticker,
                "Date":   pd.Timestamp("2000-01-03") + pd.Timedelta(days=i),
                "Open":   tri * 0.99,
                "High":   tri * 1.01,
                "Low":    tri * 0.98,
                "Close":  tri,
                "Volume": 1000.0,
                "AdjClose": tri,
            })
    return pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)


def test_label_positive_when_return_positive():
    """With a constant positive drift, all labels should be 1."""
    df = _make_df(["A"], n_days=60, daily_return=0.005)
    labelled = compute_labels(df, horizon=20)
    assert (labelled["label"] == 1).all()


def test_label_zero_when_return_negative():
    """With constant negative drift, all labels should be 0."""
    df = _make_df(["A"], n_days=60, daily_return=-0.005)
    labelled = compute_labels(df, horizon=20)
    assert (labelled["label"] == 0).all()


def test_forward_return_drops_tail_rows():
    """Rows where the horizon extends past the end should be dropped."""
    horizon = 20
    n_days  = 50
    df = _make_df(["A"], n_days=n_days)
    labelled = compute_labels(df, horizon=horizon)
    # Last `horizon` rows have no valid forward price
    assert len(labelled) == n_days - horizon


def test_multiple_tickers_independent():
    """Labels for each ticker should be computed independently."""
    df = _make_df(["A", "B"], n_days=60)
    labelled = compute_labels(df, horizon=20)
    tickers = labelled["Ticker"].unique()
    assert set(tickers) == {"A", "B"}


def test_forward_return_value():
    """forward_return should equal log(AdjClose_{t+H} / AdjClose_t)."""
    df = _make_df(["A"], n_days=30, daily_return=0.01)
    labelled = compute_labels(df, horizon=5)
    row = labelled.iloc[0]
    # At t=0: AdjClose=100, at t+5: AdjClose=100*(1.01)^5
    expected = np.log((1.01) ** 5)
    assert abs(row["forward_return"] - expected) < 1e-6
