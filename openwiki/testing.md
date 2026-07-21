---
type: "Reference"
title: "Testing"
description: "Test coverage for the Pattern pipeline: pixel-exact renderer tests, labeling tests, split tests, what is not tested, and change-oriented guidance for future modifications."
---

# Testing

## Test Suite

All tests live in `tests/` and use `pytest`. They are fully self-contained — no external data files are read or written. All test data is generated synthetically in memory.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_renderer.py -v

# Run a specific test
pytest tests/test_renderer.py::test_flat_price -v
```

---

## Test Files

### `tests/test_renderer.py` — Pixel-Exact Renderer Tests

Validates that the chart image renderer (`pattern/imaging/renderer.py`) produces pixel-correct output. Generates synthetic OHLCV and total-return-index arrays in memory and verifies pixel positions against hand-computed expectations.

**Coverage:**
- Flat-price degenerate cases (all OHLC equal → predictable bar positions)
- Known High-Low bar pixel positions
- Open/close tick column placement
- Volume bar heights (scaled to max volume = top of volume region)
- NaN handling (missing High or Low → entire day's 3 columns left black)
- Gap-row blackout (the single blank row between OHLC and volume regions)
- Bresenham line-drawing correctness (for MA line segments)
- Insufficient-data edge cases (window too short → returns None)

**Why it matters:** The renderer is the heart of the pipeline. Any pixel regression changes what the CNN sees. These tests catch geometry bugs before they propagate to training. They mirror the renderer's internal geometry constants (`HEIGHT=64`, `WIDTH=60`, `WINDOW=20`, `OHLC_ROWS=51`, `VOL_ROWS=12`) to independently verify pixel positions from first principles.

### `tests/test_labeling.py` — Label Computation Tests

Validates `compute_labels()` from `pattern/data/loader.py`. Uses synthetic DataFrames with monotonically increasing `AdjClose` prices at a known daily return rate.

**Coverage:**
- Positive drift → label = 1
- Negative drift → label = 0
- Tail rows dropped when horizon extends past the end of data
- Multiple tickers labelled independently
- `forward_return` equals the expected cumulative log return over the horizon

**Why it matters:** Label correctness is foundational — if labels are wrong, the CNN learns the wrong target. The tests use trivially computable synthetic data (constant daily return) so expected outcomes are exact.

### `tests/test_splits.py` — Split and Balance Tests

Validates `debug_split()` and `balance_labels()` from `pattern/data/splits.py`. Generates synthetic monthly time-series DataFrames in memory.

**Coverage:**
- `debug_split()` produces train/val/test with no date overlap
- Test dates are strictly after the latest train+val date
- Validation fraction matches the configured ratio within tolerance
- `balance_labels()` returns equal counts of each label class
- Minority class size is preserved (undersampling, not oversampling)

**Why it matters:** Label leakage across split boundaries would invalidate all out-of-sample results. The purge mechanism (dropping `window + horizon - 1` trading days) is critical for preventing same-ticker windows at adjacent dates from leaking into both train and val.

---

## What's Not Tested

The test suite covers the three highest-risk areas (renderer geometry, label computation, split integrity). The following are not covered by unit tests:

- **CNN forward/backward pass** — No model-level tests; relies on training convergence as validation.
- **Cache build/load** — Not unit-tested; relies on training runs to exercise the cache.
- **Backtest metrics** — No unit tests for Sharpe/NW t/turnover; relies on the paper-replication benchmark (Sharpe ≥ 1.0 target).
- **Multi-GPU orchestration** — Integration-level; not unit-testable in isolation.
- **Webapp** — No automated tests; manual verification via browser.

---

## Change-Oriented Guidance

| When changing... | Watch out for | Relevant tests |
|---|---|---|
| Image geometry (`renderer.py`) | Pixel positions must still match hand-computed references | `test_renderer.py` |
| Label computation (`loader.py`) | Forward return calculation, tail handling | `test_labeling.py` |
| Split logic (`splits.py`) | Purge gap correctness, no date overlap | `test_splits.py` |
| CNN architecture (`cnn.py`, `blocks.py`) | Spatial flow must produce correct FC input size; first-block vs inner-block conv params | Training convergence on debug config |
| Config schema (`config.py`) | Pydantic validation, computed fields | All tests (config is loaded everywhere) |
| Backtest metrics (`metrics.py`) | Newey-West lag, annualization factor, turnover definition | Compare against paper Table II |
| Multi-GPU drivers | Shared run dir idempotency, window-indices parsing | Manual / integration |
