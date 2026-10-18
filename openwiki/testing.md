---
type: "Reference"
title: "Testing"
description: "Test coverage for the Pattern pipeline: pixel-exact renderer tests, labeling tests, split tests, what is not tested, and change-oriented guidance for future modifications."
---

# Testing

## Test Suite

All tests live in `tests/` and use `pytest`. They are fully self-contained — no external data files are read or written. All test data is generated synthetically in memory. The suite depends only on `numpy`, `pandas`, and `pytest` (plus `pattern.config` for `test_splits.py`); it does not require the image cache, a GPU, or any OHLCV database.

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

### `tests/test_purge_guard.py` — Leakage Guard

Freezes the leakage guard that underwrites the entire out-of-sample claim.
`pattern/cli.py:132` sets `purge_days = cfg.image.window + cfg.label.horizon - 1`
(= 39 for the production I20/R20 config) and passes it to `get_splits`;
`_purge_tail` and `_purge_test_head` in `pattern/data/splits.py` then drop a
trading-day gap on both sides of every train→val and val→test boundary.

**The invariant.** A sample at `t` carries an image over `[t-window+1, t]` and a
label over `(t, t+horizon]`, so a train sample at `t` and a later val/test
sample at `v` require `v − t ≥ window + horizon` (= 40) trading days, else the
training label period physically overlaps the evaluation image.

**Coverage (parametrized over `debug`, `expanding`, `rolling`):**
- `test_boundary_gap_at_production_purge` — every train→val and val→test
  boundary leaves ≥ `window + horizon` (40) trading days on the global calendar
- `test_splits_are_pairwise_disjoint` — no `(ticker, date)` appears in more than
  one of train/val/test
- `test_chronological_order` — train strictly precedes val, val precedes test
- `test_purge_days_matches_cli_formula` — asserts `pattern/cli.py` still
  contains the literal `cfg.image.window + cfg.label.horizon - 1` derivation
- `test_zero_purge_is_detectably_unsafe` — teeth check: at `purge_days=0` the
  invariant must be **violated**, so the suite cannot silently go green if the
  assertions above are weakened

**Why it matters:** `tests/test_splits.py` only ever exercised `debug_split`
at the default `purge_days=0` on a single-ticker monthly fixture, so the purge
that underwrites every published number — all from `expanding`/`rolling` — was
untested. This test closes that hole. See [Kill-or-Confirm Review — L.1](research.md#l1--the-leakage-guard-is-sound).

### `tests/test_strawman_frozen.py` — Strawman Baseline Freeze

Enforces `PREREGISTRATION.md` §2 in code rather than prose. Stage 2 compares
every candidate against a frozen baseline: the `combo` (neutralized CNN + WSTR)
decile long/short book. If that series silently changes — a rescored signal, a
different price panel, an edited harness — every spanning regression against it
measures the wrong thing, and nothing would look wrong.

**Coverage (skips, rather than fails, when the artefact is absent):**
- `test_strawman_checksum_and_length` — `sha256(round(series,10))[:16]` equals
  the frozen `59f55f36169346ca` and the series is exactly 282 months
- `test_strawman_headline_numbers` — net @8bp and NW t for `full` (+7.64%, t
  3.66) and `2005+` (+2.40%, t 1.86) match within tolerance
- `test_preregistration_document_present` — `PREREGISTRATION.md` exists and
  contains its key markers (`Promotion rule`, `lockbox`, `nky_ohlcv_database`)

The frozen artefact lives at
`runs/experiments/stage1b_eval/combo_resid_wstr/signal_card.xlsx` (gitignored,
so a fresh clone skips; the assertions bite whenever the file is present).
**Why it matters:** A pre-registration nobody can verify is a promise, not a
control. See [Kill-or-Confirm Review — The Pre-registration Protocol](research.md#the-pre-registration-protocol).

### Analog kNN leakage property tests

`scripts/analog_knn/tests/test_leakage.py` property-tests the kNN leakage rules:
`outcome_available_dates` returns dates strictly after the formation date
(`avail = formation + horizon`), and same-ticker self-exclusion removes each
query's own row from its neighbor set even when it is the nearest raw
neighbour. These guard the [Visual Analog kNN](research.md#extension-points-and-draft-prds)
embedding/kNN pipeline.

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
| Split logic (`splits.py`) | Purge gap correctness, no date overlap; the production `purge_days = window+horizon-1` derivation must stay in `pattern/cli.py` | `test_splits.py`, `test_purge_guard.py` |
| CNN architecture (`cnn.py`, `blocks.py`) | Spatial flow must produce correct FC input size; first-block vs inner-block conv params | Training convergence on debug config |
| Config schema (`config.py`) | Pydantic validation, computed fields | All tests (config is loaded everywhere) |
| Backtest metrics (`metrics.py`) | Newey-West lag, annualization factor, turnover definition | Compare against paper Table II |
| Backtest price floor (`backtest_trash_tier.py`, `backtest_liquid_grid.py` `--min-price`) | `--min-price 5.0` is the deployable setting; filtering must happen **before** bucketing (same order as `mark_book.py`) or the cell shrinks and stops matching the tracked book | Compare against README §J.3 corrected table |
| Tracker book / `mark_book.py` | The verify step asserts each sleeve's `meta.json` `as_of` equals the spliced panel's last date; the backtest/paper `track` split must stay distinguishable; benchmark is cash, `ew_cell` is context only | Run `scripts/run_daily.sh` and check for `PATTERN DAILY OK`; compare `mark_book.py` NAV CAGR against the analytic +4.15% figure (README §K.3) |
| Multi-GPU drivers | Shared run dir idempotency, window-indices parsing | Manual / integration |
