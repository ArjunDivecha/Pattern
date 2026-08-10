---
type: "Reference"
title: "Pattern — OpenWiki Quickstart"
description: "Entry point for the Pattern repository wiki. Covers what the project is, repository layout, prerequisites, quick-start commands, and links to all documentation sections."
---

# Pattern — OpenWiki Quickstart

## What This Project Is

**Pattern** is a CNN-based stock price trend prediction pipeline that replicates and extends Jiang, Kelly & Xiu (2023), *"(Re-)Imag(in)ing Price Trends"* (JF 78(6)). It converts daily OHLCV data into 64×60 grayscale candlestick chart images and trains a 5-CNN ensemble to predict whether the next 20-day return will be positive. The ensemble-averaged `P(up)` score is then used to form decile-sorted long-short portfolios.

The pipeline has been run at production scale on the Russell 1000 universe (~3,000 tickers, 1996–2025) using expanding-window retraining on 8× NVIDIA A100 GPUs. Headline result: long-short D10−D1 portfolio with +17.4% annualized return, Sharpe 1.22, Newey-West t-stat +7.99 over 27 years of out-of-sample data.

Reference paper: `Pattern-2.pdf`. Full specification: `PRD.md`. Strategy summary: `strategy_memo.md`.

---

## Repository Layout

| Path | Purpose |
|---|---|
| `pattern/` | Core Python package: config, data loading, image rendering, CNN model, training loop, backtest engine |
| `scripts/` | Orchestration, multi-GPU drivers, data fetching, analysis & slicing scripts, live scoring |
| `src/` | Bloomberg data builders (R1000, NKY 225) and classification pulls |
| `configs/` | YAML configs for debug, production, expanding-window, and rolling-window modes |
| `tests/` | Pixel-exact renderer tests, labeling tests, split tests |
| `webapp/` | FastAPI live-scoring web application |
| `README.md` | Full results, pipeline status, and research addenda (A–K, incl. the $5 price-floor and return-convention corrections) |
| `PRD.md` | Product Requirements Document (complete spec) |
| `strategy_memo.md` | 2-page strategy memo with architecture and R1000 results |

---

## Prerequisites

- Python 3.14, PyTorch with CUDA or MPS
- Key dependencies: `torch`, `numpy`, `pandas`, `pyarrow`, `pydantic`, `pyyaml`, `scikit-image`, `scipy`, `matplotlib`, `fastapi`, `uvicorn`, `yfinance` (see `requirements.txt`)
- Input data: a single CSV `data/r1000_ohlcv_database.csv` with columns `Ticker, Date, Open, High, Low, Close, Volume, AdjClose, Return, MarketCap`

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Debug run (local, MPS/CPU)
```bash
python -m pattern.cli train --config configs/debug.yaml
```

### Production expanding-window (multi-GPU)
```bash
# Pre-build the image cache (~20 GB)
python scripts/run_multi_gpu.py --config configs/prod_expanding.yaml --prebuild-cache

# Train on 8 GPUs
python scripts/run_multi_gpu.py --config configs/prod_expanding.yaml --n-gpus 8
```

### Backtest and report
```bash
python -m pattern.cli backtest --config configs/prod_expanding.yaml \
    --run-dir runs/expanding/<timestamp_hash>
```

### Run tests
```bash
pytest tests/ -v
```

### Launch the live scoring webapp
```bash
cd webapp && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Documentation Sections

- [Architecture](architecture.md) — Core pipeline design: data loading, image rendering, CNN model, training loop, backtest engine, and config system
- [Workflows](workflows.md) — End-to-end workflows: training (debug/production), multi-GPU orchestration, backtesting, live scoring, data building, price splice, Tracker paper book, daily publishing job
- [Domain & Strategy](domain.md) — The financial strategy, research findings, cross-sectional slicing, trash-tier/liquid-grid analysis, ETF deployment attempts, cost/capacity conclusions, the $5 price floor and return-convention corrections (Addenda J–K), and the Tracker paper book
- [Operations](operations.md) — Configs, run directory layout, image cache, hardware setup, daily publishing job, webapp deployment, data files
- [Testing](testing.md) — Test coverage, what each test validates, and how to run them

---

## Key Concepts at a Glance

| Concept | Description |
|---|---|
| **I20/R20** | 20-day chart images (I20) predicting 20-day forward return direction (R20) |
| **Chart image** | 64×60 px grayscale, binary pixels {0,255}: 3 columns per day (open tick, H-L bar + MA + volume, close tick) |
| **5-seed ensemble** | 5 CNNs with different seeds; outputs averaged for final `P(up)` |
| **Expanding window** | Training set grows each year; 28 windows cover 1999–2026 out-of-sample |
| **Decile portfolios** | Stocks ranked cross-sectionally by `P(up)` into 10 equal-size buckets; D10−D1 = long-short |
| **Newey-West t-stat** | Corrects for 20-day overlap in daily forward-return observations (lag=19, Bartlett kernel) |
| **Log-return convention** | `forward_return` is a 20-day LOG return; the published backtests report mean-log spreads, which inflate high-dispersion cells vs real portfolio (simple) returns — see [Domain](domain.md#return-convention-and-price-floor-corrections-addenda-jk) |
| **$5 price floor** | Deployable filter (`--min-price 5.0`, `mark_book.py` `MIN_PRICE=5.0`) dropping sub-penny bankruptcy shells that inflated the backtest ~2× — see [Domain](domain.md#return-convention-and-price-floor-corrections-addenda-jk) |
| **Tracker paper book** | Two sleeves (`pattern-trash`, `pattern-megacap`) published daily to `data/tracker/` for the Tracker hub; see [Workflows](workflows.md#tracker-paper-book-scriptsmark_bookpy) |
| **Daily job** | launchd `com.arjundivecha.pattern-daily` (weekdays 13:20 PT): splice → score → mark → verify; see [Workflows](workflows.md#daily-publishing-job-scriptsrun_dailysh) |

---

## Task Routing

| Change area / intent | Wiki page | Source entry points | Important symbols / types | Focused tests | Minimal validation |
|---|---|---|---|---|---|
| Renderer / image geometry | [Architecture](architecture.md) | `pattern/imaging/renderer.py` | `render_window`, `render_batch` | `tests/test_renderer.py` | `pytest tests/test_renderer.py -v` |
| Label / split logic (`pattern/data/`) | [Architecture](architecture.md), [Testing](testing.md) | `pattern/data/{loader,splits}.py` | `load_data`, `compute_labels`, `debug_split`, `balance_labels` | `tests/test_labeling.py`, `tests/test_splits.py` | `pytest tests/test_labeling.py tests/test_splits.py -v` |
| CNN architecture | [Architecture](architecture.md) | `pattern/models/cnn.py`, `pattern/models/blocks.py` | `ChartCNN`, `ConvBlock` | (none; training convergence on debug config) | `python -m pattern.cli train --config configs/debug.yaml` (blocked) |
| Backtest metrics / price floor | [Architecture](architecture.md), [Domain](domain.md) | `pattern/backtest/{deciles,metrics,report}.py`, `scripts/backtest_trash_tier.py`, `scripts/backtest_liquid_grid.py` | `newey_west_variance`, `summarize_series`, `--min-price` | (none; compare README §J.3) | `python scripts/backtest_trash_tier.py --min-price 5.0` |
| Tracker paper book / daily job | [Workflows](workflows.md), [Operations](operations.md) | `scripts/mark_book.py`, `scripts/run_daily.sh`, `scripts/com.arjundivecha.pattern-daily.plist` | `MIN_PRICE`, `track` column, `PATTERN DAILY OK` | (none; end-to-end) | `zsh scripts/run_daily.sh` (look for `PATTERN DAILY OK`) |
| Price splice / staleness | [Workflows](workflows.md) | `scripts/splice_daily_update.py`, `scripts/score_live.py` | `tradeable` flag, `--tickers-csv`, `--tail-days` | (none) | `python scripts/splice_daily_update.py --csv` |
| Config schema | [Architecture](architecture.md) | `pattern/config.py` | `Config`, `Config.from_yaml` | All tests load config | `python -c "from pattern.config import Config; Config.from_yaml('configs/debug.yaml')"` |
| Webapp | [Workflows](workflows.md), [Operations](operations.md) | `webapp/main.py`, `webapp/scorer.py` | FastAPI `app`, `score_ticker` | (none; manual) | `cd webapp && uvicorn main:app` |
| Strategy / research findings | [Domain](domain.md) | `README.md` addenda A–K, `strategy_memo.md`, `scripts/backtest_*.py` | — | (none) | — |

---

## Backlog

- **Tracker holdings page not yet wired.** `mark_book.py` publishes `holdings_latest.csv` and the hub row carries the position count, but the Tracker hub still needs a `(list[HoldingRow], BookCharacteristics)` loader (see `MEMO-TRACKER-INTEGRATION.md`). The book itself is published; only the hub display is missing. Reason deferred: work happens in the Tracker repo, not here.
