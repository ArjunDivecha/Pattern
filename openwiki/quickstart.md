# Pattern — OpenWiki Quickstart

## What This Project Is

**Pattern** is a CNN-based stock price trend prediction pipeline that replicates and extends Jiang, Kelly & Xiu (2023), *"(Re-)Imag(in)ing Price Trends"* (JF 78(6)). It converts daily OHLCV data into 64×60 grayscale candlestick chart images and trains a 5-CNN ensemble to predict whether the next 20-day return will be positive. The ensemble-averaged `P(up)` score is then used to form decile-sorted long-short portfolios.

The pipeline has been run at production scale on the Russell 1000 universe (~3,000 tickers, 1996–2026) using expanding-window retraining on 8× NVIDIA A100 GPUs. Headline result: long-short D10−D1 portfolio with +17.4% annualized return, Sharpe 1.22, Newey-West t-stat +7.99 over 28 years of out-of-sample data.

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
| `README.md` | Full results, pipeline status, and research addenda (A–I) |
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
- [Workflows](workflows.md) — End-to-end workflows: training (debug/production), multi-GPU orchestration, backtesting, live scoring, data building, pathway merge
- [Domain & Strategy](domain.md) — The financial strategy, research findings, cross-sectional slicing, trash-tier/liquid-grid analysis, ETF deployment attempts, and cost/capacity conclusions
- [Operations](operations.md) — Configs, run directory layout, image cache, hardware setup, webapp deployment
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
