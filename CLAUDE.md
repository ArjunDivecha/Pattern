# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Replication of Jiang, Kelly, Xiu (2023) *"(Re-)Imag(in)ing Price Trends"* (JF 78(6)). Trains an ensemble of 5 CNNs on OHLC+volume candlestick chart images to predict 20-day forward return direction (**I20/R20** config). Input: a single CSV of daily OHLCV data for ~1,000 US stocks.

Reference paper: `Pattern-2.pdf`. Full spec: `PRD.md`.

## Commands

```bash
# Install dependencies (once code exists)
pip install -r requirements.txt

# Run full pipeline (debug mode)
python -m pattern train --config configs/debug.yaml

# Run tests
pytest tests/ -v

# Run a single test file
pytest tests/test_renderer.py -v

# Run with production expanding-window splits
python -m pattern train --config configs/production.yaml
```

## Code Architecture

The suggested module layout (from PRD §11) — implement in this structure:

```
pattern/
  config.py          # Pydantic config models (schema in PRD §10)
  cli.py             # Entry point: `python -m pattern train/backtest`
  data/
    loader.py        # CSV → tidy DataFrame + adjusted returns
    splits.py        # debug / expanding / rolling train-val-test splits
  imaging/
    renderer.py      # Vectorized per-stock OHLC+MA+volume image generator
    cache.py         # memmap uint8 array (N,1,H,W) + parquet sidecar index
  models/
    blocks.py        # Single conv building block (Conv→BN→LeakyReLU→MaxPool)
    cnn.py           # Parametric CNN builder for I5/I20/I60 configs
  train/
    dataset.py       # PyTorch Dataset over memmap cache
    loop.py          # Ensemble training loop with early stopping
  backtest/
    deciles.py       # Cross-sectional decile portfolio construction
    metrics.py       # Sharpe, turnover, drawdown, Newey-West t-stats
    report.py        # Auto-generate report.md + plots
tests/
  test_renderer.py   # Pixel-exact checks on OHLC/MA/volume drawing
  test_labeling.py
  test_splits.py
```

## Key Implementation Details

### Image Geometry (I20)
- Size: 64×60 px, 1 channel (grayscale), binary pixels {0, 255}
- Each day = 3 columns: `3t` = open tick, `3t+1` = H–L bar + MA pixel + volume bar, `3t+2` = close tick
- OHLC region: top 51 rows; 1 blank gap row; volume region: bottom 12 rows

### Price Normalization (per window)
1. Normalize first day's close to 1.0; reconstruct via returns
2. Scale O/H/L by `p_t / close_t` to remove splits/dividends
3. Rescale vertical axis so min/max OHLC touches image bounds
4. Volume scaled independently: max volume = top of volume region

### CNN Block (I20 has 3 blocks, channels 1→64→128→256)
```
Conv2d(kernel=(5,3), stride=(3,1), padding=(12,1), dilation=(2,1))
→ BatchNorm2d → LeakyReLU(0.01) → MaxPool2d(kernel=(2,1), stride=(2,1))
```
Head: Flatten → Dropout(0.5) → Linear → 2 logits → Softmax

### Training
- Adam lr=1e-5, batch=128, early stopping patience=2, max 100 epochs
- 5 seeds → 5 models → ensemble-averaged `P(up)`
- Class-balance training set only (50/50 undersample); test left unbalanced

### Image Cache
- Single memmap `uint8` array per config, shape `(N, 1, H, W)` keyed by `(ticker, end_date, window, has_ma, has_volume)`
- Parquet sidecar with columns `ticker, end_date, label_h, forward_return, label`
- Expected size ~20 GB for full I20 dataset

### Output Structure
Each run writes to `runs/<timestamp>_<config_hash>/` — models, predictions parquet, training log CSV, portfolio parquets, and `report.md`.

## Hardware
M4 Max with 128 GB RAM. Use MPS for PyTorch:
```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```
Maximize multiprocessing for image generation (embarrassingly parallel per ticker).

## Data
- Input CSV columns: `Ticker, Date, Open, High, Low, Close, Volume` (optionally `AdjClose, Return, MarketCap`)
- If `Return` absent, compute from `AdjClose`; if `MarketCap` absent, value-weighted backtest falls back to equal-weighted with a warning
- Confirm with user: share volume vs dollar volume, and `MarketCap` availability before implementing VW backtest
