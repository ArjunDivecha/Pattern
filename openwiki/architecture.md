---
type: "Reference"
title: "Architecture"
description: "Core pipeline design of the Pattern CNN-based stock trend prediction system: Pydantic config, data loading, image rendering, CNN model, training loop, and backtest engine."
---

# Architecture

The Pattern pipeline converts daily OHLCV stock data into chart images, trains CNN ensembles to predict forward return direction, and constructs decile-sorted long-short portfolios. The entire system is driven by a Pydantic config loaded from YAML.

## Pipeline Overview

```
CSV (OHLCV + AdjClose)
  → load_data()           # pattern/data/loader.py — tidy DataFrame + Return column
  → compute_labels()      # pattern/data/loader.py — y = 1 if 20-day fwd return > 0
  → get_splits()           # pattern/data/splits.py — debug / expanding / rolling
  → build_cache()          # pattern/imaging/cache.py — memmap images.npy + index.parquet
  → train_model() × 5      # pattern/train/loop.py — 5-seed ensemble with early stopping
  → predict()              # pattern/train/loop.py — ensemble P(up), logits, embeddings
  → run_backtest()         # pattern/backtest/report.py — deciles, Sharpe, NW t, report.md
```

---

## Config System

**Source:** `pattern/config.py`

All behavior is driven by a Pydantic `Config` model loaded from YAML via `Config.from_yaml(path)`. The schema has seven nested sections:

| Section | Key class | Purpose |
|---|---|---|
| `data` | `DataConfig` | CSV path, date format, min history days, lifecycle boundary exclusion |
| `image` | `ImageConfig` | Window (20), height (64), width (60), OHLC ratio, MA/volume flags, cache dir |
| `label` | `LabelConfig` | Horizon (20 days), train-set class balancing |
| `model` | `ModelConfig` | Block count (3), channel progression [64,128,256], conv/pool/dilation params |
| `train` | `TrainConfig` | Ensemble size (5), batch size (128), lr (1e-5), max epochs (100), early stop patience (2), seeds, device, W&B toggle |
| `split` | `SplitConfig` | Mode (debug/expanding/rolling), train years, val fraction, retrain frequency, max train years (rolling) |
| `backtest` | `BacktestConfig` | N deciles (10), weighting, holding period (20), Newey-West lags (4) |

`ImageConfig` computes `ohlc_rows` and `vol_rows` from `height` and `ohlc_height_ratio` (51 OHLC rows, 1 gap, 12 volume rows for I20).

**Key design:** The first conv block uses asymmetric stride `(3,1)` and dilation `(2,1)` to compress the sparse first layer. Deeper blocks (2+) use stride `(1,1)`, dilation `(1,1)`, same-padding. This is encoded in `ModelConfig` as separate `conv_stride`/`conv_stride_inner` fields.

---

## Data Layer

### Loader (`pattern/data/loader.py`)

- `load_data(csv_path, date_format, min_history_days)` — Reads the OHLCV CSV, computes `Return` from `AdjClose` (Bloomberg total-return index), drops tickers with insufficient history. Returns a DataFrame sorted by `(Ticker, Date)`.
- `compute_labels(df, horizon)` — For each `(ticker, date)`, computes the cumulative log return over the next `horizon` days. Label = 1 if positive, 0 if negative. Stores `forward_return` and `label` columns.
- `build_ticker_index(df)` — Creates a per-ticker dict of sub-DataFrames for fast window lookups.
- `get_window(tdf, end_date, window, lookback)` — Extracts the OHLCV + TRI arrays for a single chart image.

### Splits (`pattern/data/splits.py`)

Three split modes, all with trading-day purge gaps to prevent label leakage across train/val/test boundaries:

| Mode | Function | Description |
|---|---|---|
| `debug` | `debug_split()` | First N years for train+val (time-based 70/30), next M years for test. Single window. |
| `expanding` | `expanding_splits()` | Training window grows each retrain cycle. Returns a list of `{train, val, test}` dicts — one per retraining window (27 for the full R1000 run). |
| `rolling` | `rolling_splits()` | Fixed-length or expand-then-roll training window. `max_train_years` caps the window; once reached, it trails. |

**Purge mechanism:** `purge_days = window + horizon - 1` trading days are dropped on both sides of every train/val and val/test boundary so no image window in the earlier set can overlap the forecast horizon of any image window in the later set.

**Class balancing:** `balance_labels(df, seed)` — Undersamples the majority class to 50/50. Applied to training set only; val and test are left at natural distribution.

---

## Imaging Layer

### Renderer (`pattern/imaging/renderer.py`)

- `render_window(ohlcv, tri, ...)` — Renders one chart image. Returns `(1, H, W)` uint8 array with binary pixels `{0, 255}`.
- `render_batch(samples, ...)` — Stacked batch renderer.

**Image geometry (I20):**
- 64 rows × 60 columns (3 columns per day × 20 days)
- Top 51 rows: OHLC region (High-Low bars, open/close ticks, 20-day MA line via Bresenham)
- Row 51: blank gap
- Bottom 12 rows: volume bars (from bottom up)

**Price normalization (per window):**
1. First close → 1.0; reconstruct via returns
2. Scale O/H/L by `p_t / close_t` (removes split/dividend artifacts)
3. Rescale vertical axis so min/max OHLC touches image bounds
4. Volume scaled independently (max volume = top of volume region)

**Missing-data handling:** If H or L is NaN → entire day's 3 columns left black. If only O or C is NaN → center H-L bar drawn, missing tick omitted. MA line skips undefined days.

### Cache (`pattern/imaging/cache.py`)

- `build_cache(labelled_df, img_cfg, lbl_cfg, cache_dir)` — Parallelized (multiprocessing) across tickers. Writes `images.npy` (memmap uint8, shape `(N, 1, H, W)`) and `index.parquet` (ticker, end_date, forward_return, label, has_ma, has_volume, window). ~20 GB for the full I20 universe.
- `load_cache(cache_dir)` — Returns `(images_memmap, index_df)`.
- `compute_pixel_stats(cache_dir, train_idx)` — Mean and std over training-set pixels for normalization.

The cache is immutable after build. All training shards and backtests read the same file (read-only after build, no races).

---

## Model Layer

### ConvBlock (`pattern/models/blocks.py`)

```
Conv2d → BatchNorm2d → LeakyReLU(0.01) → MaxPool2d(kernel=(2,1), stride=(2,1))
```

Conv2d uses `bias=False` (BatchNorm absorbs it). LeakyReLU uses `inplace=True`.

### ChartCNN (`pattern/models/cnn.py`)

Parametric CNN builder supporting I5/I20/I60 via config:

- `blocks` × ConvBlock (channels: 1 → 64 → 128 → 256 for I20)
- Flatten → Dropout(0.5) → Linear → 2 logits
- Weights: Xavier uniform; biases: zero
- No softmax internally (uses `nn.CrossEntropyLoss`)
- `forward_with_features()` — Returns logits + 256-dim global-avg-pooled embedding
- FC input size computed dynamically via a dummy forward pass (no hardcoded spatial dims)

**Spatial flow (I20, input 64×60):**
| Stage | Spatial Size | Channels |
|---|---|---|
| Input | 64 × 60 | 1 |
| After Block 1 | 32 × 60 | 64 |
| After Block 2 | 15 × 60 | 128 |
| After Block 3 | 7 × 60 | 256 |
| Head | — | Flatten → Dropout(0.5) → Linear → 2 logits |

---

## Training Layer

### Dataset (`pattern/train/dataset.py`)

- `LiveDataset` — Generates images on-the-fly from an OHLCV DataFrame (for debugging/small splits).
- `CachedDataset` — Reads from the pre-built memmap cache (for production runs). Normalizes pixels using training-set mean/std.

### Training Loop (`pattern/train/loop.py`)

- `train_model(model, train_loader, val_loader, cfg, run_dir, model_name, seed)` — Adam optimizer (lr=1e-5), batch=128, early stopping patience=2 on val loss, max 100 epochs. Saves best checkpoint as `.pt` and per-epoch `training_log.csv`.
- `predict(model, loader, device, return_features=True)` — Returns `(probs, labels, logits, embeddings)`.
- Device priority: CUDA → MPS (Apple Silicon) → CPU.
- W&B integration is optional (disabled in production configs).
- `non_blocking=True` only for CUDA; on MPS/CPU it can cause sporadic NaN logits.

---

## Backtest Layer

### Deciles (`pattern/backtest/deciles.py`)

- `build_portfolios(pred_df, n_deciles, score_col, return_col)` — Per formation date: rank stocks by `p_up_mean`, cut into N equal-size deciles (D1=lowest, D10=highest), compute equal-weighted mean forward return within each bucket. Days with fewer than `2 × n_deciles` stocks are skipped.
- `long_short_series(portfolios, n_deciles)` — D_top − D_bottom time series.

### Metrics (`pattern/backtest/metrics.py`)

- `newey_west_variance(x, lags)` — HAC (Bartlett-kernel) variance estimator for overlapping observations.
- `summarize_series(returns, holding_period_days, nw_lags)` — Annualized mean, volatility, Sharpe, NW t-stat, max drawdown (on non-overlapping sub-sample).
- `compute_turnover(pred_df, n_deciles, score_col)` — Jegadeesh-Titman-style symmetric-difference turnover per formation day.

### Report (`pattern/backtest/report.py`)

- `run_backtest(run_dir, cfg, holding_period_days)` — Loads predictions parquet, builds decile portfolios, computes all metrics, and writes:
  - `backtest_portfolios.parquet` — per-day decile log returns
  - `backtest_ls.parquet` — long-short time series
  - `backtest_decile_stats.xlsx` — per-decile and summary statistics
  - `backtest_cum_return.pdf` — cumulative return plot
  - `backtest_decile_bar.pdf` — per-decile annualized return bar chart
  - `report.md` — human-readable Markdown report

---

## CLI Entry Point (`pattern/cli.py`)

```bash
python -m pattern.cli train --config configs/debug.yaml
python -m pattern.cli train --config configs/prod_expanding.yaml --window-indices "0,3,5-9" --run-dir runs/expanding/<ts>
python -m pattern.cli backtest --config configs/prod_expanding.yaml --run-dir runs/expanding/<ts>
```

The `train` command orchestrates the full pipeline: load → label → split → cache → train ensemble → predict → save. When `--window-indices` is specified, it processes only those windows (used by multi-GPU drivers). When `--run-dir` is specified, it reuses an existing directory (for shared multi-GPU runs). The first shard writes `config.yaml`, `git_sha.txt`, and `pip_freeze.txt`; subsequent shards skip these.

Per-window outputs:
- `window_NN_predictions.parquet` — ticker, end_date, label, forward_return, per-seed `p_up_*`, `p_up_mean`, `p_up_std`, logits, rank_pct, decile, window
- `window_NN_features.npz` — per-seed logits `(K,N,2)`, embeddings `(K,N,256)`, ensemble-mean embedding `(N,256)`
- `window_stats.csv` — train/val/test years, sample counts, wall seconds, peak GPU memory

The final `predictions.parquet` (concatenated across all windows) is written when all windows are processed by a single invocation (not sharded).
