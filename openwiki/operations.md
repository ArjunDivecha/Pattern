# Operations

## Configs

All configs live in `configs/` and are loaded via `Config.from_yaml(path)` (Pydantic validation).

| Config | Mode | Device | Purpose |
|---|---|---|---|
| `configs/debug.yaml` | debug | auto (MPS/CPU) | Smoke testing: 3yr train, 2yr test, single window, W&B enabled |
| `configs/production.yaml` | expanding | auto | Full-period expanding-window with 8-year initial train (not the multi-GPU prod configs) |
| `configs/prod_expanding.yaml` | expanding | cuda | 27 expanding-window retrains anchored at 1996, cache at `/data/Pattern/cache/prod_I20` |
| `configs/prod_rolling.yaml` | rolling | cuda | 27 rolling-window retrains (expand 3yr → 5yr cap, then trail), shared cache |

**Key config differences (expanding vs rolling):**

- Expanding: training window grows from `train_years` (3) to include all history up to the retrain date.
- Rolling: training window expands from `train_years` (3) up to `max_train_years` (5), then keeps a trailing 5-year window.

Both production configs share the same image cache at `/data/Pattern/cache/prod_I20` and the same CSV at `/data/Pattern/data/r1000_ohlcv_database.csv`.

---

## Run Directory Layout

Each run writes to `runs/<pathway>/<timestamp>_<config_hash>/`:

```
<timestamp>_<config_hash>/
  config.yaml                    # frozen copy of the training config
  git_sha.txt                     # git SHA at training time
  pip_freeze.txt                  # Python environment snapshot
  window_00_predictions.parquet    # per-window predictions
  ...
  window_26_predictions.parquet
  window_00_features.npz          # per-window logits + embeddings
  ...
  window_26_features.npz
  window_stats.csv               # per-window timing + GPU memory
  predictions.parquet            # concatenated final (9.25M rows for expanding)
  shard_gpu{0..7}.log            # multi-GPU driver logs
  shard_gpu{g}_w{w:02d}.log      # scheduler logs
  portfolios.parquet             # backtest output
  report.md                       # auto-generated narrative + plots
  decile_cumulative.pdf          # 10-decile log-scale cumulative returns
  top3_vs_bot3.pdf               # softer top/bot-30% version
  backtest_portfolios.parquet    # per-day decile log returns
  backtest_ls.parquet            # long-short time series
  backtest_decile_stats.xlsx     # per-decile + summary statistics
  backtest_cum_return.pdf        # cumulative return plot
  backtest_decile_bar.pdf        # per-decile annualized return bar chart
```

**Run dir naming:** `{YYYYMMDD_HHMMSS}_{config_hash}` where config_hash is the first 8 chars of the MD5 of the JSON-serialized config.

---

## Image Cache Layout

- `images.npy` — memmap `uint8` array `(N, 1, 64, 60)`, row order matches `index.parquet`
- `index.parquet` — columns: `ticker, end_date, label_h, forward_return, label, has_ma, has_volume, window`
- `pixel_stats.npz` — training-set pixel mean and std (for normalization)
- ~20 GB total for the full I20 universe (~5M samples)
- Immutable after build; every training shard and every backtest reads the same file
- Cache location configurable via `image.cache_dir` in the config

---

## Hardware

| Role | Hardware | Path |
|---|---|---|
| Remote training node | 8× NVIDIA A100-80GB | `/data/Pattern/` |
| Local development | M4 Max, 128 GB RAM, MPS | repo root |

**Compute budget (expanding pathway, actual):**
- 27 windows × 5 ensembles = 135 networks
- Wall-clock: 9h 27min (8× A100 pipelined)
- Sum of per-shard GPU hours: 64.74
- Mean per-window wall: 138.7 min
- Median peak GPU memory: 0.60 GB (batch 128)
- Cost: ~$101 at ~$10.69/h rental rate

---

## Webapp Deployment

The FastAPI live-scoring app is in `webapp/`.

### Files

| File | Purpose |
|---|---|
| `webapp/main.py` | FastAPI app: `GET /` serves the HTML page, `POST /score` scores a ticker, `GET /health` health check |
| `webapp/scorer.py` | Scoring engine: fetches OHLCV from yfinance, renders I20 image, runs full-history 5-seed ensemble, maps P(up) to 0–100 score |
| `webapp/templates/index.html` | Single-page UI |

### Model paths

The webapp loads the full-history ensemble from:
```
runs/final/equal_20260420_054833_cdef6809/ensemble_{0..4}.pt
runs/final/equal_20260420_054833_cdef6809/config.yaml
```

### Score mapping

Raw `P(up)` is mapped to a 0–100 score using historical distribution anchors from the expanding pathway:
- Score 0 → P(up) = 0.22075 (absolute min observed)
- Score 50 → P(up) = 0.50000 (neutral)
- Score 100 → P(up) = 0.66293 (absolute max observed)

### Running

```bash
cd webapp && uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Then open http://localhost:8000
```

**Note:** The webapp avoids Jinja2 templating (Python 3.14 + Jinja2 incompatibility — see commit 3acb7ad). It serves static HTML directly via `FileResponse`.

---

## Data Files

| File | Description |
|---|---|
| `data/r1000_ohlcv_database.csv` | Main R1000 OHLCV + AdjClose + Return + MarketCap (pipeline input) |
| `data/r1000_ohlcv_database.parquet` | Same, Parquet format (for analysis scripts) |
| `data/r1000_classifications.{xlsx,parquet}` | GICS + BICS classifications for R1000 tickers |
| `AssetList.xlsx` | 34-country ETF universe |
| `AssetList_liquid_etfs.xlsx` | 477 liquid ETF universe |
| `Industry28.xlsx` | 28-industry classification |
| `pattern.xlsx` | Pattern-related spreadsheet |

---

## Git and Reproducibility

Each run captures:
- `config.yaml` — frozen copy of the resolved config
- `git_sha.txt` — git HEAD at training time
- `pip_freeze.txt` — full Python environment

Two runs with identical config + seed produce bit-identical predictions (PRD §12 success criterion 5).

---

## Dependencies

Key dependencies from `requirements.txt`:

| Package | Purpose |
|---|---|
| `torch` | CNN model, training, inference |
| `numpy` | Array operations, image rendering |
| `pandas` | Data loading, manipulation |
| `pyarrow` | Parquet I/O |
| `pydantic` | Config validation |
| `pyyaml` | Config file parsing |
| `scikit-image` | Image operations (Bresenham line) |
| `scipy` | Statistical computations |
| `matplotlib` | Plotting |
| `fastapi`, `uvicorn`, `jinja2` | Webapp |
| `yfinance` | Live data fetching for webapp and ETF data |
| `openpyxl` | Excel I/O for analysis spreadsheets |
| `Pillow` | Image processing for webapp |

Additional (not in requirements.txt): `blpapi` + `BBGExtended` for Bloomberg data, `ib_insync` for IBKR integration.
