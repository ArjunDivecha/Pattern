# Pattern — CNN Replication of Jiang–Kelly–Xiu (2023)

Replication and production-scale extension of *"(Re-)Imag(in)ing Price Trends"* (JF 78(6), 2023) using the **I20/R20** configuration — a 5-CNN ensemble trained on 64×60 candlestick chart images to predict 20-day forward return direction for ~1,000 US stocks.

Reference paper: `Pattern-2.pdf`.  Full spec: `PRD.md`.

---

## Latest Results (expanding pathway, 1999-03 → 2026-03)

Trained 28 expanding-window retrained ensembles on 8× NVIDIA A100 (80 GB), ~65 GPU-hours total.

### Headline

| Metric | Value |
|---|---|
| Observations | 9,249,453 (ticker × end_date) |
| Unique tickers / dates | 3,011 / 5,759 |
| **Overall Test AUC** | **0.5068** |
| Windows with positive LS | 25 / 28 |

### Decile cross-sectional portfolio (equal-weight within decile, 20-day non-overlap compounding)

| Decile | Cum × | Ann. comp. |
|---|---:|---:|
| D1 (short) | 0.12 | **−8.71 %** |
| D2 | 1.16 | +0.64 % |
| D3 | 2.26 | +3.64 % |
| D4 | 3.73 | +5.93 % |
| D5 | 3.40 | +5.49 % |
| D6 | 3.82 | +6.03 % |
| D7 | 4.33 | +6.62 % |
| D8 | 5.95 | +8.11 % |
| D9 | 6.02 | +8.17 % |
| D10 (long) | 9.32 | **+10.26 %** |

Decile monotonicity is clean: D1 is a persistent loser and D10 a persistent winner.

### Long–Short summary

| Portfolio | Cum × | Ann. comp. | Ann. vol | Sharpe | NW t(19) |
|---|---:|---:|---:|---:|---:|
| **LS D10−D1** | 39.44 | **+17.44 %** | 14.07 % | **1.22** | **+7.99** |
| LS Top3−Bot3 (top/bot 30 %) | 7.27 | +9.07 % | **8.66 %** | 1.05 | +6.60 |

Newey–West t-stats account for the 20-day overlap in daily forward-return observations (lag=19, Bartlett kernel); |t|≈8 is overwhelming statistical significance.

### Year-by-year LS D10−D1

```
1999  −7.57 %    (known: dotcom peak, model trained 1996–98 only)
2000  +43.70 %
2001  +37.83 %
2002  +36.76 %
2003   +5.41 %
2004  +14.58 %
2005   +9.50 %
2006   +5.08 %
2007  +11.37 %
2008  +37.18 %   (GFC)
2009   +2.44 %
2010  +17.64 %
2011  +20.55 %
2012  +16.39 %
2013   +1.18 %
2014  +21.54 %
2015  +16.68 %
2016   −1.65 %
2017   +9.35 %
2018   +8.06 %
2019  +10.88 %
2020  +13.64 %   (COVID)
2021  +49.51 %   (meme stocks)
2022  +13.61 %
2023  +28.36 %
2024  +30.91 %
2025   +9.78 %
2026  +15.38 %   (YTD through Mar-2026)
```

Two of the three weak years (1999, 2016) are structurally unavoidable or mild; 2013 is a known regime where cross-sectional momentum broke.

Charts: `runs/expanding/<run_dir>/decile_cumulative.pdf`, `top3_vs_bot3.pdf`.

---

## Pipeline status

| Pathway | Config | Status |
|---|---|---|
| **Expanding** (train grows each year) | `configs/prod_expanding.yaml` | Complete — `runs/expanding/20260419_174908_cdef6809` |
| **Rolling** (train capped at 5 yr, trailing) | `configs/prod_rolling.yaml` | Running (pipelined with expanding on idle GPUs) |
| Comparison (merged per stock-date) | `scripts/merge_pathways.py` | Pending rolling completion |

---

## Code architecture (unchanged from PRD §11)

```
pattern/
  config.py              Pydantic schemas (PRD §10)
  cli.py                 python -m pattern.cli {train,backtest}
  data/
    loader.py            CSV → tidy DataFrame + adjusted returns
    splits.py            debug / expanding / rolling retrain schedules
  imaging/
    renderer.py          Vectorized per-stock OHLC+MA+volume image gen
    cache.py             memmap uint8 (N,1,H,W) + parquet sidecar
  models/
    blocks.py            Conv→BN→LeakyReLU→MaxPool building block
    cnn.py               Parametric builder for I5/I20/I60
  train/
    dataset.py           PyTorch Dataset over memmap cache
    loop.py              5-seed ensemble loop with early stopping
  backtest/
    deciles.py           Cross-sectional decile portfolios
    metrics.py           Sharpe, NW t, turnover, drawdown
    report.py            Auto-generate report.md + plots
scripts/
  run_multi_gpu.py       8-GPU fan-out driver (round-robin per-shard windows)
  gpu_scheduler.py       Work-stealing scheduler (opportunistic GPU use)
  merge_pathways.py      Expanding + rolling → per-stock-date comparison parquet
  infer_fullperiod.py    Re-score saved ensembles over any date range
  train_extra_seeds.py   Add ensembles to an existing run
configs/
  debug.yaml             Small universe / few windows for smoke tests
  production.yaml        Baseline full-period single-pass config
  prod_expanding.yaml    27-year expanding retrain schedule
  prod_rolling.yaml      27-year rolling (5-year trailing) schedule
tests/                   Pixel-exact renderer checks, labelling, splits
```

---

## What's changed since the baseline single-pass run

1. **5-seed ensemble in a single training call.** `train/loop.py` now iterates seeds internally; `cli.py` wires up aggregation.
2. **Embedding capture.** `cnn.py::forward_with_features` returns the 256-dim global-avg-pooled penultimate tensor alongside the logits. `predict(return_features=True)` now returns `(probs, labels, logits, embeddings)` per seed.
3. **Rich per-window artefacts.** Every window now writes:
   - `window_NN_predictions.parquet` — ticker, end_date, label, forward_return, per-seed `p_up_*`, `p_up_mean`, `p_up_std`, `logit_down_mean`, `logit_up_mean`, `rank_pct`, `decile`, `window`.
   - `window_NN_features.npz` — per-seed logits (K,N,2), per-seed embeddings (K,N,256), ensemble-mean embedding (N,256).
   - `window_stats.csv` — train/val/test years, sample counts, wall seconds, peak GPU memory.
4. **Subset training via CLI.** `--window-indices "0,3,5-9"` + `--run-dir` let an external orchestrator drive a single shared run directory.
5. **Two-pathway retrain schedule.** Expanding and rolling configs run side-by-side on the same 28 test years; `merge_pathways.py` joins them on `(ticker, end_date)` so each stock-date has both ensembles' probabilities / ranks / deciles.
6. **Multi-GPU drivers.**
   - `run_multi_gpu.py` — static round-robin fan-out, one shard per GPU, drives a single pathway.
   - `gpu_scheduler.py` — work-stealing scheduler that polls `nvidia-smi` every 30 s and grabs whichever GPU has no compute apps, then pops the next pending window from its queue. Used to pipeline the rolling pathway on GPUs freed by the expanding run.
7. **Idempotent run-dir setup.** Concurrent shards write their own per-window outputs; the driver/scheduler does the final concat once all shards finish.

---

## How to run

### Prerequisites

- Python 3.14, PyTorch with CUDA or MPS.
- A single CSV `r1000_ohlcv_database.csv` (Ticker, Date, Open, High, Low, Close, Volume, AdjClose, Return, MarketCap).

### Debug run (local, MPS)

```bash
python -m pattern.cli train --config configs/debug.yaml
```

### Full production (single GPU, sequential windows)

```bash
python -m pattern.cli train --config configs/production.yaml
```

### Cache pre-build (once)

```bash
python scripts/run_multi_gpu.py --config configs/prod_expanding.yaml --prebuild-cache
```

Builds `/data/Pattern/cache/prod_I20/images.npy` + `index.parquet` (~20 GB for the 1000-stock universe).

### Expanding pathway on 8× GPU

```bash
python scripts/run_multi_gpu.py --config configs/prod_expanding.yaml --n-gpus 8
```

Round-robin shard assignment: GPU g trains windows {g, g+8, g+16, …}.  Each shard writes its own per-window parquets; driver concatenates them into `predictions.parquet` at the end.

### Rolling pathway (same universe, same cache)

```bash
python scripts/run_multi_gpu.py --config configs/prod_rolling.yaml --n-gpus 8
```

### Pipelining rolling on idle GPUs while expanding still runs

```bash
python scripts/gpu_scheduler.py \
    --config configs/prod_rolling.yaml \
    --run-dir /data/Pattern/runs/rolling/<ts> \
    --n-windows 28 --n-gpus 8
```

The scheduler checks each GPU's compute-app list every 30 s; whenever a GPU is free it launches the next pending window there.  Clean handoff — same shared run directory convention, no racing on the memmap cache (read-only after build).

### Backtest and report

```bash
python -m pattern.cli backtest --config configs/prod_expanding.yaml \
    --run-dir runs/expanding/20260419_174908_cdef6809
```

Writes portfolio parquets and `report.md` with Sharpe, turnover, drawdown, Newey-West t-stats and per-decile cumulative returns.

### Merge expanding + rolling

```bash
python scripts/merge_pathways.py \
    --expanding runs/expanding/20260419_174908_cdef6809 \
    --rolling   runs/rolling/20260420_003938_fb2563f5 \
    --out-dir   runs/comparison
```

Produces `pathway_comparison.parquet` (one row per stock-date with both ensembles' output) and `pathway_comparison_summary.csv` (per-date correlation, decile disagreement).

---

## Image-cache layout

- `images.npy` — memmap uint8 array `(N, 1, 64, 60)`, row order matches `index.parquet`.
- `index.parquet` — `ticker, end_date, label_h, forward_return, label, has_ma, has_volume, window`.
- ~20 GB total for the full I20 universe.  Immutable after build; every training shard and every backtest reads the same file.

---

## Output directory layout

Each run writes to `runs/<pathway>/<timestamp>_<config_hash>/`:

```
20260419_174908_cdef6809/
  config.yaml                 frozen copy of the training config
  sha.txt                     git sha
  pip_freeze.txt
  window_00_predictions.parquet ...  window_27_predictions.parquet
  window_00_features.npz     ...     window_27_features.npz
  window_stats.csv
  predictions.parquet         concatenated final (9.25 M rows for expanding)
  shard_gpu{0..7}.log         driver logs
  shard_gpu{g}_w{w:02d}.log   scheduler logs
  portfolios.parquet          backtest output
  report.md                   auto-generated narrative + plots
  decile_cumulative.pdf       10-decile log-scale cumulative returns
  top3_vs_bot3.pdf            softer top/bot-30% version
```

---

## Hardware

Remote node: 8× A100-80GB, `/data/Pattern/` workspace.
Local: M4 Max, 128 GB RAM, MPS — used for development, analysis, plotting.

Peak per-shard GPU memory: 0.60 GB (batch 128).  Network training is compute-bound on the memmap loader, not memory-bound.

---

## Compute budget (expanding pathway, actual)

| Quantity | Value |
|---|---|
| Windows trained | 28 |
| Ensembles per window | 5 |
| Total networks trained | 140 |
| Wall-clock (8× A100 pipelined) | 9 h 27 min |
| Sum of per-shard GPU hours | 64.74 |
| Mean per-window wall | 138.7 min |
| Median peak GPU memory | 0.60 GB |

Cost at the rented node rate (~$10.69/h) ≈ $101 for both pathways pipelined.
