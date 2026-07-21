---
type: "Reference"
title: "Workflows"
description: "End-to-end workflows for the Pattern pipeline: Bloomberg data building, debug and production training, multi-GPU orchestration, backtesting, live scoring, and pathway merging."
---

# Workflows

This page covers the end-to-end workflows: data building, training (debug and production), multi-GPU orchestration, backtesting, live scoring, and pathway merging.

---

## Data Building

### R1000 Database (`src/build_r1000_database.py`)

Builds a point-in-time historical database of every stock that was ever a Russell 1000 member (1996–present), including delisted names.

**Three phases:**
1. **Constituents** — For each year-end, calls Bloomberg `INDX_MEMBERS BDS` on `"RIY Index"` with `END_DT` override. Per-year JSON saved for crash recovery.
2. **Consolidate** — Deduplicates tickers across years, pulls metadata (name, GICS sector, exchange, SEDOL). Writes `data/constituents/master_tickers.xlsx`.
3. **OHLCV** — Pulls daily OHLCV + total-return index for every ticker from 19960101 to today via `hist_batch()` (50 tickers per Bloomberg request). Each batch saved as staging parquet for resume safety. Final step combines into `data/r1000_ohlcv_database.csv` and `.parquet`.

**Prerequisites:** Bloomberg Terminal running on Windows/Parallels, `blpapi`, `BBGExtended`. Expected runtime: 1–3 hours.

### NKY 225 Database (`src/build_nky_database.py`)

Analogous to R1000 but for the Nikkei 225. Uses `INDX_MWEIGHT_HIST` on `"NKY Index"`. Early years (pre-2005) backfilled with earliest non-empty year's member list (NKY 225 has ~5% annual turnover).

### Classification Pull (`src/pull_r1000_classifications.py`)

Pulls both GICS and BICS sector/industry classifications from Bloomberg for every ticker in the R1000 prediction universe. Outputs `data/r1000_classifications.{xlsx,parquet}` with columns for GICS sector/industry group/industry/sub-industry and BICS levels 1–3.

### ETF Data (`scripts/fetch_etf_data.py`)

Pulls daily OHLCV from yfinance for every ticker in `AssetList.xlsx` (34 country ETFs). Output schema matches `r1000_ohlcv_database.csv` so the existing pipeline consumes it unchanged.

### SSE Data (`scripts/fetch_sse_data.py`)

Fetches OHLCV for single-stock ETF underlyings and their leveraged wrappers (e.g., TSLA → TSLQ/TSLZ). Validates pair availability via yfinance.

---

## Training

### Debug Run (local)

```bash
python -m pattern.cli train --config configs/debug.yaml
```

Uses `configs/debug.yaml`: 3-year train/val, 2-year test, single window, MPS device, W&B enabled. Good for smoke testing on a laptop.

### Production Expanding-Window (multi-GPU)

```bash
# 1. Pre-build the image cache (~20 GB, once)
python scripts/run_multi_gpu.py --config configs/prod_expanding.yaml --prebuild-cache

# 2. Train on 8 GPUs
python scripts/run_multi_gpu.py --config configs/prod_expanding.yaml --n-gpus 8
```

Uses `configs/prod_expanding.yaml`: 27 expanding windows anchored at 1996. Window 0 trains on 1996–98, tests 1999. Window 26 trains on 1996–2024, tests 2025. Each window trains a 5-seed ensemble (135 total networks). Cache is shared across all windows and GPUs (read-only after build).

### Production Rolling-Window (multi-GPU)

```bash
python scripts/run_multi_gpu.py --config configs/prod_rolling.yaml --n-gpus 8
```

Uses `configs/prod_rolling.yaml`: window expands from 3 years up to 5 years, then trails at 5 years. Same shared cache as expanding. 27 windows.

### Full-History Live Model (`scripts/train_full_for_live.py`)

Trains one 5-seed ensemble on the entire image cache (1996 to present) with no validation/test split. Supports equal-weight or EWMA (exponentially-weighted moving average) sample weighting. This is the final model used by the live scoring webapp.

```bash
python scripts/train_full_for_live.py --config configs/prod_expanding.yaml --tag equal
python scripts/train_full_for_live.py --config configs/prod_expanding.yaml --tag ewma_5yr --weight-mode ewma --half-life-years 5
```

### Extra Seeds (`scripts/train_extra_seeds.py`)

Adds additional ensemble members to an existing run directory.

---

## Multi-GPU Orchestration

### Static Round-Robin (`scripts/run_multi_gpu.py`)

Enumerates all windows for a config, assigns them round-robin to GPUs 0..N−1, spawns one subprocess per GPU with `CUDA_VISIBLE_DEVICES` pinned and `--window-indices` set to that GPU's shard. Each subprocess sees only its one device as `cuda:0`.

After all shards finish, the driver concatenates per-window prediction parquets into `predictions.parquet` and prints an AUC check.

### Work-Stealing Scheduler (`scripts/gpu_scheduler.py`)

Polls `nvidia-smi` every 30 seconds; whenever a GPU has no compute apps attached, pops the next pending window index from the queue and launches a training subprocess on that GPU. Used to pipeline the rolling pathway on GPUs freed by the expanding run.

```bash
python scripts/gpu_scheduler.py \
    --config configs/prod_rolling.yaml \
    --run-dir /data/Pattern/runs/rolling/<ts> \
    --n-windows 27 --n-gpus 8
```

Per-window logs go to `shard_gpu{g}_w{w:02d}.log`. Exits when the queue is empty and all launched shards have finished.

---

## Backtesting

### Standard Backtest

```bash
python -m pattern.cli backtest --config configs/prod_expanding.yaml \
    --run-dir runs/expanding/<timestamp_hash>
```

Loads `predictions.parquet` from the run directory, builds decile portfolios, computes metrics, and writes `report.md`, `backtest_*.{pdf,xlsx,parquet}`. See [Architecture](architecture.md#backtest-layer) for details.

### Generic Backtest (`scripts/backtest_generic.py`)

Reusable backtest over any `predictions.parquet` + optional universe filter. Ranks tickers cross-sectionally by `p_up_mean`, builds top X% / bottom X% portfolios, computes Sharpe, NW t, turnover, and yearly stats. Used as the foundation for most slicing experiments.

### Cross-Sectional Slicing Scripts

| Script | Purpose |
|---|---|
| `backtest_by_sector.py` | LS by BICS sector/industry |
| `backtest_by_mcap_proxy.py` | LS by dollar-volume tertiles |
| `backtest_by_feature.py` | LS by momentum, realized vol, prediction disagreement |
| `backtest_trash_tier.py` | Intersection of small + high-vol + recent-loser filters |
| `backtest_liquid_grid.py` | 27-cell grid search (dv × vol × mom) with cost-adjusted net returns |
| `trash_tier_turnover.py` | Month-over-month ticker turnover for trash-tier |
| `trash_tier_yearly.py` | Per-calendar-year LS return and turnover |
| `backtest_sse.py` | Backtest using single-stock ETF wrappers as trading instruments |
| `backtest_sse_momentum.py` | Alternative momentum signals on SSE universe |

See [Domain & Strategy](domain.md) for the research findings from these scripts.

---

## Live Scoring

### Batch Live Scoring (`scripts/score_live.py`)

Scores the tail of trading days (no forward return available yet) using the most recent expanding-window checkpoint (w26, trained on 1996–2024).

```bash
python scripts/score_live.py \
    --data-csv data/etf_ohlcv.csv \
    --out-parquet runs/etf_expanding/live_predictions.parquet
```

### Expanding-Window Scoring (`scripts/score_with_expanding.py`)

Scores a data file using the full expanding-window ensemble (all 27 windows).

### Webapp (`webapp/`)

FastAPI application for interactive single-ticker scoring. Fetches OHLCV from yfinance, renders the I20 chart, runs the full-history 5-seed ensemble, and maps raw `P(up)` to a 0–100 score using historical distribution anchors.

```bash
cd webapp && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Score mapping (from expanding pathway distribution):
- Score 0 → P(up) = 0.221 (absolute min observed)
- Score 50 → P(up) = 0.500 (neutral)
- Score 100 → P(up) = 0.663 (absolute max observed)

---

## Pathway Merge (`scripts/merge_pathways.py`)

Joins expanding and rolling prediction parquets on `(ticker, end_date)` so each stock-date has both ensembles' probabilities, ranks, and deciles.

```bash
python scripts/merge_pathways.py \
    --expanding runs/expanding/<ts> \
    --rolling   runs/rolling/<ts> \
    --out-dir   runs/comparison
```

Outputs:
- `pathway_comparison.parquet` — one row per `(ticker, end_date)` with both pathways' `p_up_mean`, `p_up_std`, rank, decile
- `pathway_comparison_summary.csv` — per-date correlation, rank correlation, decile disagreement rate

---

## Inference (`scripts/infer_fullperiod.py`)

Re-scores saved ensembles over any date range. Loads all 5 state_dicts, stacks each layer's weights into a grouped convolution for a single forward pass, and runs on MPS/CPU. Used to re-evaluate old checkpoints with updated architecture params.

---

## Cost Analysis (`scripts/ibkr_triple_tier_costs.py`)

Connects to Interactive Brokers TWS/Gateway via `ib_insync` to pull live bid/ask, short-availability, and 30-day historical spread data for the triple-tier-filter universe. Re-prices the monthly portfolios with observed IBKR spreads instead of the Corwin-Schultz proxy.

**Prerequisites:** TWS or IB Gateway running, API enabled, `ib_insync` installed. Requires a Python 3.14 compatibility shim for `ib_insync` event-loop autocreate.
