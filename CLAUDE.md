# CLAUDE.md — Pattern (operator's manual)

Guidance for a coding agent working in this repo. Global rules (light mode, doc-header
blocks, `file://` result links, FAIL-IS-FAIL) live in `~/CLAUDE.md` and are not repeated here.
Deeper prose docs: `openwiki/quickstart.md` (start there for narrative), `PRD.md` (full spec),
`README.md` (985 lines of results + research addenda A–I), `strategy_memo.md` (2-page summary).

## Purpose
Replication of Jiang, Kelly & Xiu (2023) *"(Re-)Imag(in)ing Price Trends"*. Daily OHLCV → 64×60
grayscale candlestick images → 5-seed CNN ensemble predicting 20-day-forward return direction
(the **I20/R20** config) → cross-sectional decile long-short portfolios. Runs at production scale
on the Russell 1000 (~3,000 tickers, 1996–2026) via expanding-window annual retraining.

## Architecture map (load-bearing files, absolute paths)
Repo root: `/Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern`
- `pattern/cli.py` — entry point (`train` / `backtest`), run-dir creation, per-window train loop, prediction/decile assembly. Sets `purge_days = window + horizon − 1`.
- `pattern/config.py` — Pydantic `Config` tree + `Config.from_yaml()`; all defaults live here.
- `pattern/data/loader.py` — CSV → tidy frame, adjusted returns, `compute_labels`.
- `pattern/data/splits.py` — **leakage-critical.** debug/expanding/rolling splits + trading-day purge (`_purge_tail`, `_purge_test_head`) + `balance_labels`.
- `pattern/imaging/renderer.py` — vectorized per-stock OHLC+MA+volume image generator (pixel-exact geometry).
- `pattern/imaging/cache.py` — memmap `uint8 (N,1,H,W)` image cache + parquet sidecar index + `compute_pixel_stats`.
- `pattern/models/cnn.py`, `pattern/models/blocks.py` — `ChartCNN` builder + Conv→BN→LeakyReLU→MaxPool block.
- `pattern/train/loop.py`, `pattern/train/dataset.py` — ensemble training/early-stop/`predict`; `CachedDataset` over the memmap.
- `pattern/backtest/report.py` / `deciles.py` / `metrics.py` — backtest orchestration, decile portfolios, Sharpe/turnover/Newey-West t-stats.
- `scripts/` — multi-GPU drivers (`run_multi_gpu.py`, `gpu_scheduler.py`), data fetch (`fetch_etf_data.py`, `fetch_sse_data.py`), post-hoc slicing/backtests, live scoring (`score_live.py`).
- `src/build_r1000_database.py`, `src/build_nky_database.py` — Bloomberg-backed OHLCV database builders.
- `webapp/` — FastAPI live-scoring app. `configs/{debug,production,prod_expanding,prod_rolling}.yaml`.

## Commands that work
Use the repo venv (Python 3.14): `/Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/.venv/bin/python`
- `.venv/bin/python -m pattern --help` — **verified** (rc 0); subcommands `train`, `backtest`.
- `.venv/bin/python -m pytest tests/ -q` — **verified** (21 passed, ~0.3 s; pure unit tests, no GPU/data).
- `.venv/bin/python -m pattern train --config configs/debug.yaml` — entry verified via `--help`; full run **(unverified — trains the ensemble; needs the R1000 CSV + builds the image cache; not executed here)**.
- `.venv/bin/python -m pattern backtest --run-dir <run> --config <cfg>` — entry verified via `--help`; **(unverified — needs an existing run's `predictions.parquet`)**.
- `scripts/run_multi_gpu.py`, `scripts/gpu_scheduler.py` — 8×A100 drivers **(unverified — CUDA-only; will not run on this Mac's MPS)**.

## Data locations (all under repo root; all git-ignored)
- Input panels — `data/r1000_ohlcv_database.csv` (+ `.parquet`); also `nky_`, `etf_`, `liquid_etf_`, `industry28_`, `sse_` variants. Columns: `Ticker,Date,Open,High,Low,Close,Volume[,AdjClose,Return,MarketCap]`.
- Classifications — `data/r1000_classifications.parquet` (BICS sectors for slicing).
- Image cache — `cache/` (memmap `uint8` + parquet sidecar; ~25 MB debug, full I20 ≈ 20 GB).
- Run outputs — `runs/<timestamp>_<confighash>/` (per-window `*_predictions.parquet`, `*_features.npz`, `config.yaml`, `git_sha.txt`, `window_stats.csv`, `predictions.parquet`, `report.md`). Named pathways: `runs/expanding`, `runs/rolling`, `runs/final`, `runs/liquid_etf_expanding`, `runs/sse_underlying_expanding`.

## Conventions & gotchas (repo-specific, non-obvious)
- **Purge = leakage guard.** `purge_days = window + horizon − 1` (= 39 for I20/R20). Splits use a *time-based* train/val cut (not random) because same-ticker adjacent windows have ~90% label persistence — a random split leaks. Never set `purge_days=0` outside unit fixtures. See FABLE.md P0: the purge magnitude is currently under-tested.
- **`pin_memory` only on CUDA** (`cli.py:199–203`). On MPS/CPU it is a no-op at best and caused *sporadic non-finite batches* — leave it off.
- **Balance train only.** `balance_labels` 50/50 undersamples train; val/test stay at natural class distribution so metrics are honest.
- **Low AUC is expected.** Test AUC ≈ 0.507; the edge is diffuse across thousands of names. The portfolio Newey-West t (~+8) is the real signal, not per-name accuracy.
- **Python 3.14 fragility.** Jinja2 × Python 3.14 broke the webapp `/` route once (commit `3acb7ad`). Re-check FastAPI/Jinja2 compat when touching `webapp/`.
- **Multi-GPU sharding.** Shards share one `--run-dir`; `window_stats.csv` is append-only with a guarded header; GPU is picked by the caller via `CUDA_VISIBLE_DEVICES=N`.
- **Reproducibility artifacts** (`config.yaml`, `git_sha.txt`, `pip_freeze.txt`) are written once per run dir by the first shard — don't assume every shard writes them.

## Current state
- **Research-complete.** R1000 expanding pathway is the headline: LS D10−D1 Sharpe 1.22, NW t +7.99, +17.4%/yr, 1999–2026, 25/28 test years positive.
- **Deployment question answered (mostly negative).** Alpha concentrates in small-cap, high-vol, recent-loser names that are expensive to trade; capacity ≈ $500k–$1M/side. ETF and single-stock-ETF transfer all failed (README addenda H, I). One tradable R1000 cell survives net of costs (mid-cap × high-vol × recent-loser, ~+13% net CAGR, Sharpe 0.79, t 3.86).
- **Working tree is dirty (uncommitted):** 14 modified tracked files + 1 untracked (`scripts/plot_mid_cell_since2020.py`), ~653 insertions — a doc-header sweep plus real code edits in `backtest/` and `models/cnn.py`. Review and commit before building on top.


## Cross-session messaging

Claude Code sessions can message each other directly. `ListAgents` (or `/list-agents`, `/peers`)
lists reachable sessions; `SendMessage` delivers plain text to one by name. Same-machine delivery
uses a local socket; cross-machine is reply-only via Remote Control. Use it to hand off a finding
to a session working elsewhere instead of relaying it through the user. A message is text only —
never conversation history or files; to share full context, resume the session instead.
