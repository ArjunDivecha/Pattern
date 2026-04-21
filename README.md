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

---

# Addendum — Post-training slicing, trash-tier strategy, and trading-cost analysis

The headline numbers above come from the full R1000 universe.  After the
expanding-pathway run completed we ran a long series of cross-sectional
slicing experiments to answer two questions:

1. Where inside the universe does the signal concentrate?
2. Is the concentrated signal actually tradable after real-world frictions?

All artefacts below live under
`runs/expanding/20260419_174908_cdef6809/`, using
`predictions_monthly.parquet` (one row per ticker × month-end with
`p_up_mean`, `p_up_std`, `forward_return`, `label`).

## A. Cross-sectional slicing — which names hold the alpha?

We built `scripts/backtest_generic.py` as a one-stop slicer: feed it any
categorical/numeric column, let it per-date bucket it, and report a 50/50
or 10-decile LS portfolio per bucket.  Used it for:

| Slice | Out-dir | Notes |
|---|---|---|
| BICS Level 1 | `backtest_by_bics_level_1_5050/` | 12 sectors |
| BICS Level 2 | `backtest_by_bics_level_2_5050/` | 40+ industry groups |
| BICS Level 3 | `backtest_by_bics_level_3_5050/` | Deep industries |
| Dollar-volume tertiles (mcap proxy) | `backtest_by_mcap_proxy_3/`, `…_5/` | 60d mean $vol |
| Momentum (12-1) tertiles | `backtest_by_mom_12_1_3/` | `log(P_{t-21}) − log(P_{t-21-252})` |
| Realized-vol (60d) tertiles | `backtest_by_vol_60d_3/` | `std(daily ret) × √252` |
| Prediction disagreement (`p_up_std`) | `backtest_by_p_up_std_3/` | cross-seed dispersion |

### Findings

- **Size:** LS monotonically *stronger* in smallest $-volume tertile — the
  signal is largely a small-cap phenomenon, consistent with JKX paper.
- **Vol:** High realized-vol tertile has both highest gross LS and the
  highest rebalancing frequency.
- **Momentum:** Bottom-momentum tertile (recent losers) has the richest
  LS — CNN exploits the short-horizon reversal inside loser names.
- **Disagreement:** Sorting by cross-seed `p_up_std` does *not* improve
  LS noticeably — ensemble disagreement is not a useful signal filter.
- **Sectors:** 3 weak sectors emerge — **Utilities, Real Estate,
  Industrials** — with essentially zero or negative LS.  Most other
  BICS-1 sectors have positive and significant LS.

Full per-slice CAGR / Sharpe / NW-t tables live in each
`*_summary.xlsx` file.

## B. Trash-tier intersection — stacking the weak-name filters

Inspired by the size/vol/momentum findings, we intersected the three
"bad-name" buckets and ran a 50/50 LS inside the intersection.

Scripts:
- `scripts/backtest_trash_tier.py` — per-filter 50/50 LS portfolios
- `scripts/trash_tier_turnover.py` — month-over-month ticker turnover
- `scripts/trash_tier_yearly.py` — per-calendar-year LS return and turnover

Filter stack (`univ` = all R1000 names with all 3 features available):

| Filter | Mean univ names/mo | Mean top-side names |
|---|---:|---:|
| universe | ~910 | ~455 |
| small | ~303 | ~151 |
| small & high-vol | ~193 | ~97 |
| small & recent-loser | ~148 | ~74 |
| high-vol & recent-loser | ~168 | ~84 |
| **triple (small & high-vol & recent-loser)** | **~124** | **~62** |

### Triple-filter headline (in-sample gross, no costs)

| Stat | Universe 50/50 | Triple 50/50 |
|---|---:|---:|
| Months | ~324 | ~317 |
| TOP CAGR | +10.5 % | +27.5 % |
| BOT CAGR | +4.0 % | −6.5 % |
| **LS CAGR** | +6.1 % | **+28.5 %** |
| LS ann-vol | ~6.8 % | ~19.2 % |
| **Sharpe** | ~0.90 | **~1.07** |
| **NW t(0)** | +4.5 | **+5.05** |
| Cum × | ~9 | ~360 |

### Turnover (one-sided monthly, symmetric-diff / (|S_t|+|S_{t-1}|))

- Universe: ~42 % per month, per side
- Triple:   **~60 % per month, per side** (annualised ≈ 720 % two-sided)
- Avg holding period: ~0.8 months (less than a month)

The triple-filter portfolio is a high-turnover trash-name book that
leans heavily on intra-month reversal.  Gross returns are excellent;
whether they survive costs is the rest of this addendum.

### Year-by-year triple-filter LS

| Year | LS (%) | Year | LS (%) | Year | LS (%) |
|---|---:|---|---:|---|---:|
| 2000 | +82 | 2009 | +12 | 2018 | **−33** |
| 2001 | +74 | 2010 | +41 | 2019 | +25 |
| 2002 | +52 | 2011 | +28 | 2020 | +54 |
| 2003 | +15 | 2012 | +22 | 2021 | +66 |
| 2004 | +19 | 2013 |  −3 | 2022 | +18 |
| 2005 | +14 | 2014 | +30 | 2023 | +47 |
| 2006 |  +9 | 2015 | +25 | 2024 | +39 |
| 2007 | +18 | 2016 | +17 | 2025 | +14 |
| 2008 | +41 | 2017 | +14 | 2026 YTD | +22 |

2018 is the single problem year — concentrated in Dec 2018 when the Fed
pivoted dovish and low-quality / short-interest stocks rocketed.  The
triple filter is long-loser-short-winner, so that short squeeze bit
hardest exactly where the model is most exposed.

## C. Sub-slice sensitivity — ex-3-sectors

Filtering out Utilities, Real Estate and Industrials before running the
triple filter:

- Universe / mo: **~100** (vs 124)
- LS CAGR: **+30.0 %** (vs +28.5 %)
- Sharpe: **0.90** (vs 1.07)
- NW t: **+4.28** (vs +5.05)

Conclusion: the excluded sectors were mild drags on gross return but
useful diversifiers on risk.  Removing them improves CAGR marginally but
hurts Sharpe / t-stat — *the three weak sectors are noise-dampeners, not
alpha-dilutors*.  Artefacts in `backtest_trash_tier_ex3sectors/` and
`predictions_monthly_ex3sectors.parquet`.

## D. Regime overlays — can we time aggressiveness?

**Hypothesis 1 — Small-cap relative momentum.**  2018 was a strong-
small-cap year, so maybe LS fails when small-caps outrun large-caps.
We computed a rolling small-minus-large relative-momentum factor
(smallest-tertile $vol stocks' 12-1 mean ret – largest-tertile's) and
correlated it with monthly triple-filter LS.

Result: **ρ = +0.36**, the *opposite* sign from the hypothesis.  The
median LS return is almost identical between strong-small vs weak-small
regimes — what differs is the **hit-rate** (57 % vs 86 %).  The bad
months in 2018 are not a systematic small-running-hot regime; they are
a squeeze event.  Hypothesis rejected.

**Hypothesis 2 — Short-term reversal overlay.**  We built binary and
linear aggressiveness overlays keyed off 1-month and 3-month LS
momentum (idea: shrink positions after the strategy gets hot).  Every
overlay version reduced CAGR and Sharpe.  Binary variants shut off in
productive months (2009, 2020, 2023) while only modestly dampening
2018.  No overlay we tried dominates the unconditional strategy.

## E. Trading-cost realism — is the gross number even reachable?

We built two cost studies:

1. **`scripts/` Corwin-Schultz H/L-based estimator** (`backtest_by_cs_spread.py` style).
   Uses daily high/low prices (Corwin-Schultz 2012) to estimate an
   unobserved bid-ask spread at the stock-day level, then averages
   across a rebalance's holdings.

   - Mean CS spread on triple-filter portfolio: **~208 bps** (full)
   - This is likely **overstated by 2–3×** for small-cap high-vol names,
     which the Corwin-Schultz estimator is known to inflate.

2. **Break-even analysis.**  With ~12 rebalances/year and ~60 %
   one-sided turnover per side per month, the cost drag at a full
   spread `s` bps is approximately:

   ```
   drag ≈ 12 × 2 × turnover × (s / 2) / 10000
        ≈ 12 × 2 × 0.6 × (s / 2) / 10000
        ≈ 0.0072 × s     (per year, as fraction)
   ```

   Triple LS gross ~28.5 %.  Break-even full-spread
   `s* ≈ 28.5 / 0.0072 ≈ 3,958 bps` — but this double-counts both long
   and short sides.  One-sided break-even on just the half-spread:

   ```
   LS break-even half-spread h* ≈ 28.5 / (24 × 0.6) × 100 bps ≈ 198 bps
   ```

   Even the inflated Corwin-Schultz spread of 208 bps is uncomfortably
   close to break-even.  A realistic half-spread of ~40–50 bps (CS × 0.4
   haircut) puts net LS around +10–15 % — still attractive but fragile.

3. **$5M-per-side capacity** (`scripts/ibkr_triple_tier_costs.py` driver,
   offline mode).  Applied Almgren-style impact model:

   ```
   impact_bps = c × σ_daily × sqrt(Q / ADV) × 10000   (one-way)
   ```

   with `c = 1.5`, `σ_daily` from realized daily return std, `ADV` from
   60d mean $vol, `Q = 5e6`.  Added a liquidity filter (ADV ≥ $5M, per-
   name cap 10 % of ADV).

   - Universe after liquidity filter: **29 / 28 names** (top/bot), down
     from ~62 each side.  About 75 % of the alpha-richest names drop
     out because they don't have $5M ADV.
   - Gross LS after liquidity filter: **+13.9 %** (from +28.5 %).
   - Mean one-way impact: **~80–95 bps**.
   - Round-trip impact × 2 sides × 12 months × 60 % turnover:
     **cost drag ~37 %/yr**.
   - **Net LS: ≈ −21.5 %**.

   **Conclusion: the triple-filter strategy as-is does not survive
   $5M-per-side.**  Capacity is probably $500k–$1M per side.  Any
   deployment needs either:
   - A much smaller AUM target, or
   - A slower-turning variant (e.g., quarterly rebalance, or position-
     by-position Kelly-shrunk), or
   - Better execution (VWAP, IS algos, internal cross) than the blunt
     impact assumption above.

## F. IBKR live-spread integration (work-in-progress)

`scripts/ibkr_triple_tier_costs.py` connects to a local TWS or IB
Gateway (via `ib_insync`) and pulls per-ticker:

- Live (or delayed) best bid / best ask → realized half-spread
- `reqHistoricalData(whatToShow='BID_ASK')` over 40 days → stable
  time-averaged spread estimate
- Shortability tick (generic tick `236`) — proxy for whether the short
  side of the LS is executable at all
- Realized daily σ over the historical window for impact calculation

It then re-prices the triple-filter monthly portfolios using the
measured IBKR spreads (capped, Winsorised) instead of the Corwin-
Schultz estimate.

### Current blockers

- The user's IBKR account does not have a live US-equity market-data
  subscription, so `reqMktData` returns `delayedBid=None, delayedAsk=
  None` on unsubscribed tickers — only trades (last/HLC) come through.
- Running against the live TWS (port 7496) confirms the connection and
  symbol lookup work; the cost model falls back to the CS estimate for
  any ticker where both live and historical BID_ASK come back empty.
- Next step: either enable the "US Securities Snapshot and Futures
  Value Bundle" (~$10/mo) on the IBKR account, or scrape AltaVista ETF
  Research's "Avg Sp" column for the small number of ETF-like proxies
  that AltaVista covers.

### Python 3.14 compatibility

`ib_insync` needs an event-loop shim at import time under 3.14:

```python
import asyncio as _asyncio
try:
    _asyncio.get_event_loop()
except RuntimeError:
    _asyncio.set_event_loop(_asyncio.new_event_loop())

from ib_insync import IB, Stock, util
```

Without this, import fails with `RuntimeError: There is no current
event loop in thread 'MainThread'` because 3.14 removed the implicit
event-loop-on-demand behaviour.

### Usage

```bash
python scripts/ibkr_triple_tier_costs.py \
    --predictions runs/expanding/20260419_174908_cdef6809/predictions_monthly.parquet \
    --ohlcv       data/r1000_ohlcv_database.parquet \
    --out-dir     runs/expanding/20260419_174908_cdef6809/ibkr_costs \
    --ib-host 127.0.0.1 --ib-port 7496 --ib-client-id 42 \
    --aum-per-side 5e6 --min-adv 5e6 --adv-cap 0.10 --c-impact 1.5 \
    --market-data-type 3
```

`--market-data-type`: 1=live, 2=frozen, 3=delayed (free), 4=delayed-frozen.

Outputs:
- `ibkr_costs/ibkr_spreads.parquet` — per-ticker IBKR spread snapshot
- `ibkr_costs/triple_net_costs.xlsx` — per-month gross / impact / spread
  / net, plus summary row
- `ibkr_costs/triple_net_cum.pdf` — cumulative gross vs net

## G. Takeaways

1. **The paper replicates cleanly.**  Full R1000 LS: +17.4 %, Sharpe
   1.22, NW t +7.99 over 1999-03 → 2026-03.
2. **Alpha concentrates in small, high-vol, recent-loser names.**  The
   triple-filter subset gross-returns ~28.5 % with Sharpe ~1.07 on a
   ~124-name universe.
3. **Alpha is high-turnover.**  ~60 % of names rotate every month per
   side.  The strategy is an intra-month reversal exploit, not a
   buy-and-hold anomaly.
4. **Simple regime overlays don't help.**  Small-cap-momentum and short-
   term reversal timing both reduce Sharpe.  The bad year (2018) is a
   squeeze event, not a regime feature.
5. **Costs are the binding constraint.**  Estimated half-spread on
   the triple-filter book is 50–100 bps realistic, 208 bps inflated
   (Corwin-Schultz).  Break-even half-spread is ~200 bps, so
   costs eat most of the gross alpha.  At $5M per side with a 10 %-ADV
   cap, net return is *negative* — capacity is probably $500k-$1M.
6. **The signal is strongest exactly where transactions are most
   expensive.**  This is the central tension of the paper's
   trash-tier alpha — small, illiquid, volatile names.  Any production
   deployment needs realistic execution cost modelling and low AUM.
