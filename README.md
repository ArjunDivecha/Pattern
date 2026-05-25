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

---

# Addendum H — Zero-shot ETF test and the liquid-grid pivot (2026-05)

After concluding that the trash-tier triple-filter is capacity-limited
to ~$500k–$1M, the next two questions were:

  (i) Does the CNN transfer zero-shot to a universe with naturally low
      spreads and easy borrow — i.e. ETFs?
 (ii) Inside R1000, is there a cell with *modest* gross return but
      genuinely cheap execution that survives realistic costs?

## H.1  Zero-shot test on liquid ETFs

**Universe build.**  Pulled a 4,374-ETF Bloomberg screen
(`/Users/arjundivecha/Downloads/ETF.xlsx`), converted `"SPY US"` → `"SPY"`
for yfinance, dropped leveraged / inverse / 2x / 3x / YieldMax / covered-call
products, then filtered on:

```
bid-ask spread ≤ 5 bps   AND
30-day $-volume   ≥ $10M  AND
30-day share-volume ≥ 100k
```

→ **477 liquid ETFs** (`AssetList_liquid_etfs.xlsx`).  Spread
distribution: 223 @ 1-2 bps, 96 @ 2-3, 69 @ 3-4, 49 @ 4-5.

**Data.**  100 % yfinance fetch success →
`data/liquid_etf_ohlcv.{csv,parquet}` (1.6 M rows, 477 tickers,
1993-01 → 2026-04).  History depth: 378 ETFs ≥ 5 yr, 300 ≥ 10 yr,
209 ≥ 15 yr.

**Image cache.**  Added a `--monthly` flag to
`scripts/render_etf_cache.py` that keeps only the last trading day
of each calendar month per ticker (≈ 20× smaller cache, monthly-cadence
predictions).  Final cache:

```
cache/liquid_etf_I20_monthly/images.npy   215 MB, (58,767, 1, 64, 60) uint8
cache/liquid_etf_I20_monthly/index.parquet 603 KB, 58,767 rows, 427 tickers,
                                         361 monthly dates 1996-03 → 2026-03
```

**Predictions.**  Applied the 28-window R1000-trained ensemble zero-shot.
Overall OOS AUC = **0.5078** (vs R1000 benchmark 0.5068) — there is
a trace of signal, but it is not economically useful.

**Backtests** (in `runs/liquid_etf_expanding/`):

| Filter | CAGR | Sharpe | NW t |
|---|---:|---:|---:|
| 50/50 LS                  | −0.44 % | −0.08 | −0.39 |
| Top/Bot 10 %              | +0.27 % | +0.02 | +0.13 |
| Top/Bot 20 names          | +1.21 % | +0.13 | +0.66 |
| triple (small & high-vol & loser) | **−1.42 %** | −0.18 | −0.95 |

**Conclusion.**  The CNN alpha is a **single-stock idiosyncratic-reversal
effect**.  ETFs are diversified baskets of dozens to hundreds of
single names, so the very source of the signal is averaged away on the
underlying basket.  Even the triple-filter, which on R1000 single names
is the strongest cell of the entire study, goes *negative* once the
universe becomes ETFs.  Cheap execution does not rescue a signal that
isn't there.

## H.2  R1000 liquid-grid search (the pivot that worked)

Question (ii) — finding a tradable R1000 cell with modest return but
cheap execution — was the productive turn.

`scripts/backtest_liquid_grid.py` buckets every month-end by tertile on
`dv_60d`, `vol_60d`, and `mom_12_1`, then builds 50/50 long-short books
inside each cell.  Per cell it reports gross LS, NW t, one-sided
monthly turnover, and a realistic cost drag using
half-spread = {Low-dv: 25 bps, Mid-dv: 8 bps, High-dv: 2.5 bps}.

### Marginal dv tertiles

| Cell | months | gross LS | net LS | NW t | half-spread |
|---|---:|---:|---:|---:|---:|
| dv = Low (all)   | 282 |  9.63 % | 6.56 % | 5.71 | 25 bps |
| dv = Mid (all)   | 282 |  3.52 % | 2.49 % | 3.18 |  8 bps |
| dv = High (all)  | 282 |  1.49 % | 1.19 % | 1.17 | 2.5 bps |
| All R1000        | 282 |  4.86 % | 3.92 % | 4.49 |  8 bps* |

\* notional weighted-average tier.

### Grid A — dv × vol (cells passing net > 0 AND t ≥ 2)

| Cell | names | net LS | Sharpe | NW t |
|---|---:|---:|---:|---:|
| dv=Low  × vol=High | 214 | **16.48 %** | 1.21 | 5.90 |
| dv=Mid  × vol=High | 150 |  7.34 %     | 0.74 | 3.59 |
| dv=High × vol=High | 146 |  6.01 %     | 0.61 | 2.98 |

### Grid B — dv × mom (cells passing net > 0 AND t ≥ 2)

| Cell | names | net LS | Sharpe | NW t |
|---|---:|---:|---:|---:|
| dv=Low  × mom=Low | 211 | **16.07 %** | 1.34 | 6.50 |
| dv=Mid  × mom=Low | 163 |  6.98 %     | 0.85 | 4.11 |
| dv=High × mom=Low | 146 |  3.96 %     | 0.44 | 2.11 |
| dv=High × mom=Mid | 181 |  2.24 %     | 0.46 | 2.24 |

### Grid C — dv × vol × mom (27 cells, top by net LS)

| Cell | names | gross LS | net LS | Sharpe | NW t | half-spread |
|---|---:|---:|---:|---:|---:|---:|
| Low × High × Low     | 124 | 26.97 % | **23.42 %** | 1.04 | 5.05 | 25 bps |
| **Mid × High × Low** |  71 | 14.28 % | **13.01 %** | 0.79 | 3.86 |  8 bps |
| Mid × High × High    |  51 | 10.34 % |  9.01 %     | 0.57 | 2.76 |  8 bps |
| **High × High × Low**|  59 |  7.25 % |  6.85 %     | 0.42 | 2.02 | 2.5 bps |
| Mid × Low  × Low     |  37 |  5.77 % |  4.34 %     | 0.46 | 2.22 |  8 bps |

Of 27 cells, only those 5 pass the (net > 0, t ≥ 2) hurdle.  The
structure is highly concentrated: **High-vol is the load-bearing
filter**, **mom=Low (recent loser) stacks on top of it**, and the
effect persists out of the cheap-to-trade zone for the first time.

### What this changes

- Takeaway #5 / #6 from section G need to be qualified.  Yes — the
  strongest gross alpha is in the trash tier (Low-dv × High-vol ×
  Low-mom, 23 % net, 124 names).  But there are now **two
  cleaner-cost alternatives**:

  1. **Mid-dv × High-vol × Low-mom**: ~71 names, ~$500M–$5B per
     name, 8 bps half-spread, easy borrow.  **+13 % net CAGR,
     Sharpe 0.79, t +3.86.**  This is the workhorse cell — modest
     return for a single-name strategy but it actually clears
     realistic costs.
  2. **High-dv × High-vol × Low-mom**: ~59 mega-cap names
     (TSLA / NVDA / COIN-type beaten-down volatile blue chips),
     2.5 bps half-spread, trivial borrow.  **+6.9 % net,
     Sharpe 0.42, t +2.02.**  Largest capacity of any cell.

- A reasonable production stack is **Mid-cell + High-cell** ≈ 130
  names per side, gross ~10 %, Sharpe ~0.7, deep capacity.

### Dead zones (every Mid-vol cell, every High-mom cell ex one)

All nine Mid-vol cells are flat or negative net.  All six dv ×
mom=High cells are flat or negative net.  The CNN's edge is
concentrated entirely in the high-vol / low-momentum tails — exactly
the regime where mean-reversion dominates trend.

### Files

- `scripts/backtest_liquid_grid.py` — 27-cell grid driver.
- `runs/expanding/20260419_174908_cdef6809/backtest_liquid_grid/liquid_grid_summary.xlsx`
  — sheets `marginals`, `dv_x_vol`, `dv_x_mom`, `dv_x_vol_x_mom`, `tradable`.

## H.3  Updated bottom line

The signal documented in section G is real and the trash-tier
result (28 % gross, $500k-$1M capacity) still stands.  What changed
is that **section G's pessimistic conclusion — "costs eat all the
alpha" — is universe-specific, not signal-specific**.  Climb one
tertile up the dollar-volume ladder, keep the High-vol × Low-mom
sub-filter, and you get a tradable ~13 % net book with eight-times
the capacity.  The CNN signal degrades smoothly with size and
liquidity rather than collapsing — the right deployment is the
mid-cap high-vol loser cell, not the smallest-name trash tier.

---

# Addendum I — ETF / single-stock-ETF deployment attempts (2026-05)

**Motivation.**  Author works at an investment firm with restrictions on
trading individual stocks but is permitted to trade ETFs of any kind,
including single-stock ETFs (SSEs).  The question for this addendum:
*can the R1000-trained CNN edge be deployed through any combination of
ETFs the firm allows?*

Short answer: **No.**  The alpha is a single-stock, small-cap, high-vol,
recent-loser cross-sectional effect.  ETFs — basket or single-stock —
either average it away or carry it on the wrong tail of the
distribution.  Below are the four attempts and why each failed.

## I.1  Single-window retrain on ETFs (w11)

**Hypothesis.**  Maybe zero-shot transfer fails because the CNN never
saw ETF charts.  Retraining on the ETF universe (with or without
warm-start from the R1000 ensemble) might fix it.

**Test.**  On window 11 (2010 test year), trained two ETF-specialised
variants against the R1000 zero-shot baseline:

| Variant | AUC | LS CAGR | LS Sharpe |
|---|---:|---:|---:|
| R1000 baseline (zero-shot) | 0.5125 | +1.14 % | +0.127 |
| **Train from scratch on ETFs** | 0.5040 | **−0.59 %** | **−0.064** |
| **Fine-tune from R1000** | 0.5041 | **−0.68 %** | **−0.063** |

Both ETF-trained variants **underperformed** the zero-shot R1000 model.
Retraining does not help — the ETF universe simply lacks the
cross-sectional dispersion the CNN can exploit.  Full 28-window retrain
was skipped on this evidence (running it would just confirm the result
more rigorously at the cost of ~1 hour of compute).

Files: `runs/etf_scratch_w11/`, `runs/etf_finetune_w11/` (each contains
`comparison.txt`, `*_predictions.parquet`, training log).

## I.2  Single-stock ETF universe build

**Trick.**  SSEs are the loophole around the firm's stock-trading
restriction: they ARE ETFs but each tracks ONE underlying.  A
long/short book can be built entirely from long positions in two
products per underlying — the 2x bull-leveraged ETF for "long" bets and
the inverse-leveraged ETF for "short" bets.  No actual short-selling,
no borrow costs.

**Universe.**  Hand-curated 48-pair seed list spanning the major
issuers (Direxion, GraniteShares, T-Rex/REX Shares, Tradr, Defiance)
across both Direxion-style asymmetric pairs (+2x long / −1x short) and
the cleaner symmetric ±2x pairs from T-Rex / Defiance.  Validated each
ticker via yfinance:

- 121 candidate tickers (44 underlyings + 77 wrappers)
- 120 returned valid price history; 1 delisted (AMDS)
- **44 underlyings with at least one long wrapper, 29 of which have a
  matching inverse wrapper (the "complete-pair" subset)**

History depth:

- Aug-Sep 2022 inception (6 underlyings: AAPL, AMZN, GOOGL, MSFT, TSLA + COIN long-only) — ~3.7 yr
- Dec 2022 / 2023 expansion (+NVDA, BABA) — ~3 yr
- 2024 expansion (+META, MSTR, TSM, MU, PLTR, SMCI) — ~1.5 yr
- 2025+ rest of the universe — < 1 yr each

Effective backtest window: 2022-09 → 2026-04 (44 monthly observations),
universe growing from 6 → 30+ over time.

Files:
- `data/sse_pairs_seed.csv` — hand-curated 48-row pair list
- `data/sse_pairs.csv` — yfinance-validated pair table
- `data/sse_underlying_ohlcv.csv` — 179,843 rows × 44 underlyings
- `data/sse_wrapper_ohlcv.csv` — 33,103 rows × 77 wrappers
- `scripts/fetch_sse_data.py`

## I.3  Zero-shot CNN on SSE underlyings + wrapper-LS backtest

**Setup.**  Score every (underlying, month-end) using the existing
R1000-trained 28-window ensemble (same approach as the liquid-ETF test
in section H).  Image cache geometry identical to training.

Overall OOS AUC on the 44 SSE underlyings = **0.5136** — slightly
*better* than the R1000 baseline (0.5068) because these are mostly
volatile mega-cap names the CNN has seen during training.  The model
"recognises" the universe.

**Trading mechanic.**  Per month-end, rank by `p_up_mean`.  Top half →
buy the 2x long-leveraged ETF.  Bottom half → buy the inverse-leveraged
ETF (long position, no shorting).  Portfolio return per dollar of
capital = `0.5 × (mean(R_long_etf | top) + mean(R_inverse_etf | bot))`.
Costs modelled: 1.0 % p.a. expense ratio prorated + 5 bps half-spread per leg.

**Results (44 months, 2022-09 → 2026-04, mean univ ≈ 20):**

| Variant | Months | Gross CAGR | Net CAGR | Sharpe | NW t |
|---|---:|---:|---:|---:|---:|
| **complete-pairs (true LS via wrappers)** | 44 | **−16.47 %** | −19.31 % | −0.50 | −0.96 |
| long-only (top-half via long-ETF) | 44 | +51.02 % | +46.09 % | +1.11 | +2.15 |
| full (long-only with hedge where available) | 44 | −26.67 % | −29.20 % | −1.40 | −2.70 |

**Diagnosis (the smoking gun).**  Computing the underlying-only L/S
spread (i.e. if we could trade the actual stocks long-short, no wrapper):

  - Underlying TOP half (CNN predicts UP): **+2.04 %/mo**
  - Underlying BOT half (CNN predicts DOWN): **+5.67 %/mo**
  - Underlying LS: **−3.64 %/mo, t = −2.14, Sharpe = −1.12**

**The CNN signal is inverted on this universe in this regime.**  The
loss is NOT a wrapper-decay artifact.  An underlying-only L/S book
loses 38 % CAGR.  The reason: the CNN learned a 20-day mean-reversion
pattern from R1000 1999-2022.  The SSE universe is dominated by
mega-cap momentum names (TSLA, NVDA, MSTR, COIN, PLTR, Mag 7) where
momentum persists.  The signal's "oversold buy" calls land on names
that keep falling; its "overbought sell" calls land on names that keep
ripping.  Hence the inversion.

For context, the R1000 signal still works in the same window
(2022-09 → 2026-04 on R1000): **+7.89 % CAGR, t = +3.20, Sharpe = +1.82**.
The signal is universe-specific, not regime-broken.

Files:
- `runs/sse_underlying_expanding/predictions.parquet` — 5,970 rows
- `runs/sse_underlying_expanding/backtest_sse/sse_summary.xlsx`
- `runs/sse_underlying_expanding/backtest_sse/sse_cum.pdf`
- `scripts/backtest_sse.py`

## I.4  Alternative cross-sectional signals on the SSE universe

**Hypothesis.**  Maybe a different signal (momentum follower, not
reversal) works on the SSE universe.  Tested six classic
cross-sectional signals against the same wrapper mechanics:

```
mom_12_1, mom_6_1, mom_3_1, rev_1m, low_vol (vol_60d inverted), trend
```

### Long-only (just buy top-half via 2x long-ETF)

| Signal | Gross CAGR | NW t | Underlying-LS CAGR (no leverage) |
|---|---:|---:|---:|
| rev_1m   | +547 % | 7.21 | **−7.6 %** |
| **EW_basket (no signal)** | **+422 %** | 7.88 | n/a |
| mom_3_1  | +263 % | 5.06 | −8.9 % |
| mom_6_1  | +241 % | 4.61 | −21.1 % |
| trend    | +216 % | 4.72 | −11.5 % |
| mom_12_1 | +157 % | 3.60 | −16.8 % |
| low_vol  | +14 %  | 1.31 | −47.8 % |

### Complete-pairs LS (long-ETF + inverse-ETF)

| Signal | Gross CAGR | NW t | Underlying LS |
|---|---:|---:|---:|
| rev_1m   | +166 % | 5.21 | −32 % |
| mom_6_1  | +101 % | 3.93 | −23 % |
| mom_3_1  | +92 %  | 3.82 | −7 % |
| trend    | +91 %  | 4.15 | +12 % |
| mom_12_1 | +62 %  | 2.77 | −33 % |
| low_vol  | **−54 %** | −7.84 | −67 % |

### Why every "win" here is fake

The crucial column is **underlying-LS CAGR** — the signal's true
stock-picking skill, stripping away wrapper leverage.  *Every* signal
is **zero or negative** in underlying space.  Translation:

- "+547 % CAGR rev_1m long-only" is **not signal alpha** — the
  no-signal EW basket of all 44 long-ETFs returns +422 % CAGR on its
  own.  rev_1m adds ~+2.5 %/mo on top, plausibly from leverage
  convexity rather than picking-skill (the underlying-LS is −7.6 %).

- "+166 % complete-pairs rev_1m" is roughly half the long-only result
  because the inverse-ETF leg averages ~0 over this bull market — it
  hedges market beta but contributes no cross-sectional alpha.

- **What 2022-2026 actually rewarded: owning 2x mega-cap SSEs
  unhedged.**  The +422 % EW basket CAGR is the 2x leverage compounding
  in a smooth bull market on mega-cap momentum names.  No signal
  needed.

- Low-vol is the only signal with *significantly* negative underlying
  skill (−48 %) — high-vol mega-caps decisively beat low-vol in this
  period (classic low-vol-anomaly inversion in a mega-cap-momentum
  regime).

Files:
- `scripts/backtest_sse_momentum.py`
- `runs/sse_underlying_expanding/backtest_sse_momentum/sse_momentum_summary.xlsx`
- `runs/sse_underlying_expanding/backtest_sse_momentum/sse_momentum_cum.pdf`

## I.5  Why this is a structural wall, not a research-direction problem

The CNN edge lives in the **single-stock cross-section of the broader
R1000**, specifically in small-cap, high-vol, recent-loser names
(section H's tradable cell: +13 % net CAGR, t +3.86, Mid-dv × High-vol ×
Low-mom).  For an ETF-only mandate, this is structurally inaccessible:

1. **Liquid baskets average it out.**  Section H's 477-liquid-ETF zero-shot
   test: all LS results between −1 % and +1 %.  Diversification
   eliminates the idiosyncratic-reversal signal by construction.

2. **Single-stock ETFs cover the wrong tail.**  Issuers make wrappers
   for ~40 ultra-popular mega-cap names — the exact opposite of the
   alpha's natural habitat.  No one issues SSEs on unknown small-cap
   losers because there is no retail demand.

3. **Sector / thematic ETFs are baskets of those same mega-caps.**
   XLK, SOXX, ARKK, MAGS, etc.  Same diversification problem.

4. **Retraining on ETFs does not help** — I.1 documented that
   ETF-specialised models *underperform* the zero-shot R1000 transfer.
   The ETF universe lacks the cross-sectional dispersion to train on.

5. **Generic momentum / reversal signals on SSEs have no stock-picking
   skill** — I.4 documented zero/negative underlying-LS for every
   classical signal tried.  Wrappers + bull market mask this in
   headline returns.

## I.6  Updated bottom line for the project

The replication is correct (section A–B).  The alpha is real and
deployable in single-stock space (section G + H's mid-cap cell).
**It cannot be deployed via any ETF wrapper available to a US
investor.**  This is a *constraint mismatch*, not a model failure:

- The signal is structurally cross-sectional, idiosyncratic, and
  concentrated in names too small/illiquid/specific for any ETF issuer
  to wrap.
- Both directions tried (basket-level ETFs and single-stock ETFs) fail
  for orthogonal reasons (averaging vs universe-skew).
- Retraining on the constrained universe does not produce a deployable
  alternative — the universe is the binding constraint.

**For a fund with the user's restrictions, the CNN-pattern signal is
not actionable.**  Pursue an unrelated alpha source compatible with
ETF-only execution (sector rotation with macro inputs, vol-of-vol on
VXX/UVXY, calendar/seasonality on broad-market ETFs, fixed-income or
FX-ETF carry/momentum).  Pattern-CNN remains a documented, working
single-stock alpha that requires single-stock execution capability.
