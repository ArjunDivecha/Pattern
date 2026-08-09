# Product Requirements Document (PRD)
## Visual Analog Trading — Cross-Sectional Chart kNN (“Who does this chart look like?”)

**Project codename:** `analog-knn`  
**Parent system:** Pattern (Jiang–Kelly–Xiu chart-CNN pipeline)  
**Status:** Draft v1.0 — ready for implementation  
**Date:** 2026-08-09  
**Owner:** Arjun Divecha  

**Related:**
- Pattern PRD: `PRD.md` (I20/R20 chart CNN)
- Expanding run: `runs/expanding/20260419_174908_cdef6809/`
- Signal harness: `scripts/evaluate_signal.py`
- Baseline gauntlet: `scripts/baseline_gauntlet.py`
- Model API: `pattern/models/cnn.py` → `forward_with_features()`

---

## 0. One-line summary

For every stock-chart window, retrieve the *k* historically completed chart windows of *other* stocks that look most similar in Pattern CNN embedding space, average those neighbors’ subsequent returns, and trade the cross-sectional rank of that analog forecast — asking whether **market memory of shape across names** predicts returns beyond the CNN’s own `p_up` head and beyond simple reversal.

---

## 1. Goal

### 1.1 Primary goal

Build an **end-to-end research system** that:

1. Extracts **penultimate-layer embeddings** from existing Pattern expanding-window CNN ensembles (no retrain).
2. Builds a **leakage-safe memory bank** of (ticker, formation_date, embedding, forward_return).
3. For each query \((i,t)\), finds **k nearest neighbors** with \(j \neq i\) and completed outcomes.
4. Produces an **analog return forecast** \(\hat{r}_{i,t}\) and related diagnostics.
5. Evaluates \(\hat{r}\) with the **same measuring stick** as Pattern (`evaluate_signal.py`: simple-return portfolios, $5 floor, NW t, sub-periods).

### 1.2 Success definition (falsifiable)

The project **succeeds** if any of the following hold on the Full sample after $5 price floor:

| ID | Criterion |
|----|-----------|
| **S1** | Month-end cross-sectional IC of `analog_r_mean` vs 20-day forward return has Newey–West \(t > 2\). |
| **S2** | Decile or top3−bot3 long–short on `analog_r_mean` has NW \(t > 2\) on simple returns. |
| **S3** | Residual of `analog_r_mean` on `{p_up_mean, rev_5d, mom_12_1, vol_60d}` still has IC \(t > 2\) **or** L/S \(t > 2\). |
| **S4** | Absolute rank correlation of `analog_r_mean` with `p_up_mean` **and** with `rev_5d` are both \(< 0.5\) *if* S1 or S2 holds (signal is not a rewrite of known factors). |

**Minimum ship gate:** S1 **or** S2.  
**Feature-only ship gate:** S3 even if raw book is weak (usable as multi-signal input).  
**Kill:** neither S1 nor S2 nor S3 after Phase 0 + one residual-embedding ablation (Phase 1a).

### 1.3 Non-goals (v1)

- Training a new CNN, ViT, or Kronos finetune for embeddings.
- Daily (every trading day) kNN at full R1000 scale as first deliverable.
- LLM / Grok / VLM embeddings of chart screenshots.
- Futures / FX / crypto transfer.
- Live trading or IBKR integration.
- Discrete candlestick-rule matching (H&S, doji, etc.).
- Cost optimization as a kill criterion (house rule: report turnover; do not gate on costs).

---

## 2. Background and motivation

### 2.1 What Pattern already does

Pattern trains 5-seed CNN ensembles on 64×60 I20 candlestick images and outputs `p_up_mean` = ensemble P(next 20-day return > 0). That is **own-image → classifier head**.

Known facts from the Pattern research archive:

- Raw L/S skill exists but is capacity-bound and weak after honest portfolio arithmetic / $5 floor.
- Baseline gauntlet: `rev_5d` (WSTR) has higher mean IC than raw `p_up`; residual of `p_up` on baselines is more useful than raw `p_up`.
- Dixon–Zeng (2024): much of chart-CNN behavior is recoverable from geometric features (level, slope, curvature, last-bar close vs range) and short-term reversal.

### 2.2 What this project adds

| | Pattern `p_up` | Visual analog kNN |
|--|----------------|-------------------|
| Representation | Own chart → head | Own chart → **embedding** |
| Prediction | Softmax classification | **Empirical mean of other stocks’ realized outcomes** |
| Information used | Training-set loss surface | **Historical lookalike cloud** |
| Failure mode | Head collapses to geometry/WSTR | Retrieval of “similar junk” / size-vol clusters |

**Economic claim:** visual configurations reappear across names and years; averaging completed outcomes of lookalikes is a different estimator than the CNN head, and may carry residual information after neutralizing known factors.

### 2.3 Why no new market data is required

All Phase 0–1 inputs already exist under Pattern (see §4). The only “missing” artifact is **embeddings on disk**, produced by inference through existing checkpoints — not a data purchase.

---

## 3. Scope and phases

### Phase 0 — Proof of concept (must complete first)

**Cadence:** month-end formation only.  
**Universe:** R1000 historical members as in Pattern monthly predictions.  
**Embedding:** Pattern I20 GAP (256-d), ensemble-mean of 5 seeds, L2-normalized.  
**k:** 50 (primary); also report 25 and 100.  
**Aggregator:** mean neighbor 20-day forward return (`analog_r_mean`); also median and hit-rate.  
**Eval:** `evaluate_signal.py` + correlation table vs `p_up`, `rev_5d`, residual.

**Exit:** Go / Kill / Feature-only per §1.2.

### Phase 1 — Robustness (only if Phase 0 does not hard-kill)

| Phase | Item |
|-------|------|
| **1a** | Residual embedding \(e^\perp\): strip size/vol/rev_5d/sector before kNN |
| **1b** | Size-tercile-matched neighbors only |
| **1c** | Distance-weighted mean; novelty gate (trade only if mean distance < cross-sectional p50) |
| **1d** | Dixon geometry residualization of embedding (optional) |
| **1e** | Own-history kNN control (neighbors with \(j=i\) only) to prove cross-ticker value |

### Phase 2 — Expression (only if Phase 1 leaves residual skill)

| Phase | Item |
|-------|------|
| **2a** | Disagreement book: \(z(\hat{r}) - z(p\_up)\) |
| **2b** | Non-overlapping 20-day rebalance dates (not only calendar month-end) |
| **2c** | Export monthly factor for multi-signal sleeve / Tracker paper book experiment |
| **2d** | Daily cadence (requires daily image embed; heavy) — optional |

### Explicitly deferred

- Kronos hidden-state embeddings as alternative bank.
- ViT CLS embeddings.
- Multi-horizon neighbor targets (R5/R60).
- Regime-conditioned bank (VIX tercile, etc.).

---

## 4. Inputs (data & models already on disk)

### 4.1 Required (Phase 0)

| Asset | Path | Role |
|-------|------|------|
| Monthly I20 image cache | `cache/r1000_I20_monthly/images.npy` | Chart tensors for embedding |
| Cache index | `cache/r1000_I20_monthly/index.parquet` | `ticker`, `end_date`, `forward_return`, `label` |
| Expanding checkpoints | `runs/expanding/20260419_174908_cdef6809/w{WW}_ensemble_{0-4}.pt` | 28×5 models (~140 files) |
| Expanding config | `runs/expanding/20260419_174908_cdef6809/config.yaml` | Window schedule, image/model hyperparams |
| Monthly predictions | `runs/expanding/.../predictions_monthly.parquet` | `p_up_mean`, `window` id, alignment |
| OHLCV | `data/r1000_ohlcv_spliced.parquet` | Price floor, vol, dollar volume, rev/mom |
| Baseline signals (optional reuse) | `runs/experiments/stage1_baselines/baseline_signals.parquet` | Head-to-head without recompute |

### 4.2 Optional (Phase 1)

| Asset | Path | Role |
|-------|------|------|
| Classifications | `data/r1000_classifications.parquet` | Sector dummies for \(e^\perp\) |
| Daily predictions | `runs/expanding/.../predictions.parquet` | Phase 2d alignment only |

### 4.3 Not required for v1

- New Bloomberg pulls  
- Daily full I20 memmap (only monthly cache exists; sufficient for Phase 0)  
- Precomputed `*_features.npz` (absent today; this project creates the equivalent)

### 4.4 Image / return conventions (inherit Pattern)

- **Image:** I20 geometry as in `PRD.md` §3–4 (64×60, OHLC+MA+volume).
- **Formation date `end_date`:** last trading day of the 20-day window.
- **Forward return:** 20 trading days; log return in Pattern labels; **simple returns for portfolio headlines** via `evaluate_signal.py`.
- **Price floor:** $5 on formation date for all headline metrics.

---

## 5. Embedding specification

### 5.1 Producer (who embeds)

**Not** Grok, not an LLM, not an external API.

**Producer:** frozen Pattern `ChartCNN` weights via:

```text
logits, emb = model.forward_with_features(x)   # emb: (B, 256)
```

- `emb` = global average pool over last conv block spatial dims (`pattern/models/cnn.py`).
- Input `x`: cache images normalized exactly as during Pattern inference (same mean/std / /255 convention as training/predict path — implementer must match `pattern/train/dataset.py` / predict path).

### 5.2 Ensemble aggregation

For each \((i,t)\) and expanding window \(w\) that owns that OOS date:

1. Load seeds \(k=0..4\) for window \(w\).
2. Compute \(e^{(k)}_{i,t}\), L2-normalize each seed vector.
3. \(e_{i,t} = \mathrm{normalize}\big(\frac{1}{5}\sum_k e^{(k)}_{i,t}\big)\)  
   (mean of unit vectors, then re-normalize — default).  
   Ablation: mean then normalize once; or store all 5 and average distances.

### 5.3 Walk-forward model assignment (no future model)

Each row’s embedding must be produced by the **expanding window ensemble whose test period contains `end_date`**, matching the `window` column in `predictions_monthly.parquet`.

**Forbidden:** a single late-period model embedding the entire 1999–2026 history.

**Bank composition at query date \(t\):** only embeddings whose:

- model window’s training end \(\le\) that observation’s formation date (satisfied if we only store OOS embeddings per window as produced historically), and  
- neighbor outcome completed (§6).

### 5.4 Storage schema

`runs/analog_knn/<run_id>/embeddings.parquet` (or memmap + sidecar):

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Bloomberg-style id as in cache |
| `end_date` | datetime64[ns] | Formation date |
| `window` | int16 | Expanding window index 0..27 |
| `emb_0` … `emb_255` | float32 | L2-normalized ensemble-mean embedding |
| `forward_return` | float64 | Pattern log forward return (copy from index) |
| `label` | int8 | Sign label |

Optional companion: `embeddings_meta.json` (git sha, config hash, n_rows, dim, seed rule).

**Size estimate:** ~480k × 256 × 4 ≈ 0.5 GB raw floats + keys — fine on local disk.

---

## 6. Memory bank and leakage rules

### 6.1 Neighbor eligibility for query \((i, t)\)

Include \((j, s)\) in the searchable bank iff **all** of:

| # | Rule | Rationale |
|---|------|-----------|
| L1 | \(j \neq i\) | Cross-ticker analogs only (core hypothesis) |
| L2 | \(s + H \le t\) with \(H = 20\) trading days | Neighbor’s forward return fully known at \(t\) |
| L3 | Embedding for \((j,s)\) from correct walk-forward window | No future-trained representation |
| L4 | `end_date` \(s\) and \(t\) exist in the monthly (or chosen) panel | Align to available rows |

**Implementation of L2:** precompute `outcome_available_date = end_date + 20 trading days` using the trading calendar implied by OHLCV (or approximate with Pattern’s existing forward-return validity: only rows with non-null `forward_return` enter the bank as *completed* observations, and query at \(t\) may only use bank rows with `outcome_available_date <= t`). Prefer **explicit trading-day shift** from OHLCV calendar for clarity.

### 6.2 Explicitly forbidden

- Using ticker \(i\)’s own past as a neighbor in the **primary** signal (own-history is Phase 1e control only).
- Building one FAISS index over all dates with a single model and “filtering dates at query time” without also ensuring model vintage.
- Using neighbor returns that are still open at \(t\).

### 6.3 Recommended soft filters (Phase 0 default off; Phase 1 on)

| Filter | Default Phase 0 | Phase 1 |
|--------|-----------------|---------|
| Same GICS sector exclusion | Off | Optional ablation |
| Size-tercile match | Off | 1b on |
| Min price $5 on neighbor formation | On (for bank used in trading eval of query; neighbors may keep unfloored outcomes for IC diagnostics — document choice) | On |

**Decision for v1:** Apply $5 floor only on the **query/trading universe** at evaluation time (via `evaluate_signal.py`). Bank may include sub-$5 historical neighbors (they are history). Document in run notes.

---

## 7. kNN query and forecast construction

### 7.1 Distance

\[
\delta\big((i,t),(j,s)\big) = 1 - \cos(e_{i,t}, e_{j,s}) = 1 - e_{i,t}^\top e_{j,s}
\]

with unit embeddings → cosine distance via inner product (FAISS `IndexFlatIP` on normalized vectors, convert IP to distance).

### 7.2 Neighbor count

- Primary: \(k = 50\)
- Grid: \(k \in \{25, 50, 100\}\)
- If bank size \(< k\) for early dates: use all eligible; record `k_eff`.

### 7.3 Aggregators (all stored)

| Score column | Definition |
|--------------|------------|
| `analog_r_mean` | Arithmetic mean of neighbors’ **log** `forward_return` (IC-friendly; same units as Pattern label) |
| `analog_r_med` | Median of neighbor log forwards |
| `analog_hit` | Fraction of neighbors with `forward_return > 0` |
| `analog_dist_mean` | Mean cosine distance to neighbors |
| `analog_dist_k` | Distance to \(k\)-th neighbor |
| `analog_k_eff` | Actual neighbors used |
| `analog_minus_pup` | Cross-sectional z(`analog_r_mean`) − z(`p_up_mean`) within date (Phase 2a; can compute post-hoc) |

**Primary trading score for Phase 0:** `analog_r_mean` with `--higher-is-better`.

**Note on log vs simple:** Neighbor aggregation uses Pattern’s stored log forward returns for IC and for consistency with the label the CNN was trained against. Portfolio evaluation converts via `evaluate_signal.py` (simple arithmetic on the query book). Do not invent a second return convention inside kNN without documenting it.

### 7.4 Distance-weighted variant (Phase 1c)

\[
w_m = \exp(-\delta_m / \tau),\quad \tau = \mathrm{median}\{\delta_1,\ldots,\delta_k\}
\]
\[
\hat{r} = \sum_m w_m r_m / \sum_m w_m
\]

### 7.5 Computational approach

| Phase | Index | Scale |
|-------|-------|--------|
| 0 | Exact IP (FAISS Flat or brute numpy) | ~480k vectors; per month-end query ~2–3k names against bank growing to hundreds of k |
| 1+ | FAISS IVF/HNSW if needed | Only if latency hurts research loop |

**Bank growth strategy (Phase 0):**  
For each query date \(t\) in chronological order, bank = all completed eligible rows with outcome available ≤ \(t\). Rebuild or append-only index by date chunks to avoid O(N²) naive all-pairs.

**Optimization allowed:** pre-sort embeddings by `end_date`; maintain rolling FAISS index adding rows when their outcome becomes available.

---

## 8. Residual embedding (Phase 1a) — specification

### 8.1 Purpose

Test whether kNN retrieves pure **shape** or proxies for size/vol/reversal clusters.

### 8.2 Construction (per formation date \(t\), cross-section of bank+query available that day)

Let \(e\) be 256-d embedding. Fit (on bank rows available at \(t\), or on all names present at \(t\)):

\[
e = B z + \varepsilon,\quad
z = \big[z(\mathrm{dv}),\, z(\mathrm{vol}_{60d}),\, z(\mathrm{rev}_{5d}),\, \mathbf{1}_{\mathrm{sector}}\big]
\]

where regressions are **per embedding dimension** (256 OLS) or a single multivariate projection via ridge.  
Store \(e^\perp = e - \hat{B}z\), then L2-normalize \(e^\perp\).

Run kNN in \(e^\perp\) space → scores `analog_r_mean_residemb`.

### 8.3 Size-matched neighbors (Phase 1b)

Assign query to size tercile by 60d dollar volume within date; neighbors restricted to same tercile.

---

## 9. Evaluation requirements

### 9.1 Mandatory measuring stick

All headline results **must** go through:

```bash
python3 scripts/evaluate_signal.py \
  --signal runs/analog_knn/<run_id>/analog_scores.parquet \
  --score-col analog_r_mean \
  --label "Analog kNN k=50" \
  --out-dir runs/analog_knn/<run_id>/eval_analog_r_mean
```

Harness rules (do not fork):

1. Portfolio headline = equal-weight **simple** returns.  
2. **$5 floor** before ranking.  
3. Gross of costs; report turnover as fact.  
4. Sub-periods: Full, pre-2015, 2015–2019, 2020+.  
5. Decile L/S and 50/50 (top/bottom half) at minimum; top3−bot3 if harness supports or via existing Pattern scripts.

### 9.2 Mandatory comparison table

Produce `runs/analog_knn/<run_id>/comparison.xlsx` with rows:

| Signal |
|--------|
| `p_up_mean` |
| `p_up_resid` (from baseline gauntlet if available) |
| `rev_5d` (oriented: lower past return = buy) |
| `analog_r_mean` (k=25,50,100) |
| `analog_r_med` |
| `analog_hit` |
| `analog_r_mean_residemb` (Phase 1a) |
| own-history kNN control (Phase 1e) |

Columns: mean IC, IC NW t, decile CAGR simple, decile t, 50/50 CAGR, 50/50 t, turnover, max DD.

### 9.3 Correlation / identity tests

Monthly cross-sectional Spearman of scores; time-series of those ρ; mean |ρ| of `analog_r_mean` vs each baseline.  
Double-sort: within `rev_5d` quintiles, tercile spread of analog (and vice versa).

### 9.4 Capacity / structure diagnostics

Reuse Pattern cell definitions where possible (trash = low DV × high vol; megacap; etc. from `evaluate_signal.py` CELLS):

- Analog L/S inside trash vs megacap.  
- If skill is 100% trash-only and residual emb kills IC → document as “retrieval of Pattern trash geometry,” not pure shape memory.

### 9.5 Reporting artifacts

| File | Content |
|------|---------|
| `report.md` | Human summary: go/kill, tables, caveats |
| `signal_card.{xlsx,json}` | Per-signal harness output |
| `comparison.xlsx` | Head-to-head |
| `leakage_checklist.md` | Signed-off L1–L4 verification |
| `config.yaml` | Frozen hyperparameters for the run |

---

## 10. System architecture

### 10.1 Package layout

```text
pattern/                     # existing — reuse only
scripts/analog_knn/
  config.py                  # or yaml schema
  extract_embeddings.py      # checkpoints + images.npy → embeddings
  build_calendar.py          # outcome_available_date from OHLCV
  query_knn.py               # leakage-safe bank + FAISS/numpy kNN → analog_scores
  residual_embedding.py      # Phase 1a
  run_phase0.py              # orchestrates extract → knn → eval → comparison
  tests/
    test_leakage.py          # L1–L4 property tests
    test_embedding_unit_norm.py
    test_no_self_neighbor.py
runs/analog_knn/
  <run_id>/
    config.yaml
    embeddings.parquet (or .npy + index)
    analog_scores.parquet
    eval_*/
    comparison.xlsx
    report.md
    leakage_checklist.md
```

### 10.2 Pipeline DAG

```text
[images.npy + index] + [w*_ensemble_*.pt]
            │
            ▼
    extract_embeddings.py
            │
            ▼
    embeddings.parquet
            │
            ▼
    query_knn.py  (+ OHLCV calendar for L2)
            │
            ▼
    analog_scores.parquet
            │
            ├──────────────────┐
            ▼                  ▼
  evaluate_signal.py    merge baselines → comparison.xlsx
            │
            ▼
         report.md
```

### 10.3 CLI (normative)

```bash
# Phase 0 end-to-end
python3 -m scripts.analog_knn.run_phase0 \
  --expanding-dir runs/expanding/20260419_174908_cdef6809 \
  --image-cache cache/r1000_I20_monthly \
  --ohlcv data/r1000_ohlcv_spliced.parquet \
  --k 50 \
  --out-dir runs/analog_knn/phase0_<timestamp>

# Pieces
python3 -m scripts.analog_knn.extract_embeddings ...
python3 -m scripts.analog_knn.query_knn --k 50 --score-cols mean,med,hit ...
python3 scripts/evaluate_signal.py --signal ... --score-col analog_r_mean ...
```

### 10.4 Dependencies

- Existing Pattern env (torch, numpy, pandas, pyarrow).
- **faiss-cpu** (preferred) or pure numpy for Phase 0 exact search.
- No cloud GPU required for Phase 0 if CPU embedding is acceptable; GPU preferred for extract speed.

### 10.5 Performance budgets (soft)

| Step | Budget |
|------|--------|
| Embedding extract (monthly cache, 5 seeds × 28 windows) | < 8 GPU-hours or < 24 CPU-hours |
| kNN all month-ends | < 2 hours on laptop with FAISS Flat |
| Full Phase 0 eval suite | < 30 minutes |

---

## 11. Configuration schema

`runs/analog_knn/<run_id>/config.yaml` example:

```yaml
project: analog_knn
version: "1.0"
phase: 0

paths:
  expanding_dir: runs/expanding/20260419_174908_cdef6809
  image_cache: cache/r1000_I20_monthly
  ohlcv: data/r1000_ohlcv_spliced.parquet
  predictions_monthly: runs/expanding/20260419_174908_cdef6809/predictions_monthly.parquet
  baselines: runs/experiments/stage1_baselines/baseline_signals.parquet

embedding:
  dim: 256
  ensemble: mean_of_l2_normalized_seeds
  seeds: [0, 1, 2, 3, 4]
  walk_forward: per_window_oos_only

knn:
  k: 50
  k_grid: [25, 50, 100]
  metric: cosine
  exclude_self: true
  horizon_trading_days: 20
  aggregator: [mean, median, hit]

eval:
  min_price: 5.0
  primary_score: analog_r_mean
  higher_is_better: true

phase1:
  residual_embedding: false
  size_match: false
  distance_weighted: false
```

---

## 12. Testing requirements

### 12.1 Unit / property tests (CI-able)

| Test | Asserts |
|------|---------|
| `test_no_self_neighbor` | For random queries, no neighbor has same ticker |
| `test_outcome_completed` | Every neighbor satisfies `outcome_available_date <= query_date` |
| `test_embedding_norm` | ‖e‖₂ ≈ 1 within 1e-4 |
| `test_window_alignment` | embedding window id matches predictions_monthly window for same (ticker, date) |
| `test_score_coverage` | analog_scores row count ≈ monthly panel after joins |
| `test_early_k_eff` | Early dates may have `k_eff < k` without crash |

### 12.2 Leakage checklist (manual + automated)

Before any “this works” claim, `leakage_checklist.md` must record pass/fail on L1–L4 with the command used to verify.

### 12.3 Smoke path

Tiny config: 3 tickers × 24 months, k=5, 1 seed, assert pipeline writes scores and eval JSON without NaN explosion.

---

## 13. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Embedding encodes size/vol not shape | Phase 1a residual emb; size-matched kNN; cell diagnostics |
| Analog ≡ WSTR / p_up | Correlation table + residual-on-baselines test S3/S4 |
| Early-sample empty bank | Start IC/L/S from 2003 or first date with median k_eff ≥ 25 |
| Log/simple confusion | Only harness for headlines; document aggregation units |
| Wrong image normalization vs training | Copy exact transform from Pattern predict path; golden-vector test on 1 checkpoint |
| FAISS approximate error | Phase 0 exact Flat only |
| Overfitting k and aggregators | Pre-register k=50 mean as primary; grid is secondary |

---

## 14. Deliverables checklist

### Phase 0 (required)

- [ ] `extract_embeddings.py` + embeddings artifact  
- [ ] `query_knn.py` + `analog_scores.parquet` for k ∈ {25,50,100}  
- [ ] `evaluate_signal` cards for primary scores  
- [ ] `comparison.xlsx` vs p_up / rev_5d / residual  
- [ ] `report.md` with Go / Kill / Feature-only decision  
- [ ] `leakage_checklist.md` all L1–L4 pass  
- [ ] Tests in `scripts/analog_knn/tests/` green  

### Phase 1 (if not killed)

- [ ] Residual embedding scores + eval  
- [ ] Size-matched scores + eval  
- [ ] Own-history control  
- [ ] Updated report  

### Phase 2 (if residual skill remains)

- [ ] Disagreement signal card  
- [ ] Export for multi-signal / Tracker experiment notes  

---

## 15. Decision rubric (end state)

| Outcome | Condition | Action |
|---------|-----------|--------|
| **KILL** | No S1/S2/S3 after Phase 0 + 1a | Archive under Investment Learnings; do not re-propose without new embedding source |
| **FEATURE** | S3 holds; S1/S2 weak or capacity-bound | Export monthly `analog_r_mean` as feature; do not standalone book |
| **BOOK CANDIDATE** | S1/S2 hold after $5 floor **and** skill not solely trash cell **and** residual emb keeps ≥50% of IC | Consider paper sleeve / multi-signal weight; capacity study before capital |
| **REWRITE OF CNN** | S1 holds but S4 fails (ρ with p_up > 0.7) and residual on p_up dies | Treat as interpretability of p_up, not new alpha |

---

## 16. Open implementation choices (resolve in Phase 0 config, not later)

| Choice | Default for v1 |
|--------|----------------|
| Neighbor return units in aggregator | Log `forward_return` from Pattern index |
| Primary k | 50 |
| Primary aggregator | Mean |
| Index library | faiss-cpu Flat IP; numpy fallback |
| $5 floor on neighbors in bank | Off for bank; On for query eval |
| Sector exclusion | Off |
| Own-ticker neighbors | Off (primary) |
| Month-end definition | Dates present in `predictions_monthly` / monthly cache (must match) |

Any change from defaults requires a new `run_id` and comparison to the default run.

---

## 17. Acceptance criteria for “PRD complete / implementation done”

Implementation is **done for Phase 0** when:

1. A third party can run `run_phase0` with only the paths in §4 and reproduce `comparison.xlsx` within numerical tolerance (IC to 1e-3, CAGR to 0.1pp).  
2. Leakage tests pass.  
3. `report.md` states an explicit decision from §15 with the supporting table.  
4. No GPU training was required; only inference + kNN + eval.

---

## 18. Appendix A — Mapping to Pattern code

| Need | Existing hook |
|------|----------------|
| Embeddings | `ChartCNN.forward_with_features` |
| Predict loop pattern | `pattern/train/loop.py` `predict(..., return_features=True)` |
| Per-window feature dump (historical intent) | `pattern/cli.py` `window_*_features.npz` (not present on disk for expanding run — re-extract) |
| Eval | `scripts/evaluate_signal.py` |
| Baselines | `scripts/baseline_gauntlet.py` |
| Image load | Monthly `images.npy` + `index.parquet` (mirror dataset indexing used for cache) |

## 19. Appendix B — Worked neighbor example (illustrative)

Query: `AAPL` on 2015-06-30, embedding \(e\).  
Eligible neighbor: `XYZ` on 2008-11-28 if (i) `XYZ ≠ AAPL`, (ii) 20 trading days after 2008-11-28 are ≤ 2015-06-30, (iii) embedding of XYZ@2008-11-28 was produced by the expanding model that was OOS for that 2008 date.  
If XYZ is among 50 nearest, its realized 20-day return after 2008-11-28 enters the mean that becomes `analog_r_mean` for AAPL@2015-06-30.

## 20. Appendix C — Relation to deferred ideas

This PRD is **only** visual analog kNN. Related ideas (geometry residual of Dixon features alone; ViT factors; curve-surface images) are **out of scope** unless promoted by a separate PRD. The disagreement book (analog vs p_up) is Phase 2 of *this* project because it reuses the same scores at near-zero cost.

---

## Document control

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-08-09 | Initial complete PRD for Phase 0–2 visual analog kNN |

**End of PRD.**
