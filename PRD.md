# Product Requirements Document (PRD)
## Stock Chart CNN Training Pipeline — Replication of Jiang, Kelly, Xiu (2023)

**Reference paper:** Jiang, J., Kelly, B., and Xiu, D. (2023). *(Re-)Imag(in)ing Price Trends.* The Journal of Finance, 78(6), 3193–3249.

This pipeline replicates the paper's image-based CNN methodology for forecasting stock return direction. We target the **I20/R20** configuration (20-day OHLC+volume chart images predicting the sign of the subsequent 20-day return). Other horizons (I5/R5, I5/R20, I60/R60, etc.) are out of scope for v1 but the code must be parameterized so they can be enabled via config later.

---

## 1. Goal

Train an ensemble of five CNNs on chart images derived from daily OHLCV data to predict whether the next 20-day return is positive. Use the ensemble-averaged `P(up)` to form decile-sorted long-short portfolios and evaluate out-of-sample performance (Sharpe, turnover, drawdown), mirroring the paper's empirical protocol.

---

## 2. Input Data

- **Source:** single CSV file.
- **Required columns:** `Ticker`, `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
  - Optionally: `AdjClose` / `Return` / `MarketCap` / `Shares`. If `Return` is not provided it will be computed from adjusted close; if `MarketCap` is absent, value-weighted portfolios fall back to equal weighting with a warning.
- **Scope:** ~1,000 US stocks, ~20 years of daily data.
- **Cleaning requirements:**
  - Sorted by `(Ticker, Date)`.
  - Missing OHLC on a given day → column of image pixels left blank (paper §I.A).
  - Stocks with IPO or delisting inside the full window are retained; image generation simply skips days where the required look-back is incomplete (paper excludes IPO/delisted within its data window — we note this as a deviation for pragmatism, configurable via `exclude_lifecycle_boundaries: bool`).

---

## 3. Price Normalization (paper §III.A)

For each rolling window before rendering:

1. Normalize the first day's close to `1.0`.
2. Reconstruct subsequent closes from returns: `p_{t+1} = (1 + r_{t+1}) * p_t`.
3. Scale each day's O/H/L by the ratio `p_t / close_t` so they share the adjusted price scale (i.e., splits/dividends do not create artificial jumps).
4. After the full window is constructed, rescale the vertical axis so the min and max of the OHLC path touch the bottom and top of the OHLC region of the image.
5. Volume is independently scaled so the window's maximum volume equals the top of the volume region.

---

## 4. Image Generation (paper §I.A–B)

### 4.1 Geometry (I20 specification)

- **Width:** `3 × 20 = 60 pixels` (each day = 3 pixel columns: left = open tick, center = High–Low bar, right = close tick).
- **Height:** `64 pixels`, split as:
  - **OHLC region:** top `51` rows (≈ 4/5 of 64).
  - **Gap:** 1 row (blank).
  - **Volume region:** bottom `12` rows (≈ 1/5).
- **Channels:** 1 (grayscale). Pixel values are `{0, 255}` before normalization.
- **Background:** black (`0`). Foreground marks: white (`255`).

For I5 (width 15, height 32) and I60 (width 180, height 96), geometry follows the paper's 3n rule and the 4/5 : 1/5 OHLC/volume split.

### 4.2 Per-day drawing rules

For day `t` of the window (0-indexed columns `3t`, `3t+1`, `3t+2`):

- Column `3t+1` (center): vertical line from the row corresponding to `Low_t` up to the row corresponding to `High_t`.
- Column `3t` (left): single pixel at the row corresponding to `Open_t`.
- Column `3t+2` (right): single pixel at the row corresponding to `Close_t`.
- **Moving average line:** an MA over the same window length as the image (20 days for I20). For each day `t`, plot one pixel in the center column at the row corresponding to `MA20_t`, then connect consecutive MA points with a straight line (Bresenham or `skimage.draw.line`). The MA uses the normalized close series.
- **Volume bar:** in the volume region, column `3t+1`, from the bottom up to the row corresponding to `Volume_t / max_volume_in_window`.

### 4.3 Missing-data handling

- If `High_t` or `Low_t` is missing → leave columns `3t..3t+2` fully black for that day.
- If only `Open_t` or `Close_t` is missing → draw the center H–L bar only; omit the missing tick.
- MA line skips a day if that day's MA value is undefined; consecutive drawn MA points are connected by line segments (no connection across gaps).

### 4.4 Caching

- Generated images are persisted once, keyed by `(ticker, end_date, window, has_ma, has_volume)`.
- **Storage format:** a single memory-mapped `uint8` numpy array per config, shape `(N, 1, H, W)`, with a parquet sidecar indexing `(ticker, end_date, label_h, forward_return, label)`.
- Expected size for I20 with 1k stocks × 20 yrs: ~5M samples × 64×60 bytes ≈ ~20 GB. Cache location configurable.

---

## 5. Labels (paper §II.B)

- For an image ending at date `t`, label `y = 1` if the cumulative log return from `t+1` to `t+H` is strictly positive, else `y = 0`. Here `H = 20` for R20.
- Forward returns are computed from total-return-adjusted prices (splits + dividends). Return column is stored alongside the label so the backtest stage can reuse it.
- **Class balancing (training set only):** after labels are computed for training images, resample (undersample the majority class) so train and validation each have ~50/50 positives/negatives. Test set is left unbalanced (real-world distribution).
- **Pixel normalization:** compute mean and std over all training pixels; apply the same stats to val/test images.

---

## 6. CNN Architecture (paper Fig. 3 + Appendix A, for I20)

Three stacked **building blocks**, each:

```
Conv2d(in_c, out_c, kernel=(5,3), stride=(3,1), padding=(12,1), dilation=(2,1))
 → BatchNorm2d(out_c)
 → LeakyReLU(negative_slope=0.01)
 → MaxPool2d(kernel=(2,1), stride=(2,1))
```

- **Channel progression for I20:** 1 → 64 → 128 → 256.
- **Spatial flow (input 64×60):**
  - Block 1 out: `32 × 60 × 64`
  - Block 2 out: `15 × 60 × 128`
  - Block 3 out: `7 × 60 × 256`
- **Head:**
  - Flatten.
  - Dropout(p = 0.50).
  - Fully-connected linear layer → 2 logits.
  - Softmax yields `[P(down), P(up)]`.

### Block counts per configuration (paper Fig. 3)

| Config | Input H×W | Blocks | Final channels |
|---|---|---|---|
| I5  | 32×15  | 2 | 128 |
| I20 | 64×60  | 3 | 256 |
| I60 | 96×180 | 4 | 512 |

### Initialization & regularization

- **Weights:** Xavier (Glorot) uniform.
- **Biases:** zero.
- **Dropout:** 0.50 on the FC input only (no dropout inside conv blocks — paper §II.B).
- **BatchNorm:** between conv and activation, default momentum.

---

## 7. Training Protocol (paper §II.B)

- **Loss:** binary cross-entropy on softmax outputs:  `L = -y log(ŷ) - (1-y) log(1-ŷ)`.
- **Optimizer:** Adam, `lr = 1e-5`, default betas.
- **Batch size:** 128.
- **Early stopping:** patience = 2 epochs on validation loss; max 100 epochs cap.
- **Ensemble:** train **5 models** per config, each with a different seed. Report averaged probabilities for evaluation (paper §III.A, following Gu, Kelly, Xiu 2020).
- **Hardware:** single GPU (CUDA) preferred; code must fall back to MPS/CPU.
- **Reproducibility:** every run logs the seed, config hash, git SHA, and pip-freeze.

### Train/test splits

Two modes, selectable via config:

**(a) Debug mode — default for bring-up:**
- 3 years train/validation (earliest 3 years of data).
  - Within those 3 years, random 70/30 split by (ticker, date) pairs (paper uses random 70/30).
- 2 years test (immediately following). All remaining years are unused.
- Single fit, no retraining.

**(b) Production mode — expanding-window retraining:**
- Initial training window = first `W_init` years (e.g., 8), 70/30 random train/val split.
- Retrain every `R` years (e.g., 1 or 2) using all available history up to that retrain date.
- Out-of-sample predictions accumulated across retraining epochs form the full test series.
- Rolling-window variant (fixed-length training window) also supported via `split.mode: rolling`.

Config keys: `split.mode ∈ {debug, expanding, rolling}`, `split.train_years`, `split.val_fraction`, `split.test_years`, `split.retrain_every_years`, `split.start_year`.

---

## 8. Inference & Portfolio Backtest (paper §III.B–D)

At each test-sample rebalance date (every 20 trading days for R20):

1. For each eligible stock (≥ 20 days of history ending on the rebalance date), render the image and score it through all 5 ensemble members; average to get `P̂_up`.
2. Cross-sectionally rank stocks and form **decile portfolios** (1 = lowest `P̂_up`, 10 = highest).
3. Construct:
   - **Equal-weighted** decile returns.
   - **Value-weighted** decile returns (weights ∝ `MarketCap` at rebalance date).
   - **Long-short H–L** = decile 10 − decile 1 for both weightings.
4. Holding period = 20 days; no intra-period rebalancing.
5. Report per-decile and H–L:
   - Mean holding-period return, annualized Sharpe (assume 252 trading days / H = 12.6 rebalances per year).
   - Annualized volatility, max drawdown.
   - Monthly turnover (match paper's definition: average absolute change in portfolio weights per month).
   - Significance stars (Newey–West `t`-stats at 1%/5%/10%).
6. Save: per-rebalance predictions, decile membership, return series, and a summary table replicating the paper's Table II column `I20/R20`.

---

## 9. Outputs & Artifacts

Per run (under `runs/<timestamp>_<config_hash>/`):

- `config.yaml` — resolved config.
- `images/` — memmapped image cache (shared across runs when possible; symlinked).
- `models/ensemble_k.pt` for `k = 1..5`.
- `training_log.csv` — per-epoch train/val loss + accuracy per ensemble member.
- `predictions.parquet` — columns `ticker, date, p_up_mean, p_up_std, label, fwd_ret, mktcap`.
- `portfolio/decile_returns.parquet`, `portfolio/summary.csv`, `portfolio/plots/*.png`.
- `report.md` — auto-generated, includes Sharpe table, cumulative-return plot of H–L, decile spread plot, and a side-by-side with paper Table II I20/R20 numbers.

---

## 10. Configuration Schema (YAML)

```yaml
data:
  csv_path: ./data/stocks.csv
  date_format: "%Y-%m-%d"
  min_history_days: 252
  exclude_lifecycle_boundaries: false

image:
  window: 20            # I20
  height: 64
  width: 60             # = 3 * window
  ohlc_height_ratio: 0.8
  include_ma: true
  include_volume: true
  cache_dir: ./cache/images_I20

label:
  horizon: 20           # R20
  balance_train: true

model:
  blocks: 3
  channels: [64, 128, 256]
  conv_kernel: [5, 3]
  conv_stride: [3, 1]
  conv_dilation: [2, 1]
  pool_kernel: [2, 1]
  leaky_slope: 0.01
  fc_dropout: 0.5

train:
  ensemble_size: 5
  batch_size: 128
  lr: 1.0e-5
  optimizer: adam
  max_epochs: 100
  early_stop_patience: 2
  seeds: [0, 1, 2, 3, 4]
  device: auto

split:
  mode: debug           # debug | expanding | rolling
  start_year: null      # null = earliest in CSV
  train_years: 3
  val_fraction: 0.30
  test_years: 2
  retrain_every_years: 1   # production modes only

backtest:
  n_deciles: 10
  weighting: [equal, value]
  holding_period_days: 20
  newey_west_lags: 4

output_dir: ./runs
```

---

## 11. Code Structure (suggested)

```
pattern/
  data/
    loader.py           # CSV → tidy pandas + adjusted returns
    splits.py           # debug / expanding / rolling
  imaging/
    renderer.py         # vectorized OHLC+MA+volume image generator
    cache.py            # memmap + parquet index
  models/
    cnn.py              # parametric CNN builder (I5/I20/I60)
    blocks.py
  train/
    dataset.py          # torch Dataset over memmap
    loop.py             # ensemble training + early stop
  backtest/
    deciles.py
    metrics.py          # Sharpe, turnover, drawdown, NW t-stat
    report.py
  cli.py                # `python -m pattern train …` entry points
  config.py             # pydantic models
tests/
  test_renderer.py      # pixel-exact checks vs paper Fig. IA.2
  test_labeling.py
  test_splits.py
```

---

## 12. Success Criteria

1. **Pixel-exact rendering:** unit tests assert OHLC-bar, MA-line, and volume-bar pixel positions match hand-computed references for at least three stock-date samples.
2. **Training stability:** validation loss decreases monotonically (on average across the 5 seeds) during the first 5 epochs on the debug split.
3. **Out-of-sample predictive power:** on the debug split's 2-year test window, the ensemble achieves:
   - AUC > 0.52 (paper achieves ~0.53–0.55).
   - Decile-1 vs decile-10 mean-return spread > 0 at 10% significance.
4. **Paper replication benchmark (production mode, eventually):** I20/R20 equal-weight H–L annualized Sharpe ≥ 1.0 on 2001+ out-of-sample period (paper: ~2.2 on CRSP universe; some slippage expected on a smaller/different universe).
5. **Reproducibility:** two runs with identical config + seed produce bit-identical predictions.

---

## 13. Phased Delivery

- **Phase 1 (debug, 3/2 split, I20/R20):** data loader, image renderer + pixel tests, single-model training loop, basic predictions dump. Goal: green lights on success criteria 1–3.
- **Phase 2 (ensemble + backtest):** 5-seed ensembling, decile portfolios, Sharpe/turnover/drawdown, paper-style report.
- **Phase 3 (production splits):** expanding-window and rolling-window retraining.
- **Phase 4 (extra configs, optional):** enable I5/R5, I5/R20, I60/R60 via config to probe which config performs best on the user's universe.

---

## 14. Explicit Deviations from the Paper (v1)

| Area | Paper | This PRD v1 | Rationale |
|---|---|---|---|
| Universe | CRSP NYSE/AMEX/NASDAQ 1993–2019 | User-supplied CSV (~1k tickers, ~20 yrs) | Data availability |
| IPO/delist exclusion | Dropped | Kept (configurable) | More training data; minor bias |
| Test window | 2001–2019, no retraining | Debug: 2 yrs no retrain → Prod: rolling/expanding | Faster iteration during bring-up |
| Configurations | All 9 Ix/Ry | I20/R20 only (extensible) | Scope |

---

## 15. Open Questions (to resolve during Phase 1)

- Confirm `MarketCap` availability in the CSV; otherwise VW backtest is deferred.
- Confirm whether `Volume` is share volume or dollar volume; paper uses share volume.
- Choice of Newey–West lags in monthly t-stats (default 4).
