---
type: "Reference"
title: "Domain & Strategy"
description: "Financial strategy and research findings for the Pattern CNN signal: headline results, cross-sectional slicing, trash-tier intersection, regime overlays, trading-cost analysis, liquid-grid pivot, ETF deployment failures, the $5 price-floor and return-convention corrections (Addenda J-K), and the Tracker paper book."
---

# Domain & Strategy

This page covers the financial strategy, key research findings, and deployment conclusions. The research is documented in detail in the README.md addenda (sections A–I) and summarized here.

## The Core Strategy

**I20/R20 configuration:** 20-day OHLCV candlestick chart images (I20) are fed to a 5-CNN ensemble that predicts the sign of the subsequent 20-day forward return (R20). No hand-crafted features — the CNN learns directly from the raw visual structure of price action.

**Portfolio construction:** At each 20-day rebalance date, stocks are scored by ensemble-averaged `P(up)`, ranked cross-sectionally into 10 deciles. D10 (highest P(up)) is the long book, D1 (lowest) is the short book. Long-short = D10 − D1, equal-weight within each decile.

**Training protocol:** Expanding-window retraining anchored at 1996. 28 windows, each training a 5-seed ensemble. Window 0 trains on 1996–98, tests 1999. Window 27 trains on 1996–2025, tests 2026. All out-of-sample predictions are strictly forward-looking.

---

## Headline Results (Expanding Pathway, 1999–2026)

| Metric | Value |
|---|---|
| Observations | 9,249,453 (ticker × end_date) |
| Unique tickers / dates | 3,011 / 5,759 |
| Overall test AUC | 0.5068 |
| Windows with positive LS | 25 / 28 |

### Long-Short Summary

| Portfolio | Cum × | Ann. Return | Ann. Vol | Sharpe | NW t-stat |
|---|---:|---:|---:|---:|---:|
| **D10 − D1** | 39.4× | **+17.44%** | 14.07% | **1.22** | **+7.99** |
| Top 30% − Bot 30% | 7.3× | +9.07% | 8.66% | 1.05 | +6.60 |

The monotonic decile spread (D1: −8.7%/yr, D10: +10.3%/yr) confirms a persistent cross-sectional signal. The strategy performs particularly well in high-volatility regimes (2000–02, 2008, 2021) and is resilient across market crises. See `strategy_memo.md` for the full year-by-year table.

---

## Cross-Sectional Slicing — Where Does Alpha Concentrate?

After the main run, a series of slicing experiments answered: *where inside the R1000 universe does the signal concentrate?* Scripts: `scripts/backtest_by_sector.py`, `backtest_by_mcap_proxy.py`, `backtest_by_feature.py`, `backtest_generic.py`.

### Key Findings

| Slice | Finding |
|---|---|
| **Size (dollar-volume tertiles)** | LS monotonically stronger in smallest $-volume tertile — the signal is largely a small-cap phenomenon |
| **Realized volatility** | High-vol tertile has both highest gross LS and highest rebalancing frequency |
| **Momentum (12-1)** | Bottom-momentum tertile (recent losers) has richest LS — CNN exploits short-horizon reversal inside loser names |
| **Prediction disagreement (p_up_std)** | Cross-seed dispersion does not improve LS — ensemble disagreement is not a useful signal filter |
| **Sectors** | Three weak sectors: Utilities, Real Estate, Industrials (zero or negative LS). Most other BICS-1 sectors are positive and significant. However, removing the three weak sectors improves CAGR marginally but hurts Sharpe — they are noise-dampeners, not alpha-dilutors. |

Source: README.md Addendum A.

---

## Trash-Tier Intersection Strategy

**Hypothesis:** If alpha concentrates in small, high-vol, recent-loser names, stacking those filters should concentrate the signal further.

**Filter stack:**
- Small (lowest dollar-volume tertile)
- High-vol (highest realized-vol tertile)
- Recent-loser (lowest 12-1 momentum tertile)
- **Triple intersection:** small ∩ high-vol ∩ recent-loser → ~124 names/month, ~62 per side

### Triple-Filter Results (Gross, No Costs, No Price Floor)

> **These numbers are contaminated by sub-penny bankruptcy shells.** A `-Q` suffix ticker at a $0.30 close has a one-tick spread of ~333 bp; the long/short edge pays the bid-ask bounce entirely away, and a one-cent move is a +4,800% return that inflates the backtest roughly 2×. The corrected, deployable figures are in [Return-Convention and Price-Floor Corrections (Addenda J–K)](#return-convention-and-price-floor-corrections-addenda-jk). Source: README.md Addendum J.

| Stat | Universe 50/50 | Triple 50/50 |
|---|---:|---:|
| Months | ~324 | ~317 |
| TOP CAGR | +10.5% | +27.5% |
| BOT CAGR | +4.0% | −6.5% |
| **LS CAGR** | +6.1% | **+28.5%** |
| LS ann-vol | ~6.8% | ~19.2% |
| **Sharpe** | ~0.90 | **~1.07** |
| **NW t** | +4.5 | **+5.05** |

One-sided monthly turnover: ~60% per side (annualized ~720% two-sided). Average holding period: ~0.8 months. This is a high-turnover intra-month reversal strategy, not a buy-and-hold anomaly.

2018 is the single problem year (−33% LS) — concentrated in December 2018 when the Fed pivoted dovish and low-quality short-interest stocks rocketed. The triple filter is long-loser / short-winner, so that short squeeze hit hardest exactly where the model is most exposed.

Scripts: `scripts/backtest_trash_tier.py`, `scripts/trash_tier_turnover.py`, `scripts/trash_tier_yearly.py`. Source: README.md Addendum B.

---

## Regime Overlays — Can We Time Aggressiveness?

Two hypotheses tested:

1. **Small-cap relative momentum** — Maybe LS fails when small-caps outrun large-caps. Computed rolling small-minus-large relative-momentum factor. Result: ρ = +0.36 (opposite sign). The bad months in 2018 are a squeeze event, not a systematic regime. **Rejected.**

2. **Short-term reversal overlay** — Shrink positions after the strategy gets hot (1-month and 3-month LS momentum). Every overlay version reduced CAGR and Sharpe. **Rejected.**

Conclusion: Simple regime overlays don't help. The unconditional strategy dominates. Source: README.md Addendum D.

---

## Trading-Cost Analysis — Is the Gross Number Tradable?

### Corwin-Schultz Spread Estimate

Mean CS spread on triple-filter portfolio: ~208 bps (likely overstated 2–3× for small-cap high-vol names). Break-even half-spread: ~198 bps. A realistic half-spread of 40–50 bps puts net LS around +10–15%.

### $5M-Per-Side Capacity Test

Applied Almgren-style impact model with a 10%-ADV cap and $5M AUM per side:
- Universe after liquidity filter: 29/28 names (down from ~62 per side) — 75% of alpha-richest names drop out (ADV < $5M)
- Gross LS after filter: +13.9% (from +28.5%)
- Cost drag: ~37%/yr
- **Net LS: ≈ −21.5%**

**Conclusion:** The triple-filter strategy does not survive $5M per side. Capacity is estimated at $500k–$1M per side.

Scripts: `scripts/ibkr_triple_tier_costs.py`. Source: README.md Addendum E–F.

---

## Liquid-Grid Pivot — The Tradable Cell

**Question:** Is there an R1000 cell with modest gross return but genuinely cheap execution that survives realistic costs?

`scripts/backtest_liquid_grid.py` buckets every month-end by tertile on dollar-volume (`dv_60d`), realized volatility (`vol_60d`), and momentum (`mom_12_1`), then builds 50/50 LS inside each cell. Cost drag uses realistic half-spreads: Low-dv: 25 bps, Mid-dv: 8 bps, High-dv: 2.5 bps.

### Tradable Cells (Net > 0 AND t ≥ 2, No Price Floor)

> **Pre-price-floor figures.** The cells below are before the Addendum-J $5 price floor; re-filtered results with the floor are in [Return-Convention and Price-Floor Corrections (Addenda J–K)](#return-convention-and-price-floor-corrections-addenda-jk). The mid-cap high-vol loser "workhorse" cell (Mid-dv × High-vol × Low-mom) recommended here does **not** survive the price-floor re-filter in the since-2020 regime.

| Cell | Names | Net LS | Sharpe | NW t | Half-spread |
|---|---:|---:|---:|---:|---:|
| Low × High × Low | 124 | **23.42%** | 1.04 | 5.05 | 25 bps |
| **Mid × High × Low** | 71 | **13.01%** | 0.79 | 3.86 | 8 bps |
| Mid × High × High | 51 | 9.01% | 0.57 | 2.76 | 8 bps |
| **High × High × Low** | 59 | 6.85% | 0.42 | 2.02 | 2.5 bps |
| Mid × Low × Low | 37 | 4.34% | 0.46 | 2.22 | 8 bps |

**Key insight:** High-vol is the load-bearing filter. Mom=Low (recent loser) stacks on top. The effect persists out of the cheap-to-trade zone for the first time.

**Production recommendation:** Mid-cell + High-cell ≈ 130 names per side, gross ~10%, Sharpe ~0.7, deep capacity. The mid-cap high-vol loser cell (Mid-dv × High-vol × Low-mom, +13% net, 8 bps spread, easy borrow) is the workhorse — modest return for a single-name strategy but it clears realistic costs.

Source: README.md Addendum H.

---

## ETF Deployment Attempts — Why They Failed

The author works at an investment firm with restrictions on trading individual stocks but is permitted to trade ETFs. Four attempts were made to deploy the CNN signal through ETFs:

### 1. Zero-Shot on Liquid ETFs (477 ETFs)

The CNN alpha is a **single-stock idiosyncratic-reversal effect**. ETFs are diversified baskets — the source of the signal is averaged away. OOS AUC = 0.508 (trace of signal, not economically useful). All backtest variants (50/50, top/bot 10%, top/bot 20 names, triple-filter) are flat or negative.

### 2. Single-Window Retrain on ETFs (w11)

Both train-from-scratch and fine-tune-from-R1000 **underperformed** the zero-shot baseline. Retraining does not help — the ETF universe lacks the cross-sectional dispersion the CNN can exploit.

### 3. Single-Stock ETFs (SSEs) as Trading Instruments

SSEs are ETFs that track a single underlying stock. Long/short can be built entirely from long positions in leveraged ETFs (2x bull for long, inverse for short — no actual shorting needed).

**Result:** The CNN signal is **inverted** on the SSE universe. Underlying LS: −3.64%/mo, t = −2.14. The CNN learned a 20-day mean-reversion pattern from R1000 1999–2022. The SSE universe is dominated by mega-cap momentum names (TSLA, NVDA, MSTR, COIN, PLTR, Mag 7) where momentum persists. The signal's "oversold buy" calls land on names that keep falling; "overbought sell" calls land on names that keep ripping.

### 4. Alternative Signals on SSE Universe

Tested six classic cross-sectional signals (mom_12_1, mom_6_1, mom_3_1, rev_1m, low_vol, trend). Every underlying-only LS is negative. The best wrapper results come from long-only strategies, but these are driven by the leveraged ETF beta, not signal quality.

**Bottom line:** The CNN alpha is a single-stock, small-cap, high-vol, recent-loser cross-sectional effect. ETFs — basket or single-stock — either average it away or carry it on the wrong tail of the distribution. The signal is universe-specific, not regime-broken (R1000 signal still works in the same 2022–2026 window).

Source: README.md Addendum I. Scripts: `scripts/backtest_sse.py`, `scripts/backtest_sse_momentum.py`, `scripts/finetune_etf_w11.py`.

---

## Return-Convention and Price-Floor Corrections (Addenda J–K)

Two corrections issued in 2026-08 supersede the headline CAGR figures in every
prior section, including the trash-tier and liquid-grid tables above. Both are
documented in README.md addenda J (penny-stock contamination) and K (return
convention) and implemented in the backtest scripts' `--min-price` flag and the
[`mark_book.py`](workflows.md#tracker-paper-book-scriptsmark_bookpy) daily marker.

The signal itself is **not** refuted — the decile monotonicity, the AUC, and the
sign and significance of the spread all stand. What changes is the *magnitude*:
the deployable gross is materially lower, and the trash tier is not obviously a
viable standalone book once costs are applied.

### J — Penny-Stock Contamination and the $5 Price Floor

Sub-penny bankruptcy shells (`-Q` suffix tickers, e.g. DZSIQ, SPWRQ) sort into the
small/high-vol/loser cell by construction and contribute +4,800% one-tick
returns that inflated the published backtest roughly 2×.

**Fix:** Both `scripts/backtest_trash_tier.py` and `scripts/backtest_liquid_grid.py`
gained a `--min-price` flag; `--min-price 5.0` is the deployable setting (drops
~7.1% of rows: 453,002 → 421,007), `0` reproduces the old numbers. The [Tracker
paper book](workflows.md#tracker-paper-book-scriptsmark_bookpy) applies the floor
(`MIN_PRICE = 5.0`) at selection so the tracked series is the deployable one.

**Winsorizing vs price-flooring — different diagnoses.** Winsorizing at ±100%
barely moved the result (+23.5% → +21.0%); a $5 floor roughly halved it
(+23.5% → +12.8%). The driver is a broad population of low-priced names whose
returns are *real* but untradeable — a **capacity finding**, not a data-cleaning
one, that compounds the §E/§G cost conclusion rather than replacing it.

### Trash tier (Low-dv × High-vol × Low-mom), full sample 1999–2026

| Variant | Gross CAGR | Sharpe | NW t | Mean names |
|---|---:|---:|---:|---:|
| As published (no floor) | +23.5% | 1.04 | 5.04 | 124 |
| Winsorize returns ±100% | +21.0% | 1.18 | 5.72 | 124 |
| **Price ≥ $5** | **+12.8%** | **0.76** | **3.71** | **90** |
| Price ≥ $5 + winsorize | +10.1% | 0.79 | 3.82 | 90 |

Same cell, since 2020: +43.9% → **+26.7%** (Sharpe 1.14, t 2.64) at the $5 floor.

### Liquid grid, price ≥ $5, cells passing net > 0 and t ≥ 2

Full sample (282 months):

| Cell | Names | Gross | Net | Sharpe | t | Half-spread |
|---|---:|---:|---:|---:|---:|---:|
| Low × High × Low (trash) | 90 | +12.8% | **+8.9%** | 0.76 | 3.71 | 25 bp |
| Mid × High × High | 54 | +9.3% | +7.9% | 0.55 | 2.66 | 8 bp |
| Mid × High × Low | 70 | +8.4% | +7.1% | 0.55 | 2.67 | 8 bp |
| Low × High (marginal) | 175 | +9.8% | +6.2% | 0.92 | 4.46 | 25 bp |
| Mid × High (marginal) | 151 | +6.7% | +5.5% | 0.66 | 3.22 | 8 bp |
| All R1000 | 1,451 | +2.8% | +1.9% | 0.61 | 2.97 | 8 bp |

Since 2020 (63 months):

| Cell | Names | Gross | Net | Sharpe | t | Half-spread |
|---|---:|---:|---:|---:|---:|---:|
| Low × High × Low (trash) | 101 | +26.7% | **+22.9%** | 1.14 | 2.64 | 25 bp |
| High × High × High | 52 | +16.1% | +15.7% | 0.93 | 2.14 | 2.5 bp |
| Low × High (marginal) | 178 | +14.4% | +10.8% | 1.26 | 2.91 | 25 bp |
| **High × High (mega-cap, high-vol)** | 126 | +10.8% | **+10.5%** | **1.23** | 2.84 | **2.5 bp** |
| All R1000 | 1,360 | +3.9% | +2.9% | 0.97 | 2.24 | 8 bp |

**Mid × High × Low — the "workhorse" cell recommended in [§H.2](#liquid-grid-pivot--the-tradable-cell)
— does not appear in the since-2020 tradable list at all**, consistent with the
regime finding recorded there. Its place is taken by the **High-dv × High-vol**
mega-cap cell (+10.5% net, 2.5 bp spread, Sharpe 1.23, deepest capacity), the
one cell that could clear the author's trading constraints — but it rests on
only 63 months and did **not** work full-sample, so it is a lead to test, not a
result.

Source: README.md Addendum J. Scripts: `scripts/backtest_trash_tier.py`,
`scripts/backtest_liquid_grid.py` (`--min-price 5.0`).

### K — The Return Convention: Mean Log ≠ Portfolio Return

This is the larger of the two corrections. `forward_return` in the predictions
file is a **20-trading-day LOG return** (verified empirically: median
`|log(P_t+20/P_t) − forward_return| = 0.00000`). Every LS number this project
published is therefore a difference of **mean log returns** across names, not an
equal-weighted portfolio return.

An equal-weighted portfolio earns the mean of **simple** returns. By Jensen,
`mean(log) < log(1 + mean(simple))`, with the gap ≈ `σ²/2` where σ is the
*cross-sectional* dispersion of returns within the leg. For a long/short spread
the short leg is subtracted, so **its dispersion penalty is added to the
reported spread** — the inflation is largest exactly where dispersion is
largest (the trash tier, bottom-leg cross-sectional SD ≈ 0.178 per 20-day
period → σ²/2 ≈ 1.58%/month of phantom return).

### Measured (trash tier, $5 floor, 282 months)

| Convention | Mean/month | CAGR |
|---|---:|---:|
| Mean log returns (as published) | +0.892% | **+11.29%** |
| Simple returns (a portfolio) | +0.425% | **+4.15%** |
| **Difference** | | **+7.14 pp** |

On the broad R1000 universe the gap shrinks to ~1 pp (+5.10% → +4.12%) because
cross-sectional dispersion is far lower. The convention inflates concentrated,
high-dispersion cells and barely touches diversified ones.

A related consequence: with **no price floor** the simple convention produces
months worse than −100% (shorting a name that 10×'s loses 900%), so the naive
unhedged decile book is not a computable portfolio at all. The log convention
bounded that tail and hid it; the $5 floor is what makes the tracked series
well-defined.

### Stacking the corrections (trash tier, full sample)

| Trash tier, full sample | CAGR |
|---|---:|
| As originally published | +28.5% |
| After the $5 price floor (addendum J) | +12.8% |
| After portfolio arithmetic (addendum K) | **+4.2%** |
| Less ~25 bp half-spread turnover drag | **≈ 0%** |

That last line is the honest bottom line for the trash tier as a standalone
book. The [`mark_book.py`](workflows.md#tracker-paper-book-scriptsmark_bookpy) daily
marker independently reproduces the +4.15% analytic figure (+4.24% including
the daily-marking calendar), which is the cross-check that the *implementation*
is right and the *convention* was the problem.

Source: README.md Addendum K.

---

## Tracker Paper Book

`scripts/mark_book.py` publishes the Pattern CNN long/short book as a daily NAV
series that the external **Tracker** hub reads. Tracker is read-only — Pattern
publishes, Tracker consumes; if a Pattern row shows STALE on the hub, the fix
lives in Pattern's [daily job](workflows.md#daily-publishing-job-scriptsrun_dailysh),
not in Tracker. Full integration memo: `MEMO-TRACKER-INTEGRATION.md`.

### Two Sleeves

Two sleeves are published because they answer different questions and paper
tracking is free:

| Sleeve | Cell | Since-2020 net | Why track it |
|---|---|---:|---|
| `pattern-trash` | Low-dv × High-vol × Low-mom | +22.9% (t 2.64) | the actual alpha; capacity $500k–$1M, 25bp spreads, hard borrow |
| `pattern-megacap` | High-dv × High-vol | +10.5% (t 2.84) | 2.5bp spreads, Sharpe 1.23, deepest capacity — the cell that could clear the author's constraints |

Both are 50/50 dollar-neutral long/short inside a cross-sectional cell, marked
daily but rebalanced on the 20-day cycle (names held fixed within a period so
intra-period performance is visible). Returns are **gross of costs** per house
convention; the book needs single-stock shorting, so both are `mode: paper`.

### Honest book arithmetic

The marker deliberately uses real portfolio arithmetic — actual prices, held
shares, **simple** returns — not the mean-log statistic in the published
backtests (see [Return-Convention Corrections](#k--the-return-convention-mean-log--portfolio-return)).
Leg return is `mean_j(P_j(t) / P_j(rebal) − 1)`; `LS(t) = leg_long − leg_short`;
`NAV(t) = NAV(rebal) × (1 + LS(t))`. Benchmark is **cash**, not the cell's
equal-weight return: these are dollar-neutral books, so excess against a
long-only cell is a beta statement, not alpha (`ew_cell` is carried in the file
as context only).

### Backtest/Paper split

The `track` column in `nav_daily.csv` records `backtest` before `paper_start`
and `paper` after, and `meta.json` carries `paper_start`. The backtest leg has
already been revised twice (addenda J and K) and must not quietly become the
track record — any page rendering this series must keep the two legs
distinguishable. This is why the daily job publishes a [verify step](workflows.md#daily-publishing-job-scriptsrun_dailysh)
keyed on proof of the action, not an exit code.

Source: `scripts/mark_book.py`, `README.md` §K.5, `MEMO-TRACKER-INTEGRATION.md`.

---

## Summary of Key Takeaways

1. **The paper replicates cleanly.** Full R1000 LS: +17.4%, Sharpe 1.22, NW t +7.99 over 1999–2026 (mean-log convention; see correction 8).
2. **Alpha concentrates in small, high-vol, recent-loser names.** Triple-filter gross *as originally published*: +28.5%, Sharpe 1.07, ~124 names — **contaminated by sub-penny shells; see correction 8.**
3. **Alpha is high-turnover.** ~60% monthly rotation per side. It's an intra-month reversal exploit.
4. **Simple regime overlays don't help.** 2018 is a squeeze event, not a regime feature.
5. **Costs are the binding constraint for the trash tier.** Capacity ~$500k–$1M per side.
6. **The liquid-grid pivot finds tradable cells.** Pre-floor workhorse: Mid-dv × High-vol × Low-mom, +13% net, 8 bps spread, deep capacity. After the $5 floor the since-2020 tradable lead shifts to High-dv × High-vol (mega-cap, +10.5% net, 2.5 bp).
7. **ETF deployment doesn't work.** The signal is single-stock idiosyncratic; ETFs average it away or invert it.
8. **Two corrections halve the headline (Addenda J–K).** A $5 price floor (J) cuts the trash tier +23.5% → +12.8%; portfolio arithmetic (K) cuts it further to **+4.2% gross**, and ~25 bp half-spread turnover drag leaves it **≈ 0% net**. The signal stays real (NW t 2.6–3.8 across every filtered variant); the magnitude was about double, and the trash tier is not obviously a standalone book. See [Return-Convention and Price-Floor Corrections](#return-convention-and-price-floor-corrections-addenda-jk) and the [Tracker paper book](#tracker-paper-book).
