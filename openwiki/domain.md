# Domain & Strategy

This page covers the financial strategy, key research findings, and deployment conclusions. The research is documented in detail in the README.md addenda (sections A–I) and summarized here.

## The Core Strategy

**I20/R20 configuration:** 20-day OHLCV candlestick chart images (I20) are fed to a 5-CNN ensemble that predicts the sign of the subsequent 20-day forward return (R20). No hand-crafted features — the CNN learns directly from the raw visual structure of price action.

**Portfolio construction:** At each 20-day rebalance date, stocks are scored by ensemble-averaged `P(up)`, ranked cross-sectionally into 10 deciles. D10 (highest P(up)) is the long book, D1 (lowest) is the short book. Long-short = D10 − D1, equal-weight within each decile.

**Training protocol:** Expanding-window retraining anchored at 1996. 27 windows, each training a 5-seed ensemble. Window 0 trains on 1996–98, tests 1999. Window 26 trains on 1996–2024, tests 2025. All out-of-sample predictions are strictly forward-looking.

---

## Headline Results (Expanding Pathway, 1999–2025)

| Metric | Value |
|---|---|
| Observations | 9,249,453 (ticker × end_date) |
| Unique tickers / dates | 3,011 / 5,759 |
| Overall test AUC | 0.5068 |
| Windows with positive LS | 25 / 27 |

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

### Triple-Filter Results (Gross, No Costs)

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

### Tradable Cells (Net > 0 AND t ≥ 2)

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

## Summary of Key Takeaways

1. **The paper replicates cleanly.** Full R1000 LS: +17.4%, Sharpe 1.22, NW t +7.99 over 1999–2025.
2. **Alpha concentrates in small, high-vol, recent-loser names.** Triple-filter gross: +28.5%, Sharpe 1.07, ~124 names.
3. **Alpha is high-turnover.** ~60% monthly rotation per side. It's an intra-month reversal exploit.
4. **Simple regime overlays don't help.** 2018 is a squeeze event, not a regime feature.
5. **Costs are the binding constraint for the trash tier.** Capacity ~$500k–$1M per side.
6. **The liquid-grid pivot finds tradable cells.** Mid-dv × High-vol × Low-mom: +13% net, 8 bps spread, deep capacity.
7. **ETF deployment doesn't work.** The signal is single-stock idiosyncratic; ETFs average it away or invert it.
