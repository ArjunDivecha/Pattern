---
type: "Reference"
title: "Kill-or-Confirm Review (Addendum L)"
description: "The Addendum L research arc: leakage-guard audit, unified signal harness, baseline gauntlet, factor neutralization, pre-registered GBDT label bake-off, Dixon geometry vetting, and the rlcc close-location signal that failed out-of-sample replication. Revises the headline conclusions in Domain & Strategy."
openwiki:
  roles: [domain, workflow, testing]
  change_kinds: [research, lifecycle]
  source_paths: [PREREGISTRATION.md, scripts/evaluate_signal.py, scripts/baseline_gauntlet.py, scripts/neutralize_p_up.py, scripts/gbdt_label_bakeoff.py, scripts/rlcc_sweep.py, scripts/rlcc_international.py, scripts/residual_geometry/build_signals.py]
  test_paths: [tests/test_purge_guard.py, tests/test_strawman_frozen.py]
  invariants: ["Production purge gap is exactly window+horizon (40 trading days) for all split modes.", "Every ranking signal is scored by evaluate_signal.py with simple returns and a $5 price floor as the headline.", "The Stage-2 strawman baseline carries a checksum and headline numbers frozen in tests/test_strawman_frozen.py.", "The full sample 1999-2026 is reported but explicitly NOT usable for promotion; only 2005+ is decisive."]
  validation_commands: ["pytest tests/test_purge_guard.py tests/test_strawman_frozen.py -v"]
---

# Kill-or-Confirm Review (Addendum L)

Addendum L (README §L.1–L.13) is the project's largest research continuation.
After the J/K corrections, rather than reflexively tuning hyperparameters, it
stepped back and asked two questions that had never been asked: **is the
out-of-sample machinery actually sound**, and **does the CNN know anything a
free signal doesn't?** Answers: yes, and — as deployed — no.

This page revises the headline conclusions in [Domain & Strategy](domain.md).
The pipeline and the J/K corrections stand; what changes is the *case for
spending further compute* and the *magnitude* of the deployed signal. The
research methodology (leakage guard, one measuring stick, factor
neutralization) is now sound and reusable — the effect being harvested is
what decayed.

## When to consult this page

- Before re-running training, tuning hyperparameters, or proposing a new
  signal variant on this panel — the pre-registered verdict (§L.9) is **no
  promotion**, and the lockbox is spent.
<!-- openwiki: broken internal link [workflows.md#signal-evaluation-scriptsevaluate_signalpy] heading anchor "signal-evaluation-scriptsevaluate_signalpy" does not exist in "workflows.md". Fix the href or restore the target, then delete this comment. -->
- When adding a new ranking signal — score it with the [unified harness](workflows.md#signal-evaluation-scriptsevaluate_signalpy), not a bespoke backtest, so numbers stay comparable.
- When touching split/purge logic — the gap is frozen at production width by
<!-- openwiki: broken internal link [testing.md#tests-test_purge_guardpy--leakage-guard] heading anchor "tests-test_purge_guardpy--leakage-guard" does not exist in "testing.md". Fix the href or restore the target, then delete this comment. -->
  [tests/test_purge_guard.py](testing.md#tests-test_purge_guardpy--leakage-guard).

---

## The Gates

| Gate | Section | Question | Verdict |
|---|---|---|---|
| G0 | [L.1](#l1--the-leakage-guard-is-sound) | Is the OOS machinery sound? | **PASS** — no look-ahead; the problem was return arithmetic, never leakage |
| G1 | [L.3](#l3--the-baseline-gauntlet) | Does the CNN beat free signals? | **FAIL as deployed** — raw `p_up` +1.68%/yr (t 1.05); beaten by 5-day reversal. Residual passes (t 3.01). |
| — | [L.4–L.5](#l4l5--factor-neutralization) | Is the orthogonal content real? | **Full sample: yes** (FM t 3.07→3.40). **Since 2020: no** (+0.54% t 0.25 fully neutralized). |
| — | [L.6](#l6--everything-happened-before-2005) | When did it work? | **1999–2004.** Every signal is ~flat 2005–2019. |
| — | [L.7](#l7--two-negative-diagnostics) | Rolling? Inverse-vol weighting? | Both negative. More data beats recent data. |
| — | [L.8–L.11](#l8l11--dixon-geometry-and-the-rlcc-signal) | Dixon geometry / `rlcc`? | `rlcc` survives at the fair (t+1-open) entry: +6.18% net, t 3.41, 2005+. |
| — | [L.9](#l9--stage-2-label-bake-off-no-promotion) | Does a corrected label beat the strawman? | **NO PROMOTION** — no label beats seed noise; half the edge is bid-ask bounce. |
| — | [L.12](#l12--rlcc-parameter-sweep) | Is `rlcc` a needle or a plateau? | Plateau (`n_last ≤ 2`, `lookback = 60`); month-end concentration (6.5×). |
| — | [L.13](#l13--international-replication-fails) | Does `rlcc` replicate out of sample? | **FAILS** — 1 of 3 markets; month-end effect inverts abroad. |

---

## L.1 — The Leakage Guard Is Sound

`tests/test_splits.py` only ever exercised `debug_split` at the default
`purge_days=0` on a single-ticker monthly fixture, asserting naive set
non-overlap. The purge that underwrites every published number — all of it
from `expanding`/`rolling` — was untested.

<!-- openwiki: broken internal link [testing.md#tests-test_purge_guardpy--leakage-guard] heading anchor "tests-test_purge_guardpy--leakage-guard" does not exist in "testing.md". Fix the href or restore the target, then delete this comment. -->
[`tests/test_purge_guard.py`](testing.md#tests-test_purge_guardpy--leakage-guard)
now freezes it. The invariant: a sample at `t` carries an image over
`[t-window+1, t]` and a label over `(t, t+horizon]`, so a train sample at `t`
and a later sample at `v` require `v − t ≥ window + horizon = 40` trading days,
else the training label period physically overlaps the evaluation image. At
the production `purge_days = 39` (= `window + horizon − 1`, set in
`pattern/cli.py:132`), the gap is **exactly 40 trading days** — the tight bound
— for all three split modes. A teeth-check asserts the invariant is
**violated** at `purge_days=0`, so the suite cannot silently go green.

**No leakage exists.** The out-of-sample claim was always sound; the problem
was the return arithmetic ([Addendum K](domain.md#k--the-return-convention-mean-log--portfolio-return)), never look-ahead.

Related fix: `.gitignore` carried a bare `data/`, which matched a directory at
any depth and excluded the source package `pattern/data/` (including
`splits.py` and `loader.py`) from every commit since the first. The two most
correctness-critical modules in the repo had never been under version
control.

## L.2 — One Measuring Stick

<!-- openwiki: broken internal link [workflows.md#signal-evaluation-scriptsevaluate_signalpy] heading anchor "signal-evaluation-scriptsevaluate_signalpy" does not exist in "workflows.md". Fix the href or restore the target, then delete this comment. -->
[`scripts/evaluate_signal.py`](workflows.md#signal-evaluation-scriptsevaluate_signalpy)
scores **any** ranking signal identically: portfolio (simple-return)
arithmetic as the headline, the mean-log number kept only as a labelled
cross-reference, `$5` floor, deciles + 50/50, NW t, turnover, sub-periods.
Prior to this, each backtest script computed returns its own way — which is
how the 7× [reporting error](domain.md#return-convention-and-price-floor-corrections-addenda-jk) survived.

Two rules it enforces, both learned the hard way:
1. **Portfolio arithmetic is the headline.** An equal-weighted book earns the
   mean of simple returns; the repo's `forward_return` is a 20-day LOG return,
   and compounding mean-log differences overstates a long/short spread by
   ~`σ²/2` of the short leg (+7.14pp/yr on the trash tier).
2. **A price floor is mandatory.** Without it the simple-return decile book
   is not even computable — months fall below −100% — and the unfloored bottom
   decile compounds at +92.5%/yr (the penny-shell blowup, now quantified).

It also reports a `computable` flag instead of a misleading number. Validated
against four independently-derived pins before use.

## L.3 — The Baseline Gauntlet

The paper names weekly reversal (WSTR) and TREND as its closest competitors,
yet no simple-signal comparison had ever been run on R1000 — every backtest in
this repo ranks by `p_up` and uses momentum/vol only to define buckets.

<!-- openwiki: broken internal link [workflows.md#signal-evaluation-scriptsevaluate_signalpy] heading anchor "signal-evaluation-scriptsevaluate_signalpy" does not exist in "workflows.md". Fix the href or restore the target, then delete this comment. -->
[`scripts/baseline_gauntlet.py`](workflows.md#signal-evaluation-scriptsevaluate_signalpy)
builds, at each month-end: `rev_1m` (STR), `rev_5d` (WSTR), `mom_12_1`,
`vol_60d`, and a `combo` blend — then tests IC, head-to-head, and double-sorts.

**Predictive IC** (Spearman vs 20-day forward return, oriented, 283 months):

| Signal | mean IC | NW t | % months +ve |
|---|---:|---:|---:|
| **rev_5d (WSTR)** | **0.0218** | **3.43** | 54.6 |
| combo z(−STR)+z(−vol) | 0.0172 | 2.08 | 54.6 |
| mom_12_1 | 0.0154 | 1.60 | 57.1 |
| vol_60d | 0.0135 | 1.03 | 50.4 |
| rev_1m (STR) | 0.0125 | 1.79 | 52.1 |
| **p_up (CNN)** | **0.0099** | 2.61 | 57.4 |

The CNN has the **lowest mean IC of the six**. A one-line 5-day reversal has
more than double the predictive correlation of a 5-seed CNN ensemble that cost
65 A100-hours.

**Honest long/short** (simple returns, $5 floor, identical rows, 282 months):

| Signal | Decile CAGR | t | 50/50 CAGR | t | Turnover |
|---|---:|---:|---:|---:|---:|
| **rev_5d (WSTR)** | **+7.87%** | 2.53 | +4.44% | 3.00 | 0.85 |
| **p_up residual** | **+4.19%** | **3.01** | +1.68% | 2.70 | 0.89 |
| **p_up (as deployed)** | **+1.68%** | **1.05** | +0.46% | 0.63 | 0.89 |
| mom_12_1 | +1.47% | 1.05 | +0.47% | 0.50 | 0.28 |
| rev_1m (STR) | −1.46% | 0.25 | +2.09% | 1.25 | 0.84 |

**Verdict: the CNN fails the primary gate, but not for the expected reason.**
As deployed (+1.68%/yr, t 1.05) it is not distinguishable from zero and is
beaten by WSTR by 6.2pp/yr. But it is **not a repackaged reversal** —
cross-sectional rank correlation of `p_up` against every baseline is low
(max |ρ| = 0.18). Regressing `p_up` on the baselines and ranking on the
**residual** gives +4.19%/yr at t 3.01 — the highest t-statistic of anything
tested, with the best Sharpe stability across sub-periods. The residual is
*better* than the raw signal, which is only possible if the raw signal's
factor tilts are actively hurting it (the CNN's low-vol preference is a bet on
the worst-performing signal, −12.2%/yr).

This redirected the work toward **post-processing, not retraining**:
neutralize `p_up` against the baseline factors — which costs no GPU time.

## L.4–L.5 — Factor Neutralization

<!-- openwiki: broken internal link [workflows.md#signal-evaluation-scriptsevaluate_signalpy] heading anchor "signal-evaluation-scriptsevaluate_signalpy" does not exist in "workflows.md". Fix the href or restore the target, then delete this comment. -->
[`scripts/neutralize_p_up.py`](workflows.md#signal-evaluation-scriptsevaluate_signalpy)
asks three questions the portfolio test alone cannot answer:

1. **Fama-MacBeth** — is the incremental content statistically real? Per
   formation date, regress the forward simple return on rank-normalized
   signals; average across months with a Newey-West t.
2. **Which exposure is doing the damage** — neutralize against subsets.
3. **Is it implementable** — the G1 residual used contemporaneous cross-sectional
   betas (no look-ahead: the forward return never enters); `resid_all_causal`
   refits with only months strictly before `t` as a robustness check.

### Fama-MacBeth: the cleanest evidence (L.4)

| Spec | Regressor | Mean coef (%) | NW t |
|---|---|---:|---:|
| univariate | **p_up** | 0.113 | **0.76** |
| multivariate | **p_up** | **0.326** | **3.07** |
| multivariate | rev_5d (WSTR) | 0.646 | 3.47 |
| multivariate | mom_12_1 | 0.562 | 1.51 |
| multivariate | rev_1m (STR) | 0.378 | 1.52 |
| multivariate | vol_60d | −0.562 | −1.09 |

On its own `p_up` does not predict returns (t 0.76). Controlling for the
baselines, its coefficient triples and becomes strongly significant (t 3.07).
This is a textbook suppression effect: the CNN's factor tilts were masking its
own signal. Only two regressors are significant in the multivariate spec —
WSTR and `p_up`.

### Net of costs, and the honest bottom line

Turnover is ~0.89 one-sided monthly for every `p_up` variant. Decile LS, net:

| Signal | Gross | Net @ 8bp | Net @ 25bp | Sharpe | t |
|---|---:|---:|---:|---:|---:|
| **combo: neutralized p_up + WSTR** | **9.48%** | **+7.77%** | +4.14% | 0.76 | 3.70 |
| rev_5d (WSTR) alone | 7.87% | +6.23% | +2.74% | 0.52 | 2.53 |
| p_up neutralized alone | 5.60% | +3.89% | +0.25% | **0.77** | **3.72** |
| **p_up raw (as deployed)** | 1.68% | **−0.02%** | −3.64% | 0.22 | 1.05 |

The equal-weight combination of neutralized `p_up` and WSTR is the best book
full-sample (+7.8%/yr net, t 3.70) and beats either component alone — the two
carry genuinely complementary information. Since 2020, though, adding WSTR
*hurts* (Sharpe 0.81 → 0.57) because WSTR has decayed; the pure neutralized
CNN is the better recent book.

### Extended neutralization (L.5): size + sector

L.4 neutralized against four price-based baselines only. The residual could
still load on size, liquidity or sector. The CNN's sector tilt is the
volatility tilt in disguise (favours Real Estate, Utilities, Financials —
classic low-vol/high-payout). Adding `log(dv_60d)` and 10 BICS-level-1 sector
dummies:

| Spec | p_up coef (%) | NW t |
|---|---:|---:|
| univariate | 0.113 | 0.76 |
| + 4 baselines | 0.326 | 3.07 |
| **+ baselines + size + sector** | **0.330** | **3.40** |

Full sample: robust — adding size and sector *strengthens* `p_up` slightly.
But the recent period does not survive:

| Neutralized against | Full | Since 2015 | **Since 2020** |
|---|---|---|---|
| baselines | 5.60% / t 3.72 | 3.97% / t 1.80 | 5.69% / t 1.63 |
| + size | 5.05% / t 3.50 | 3.10% / t 1.44 | 3.32% / t 1.00 |
| + sector | 5.11% / t 3.85 | 2.71% / t 1.35 | 2.68% / t 0.84 |
| **+ size + sector** | **4.65% / t 3.57** | 2.40% / t 1.20 | **0.54% / t 0.25** |

**Since 2020: it disappears.** Fully neutralized, the signal falls to +0.54%
(Sharpe 0.11, t 0.25) — a large share of the recent apparent performance is
size/sector tilts, not chart-reading. With 63 months the power cannot settle
it either way. This is an argument for paper-tracking the neutralized signal
rather than for deploying or killing it.

## L.6 — Everything Happened Before 2005

<!-- openwiki: broken internal link [workflows.md#signal-evaluation-scriptsevaluate_signalpy] heading anchor "signal-evaluation-scriptsevaluate_signalpy" does not exist in "workflows.md". Fix the href or restore the target, then delete this comment. -->
[`scripts/plot_signal_comparison.py`](workflows.md#signal-evaluation-scriptsevaluate_signalpy)
plots the cumulative books. Two things are immediately visible that no summary
statistic in L.3–L.5 revealed.

**First, raw `p_up` loses money net of costs over 27 years: ×0.99.** That is
the signal the Tracker paper book currently ranks on.

**Second, and far more important: every line goes flat after 2004.**

Decile LS by era, net of 8bp (t-statistics on gross):

| Signal | 1999–2004 | **2005–2019** | 2020–2026 |
|---|---:|---:|---:|
| raw p_up | +4.7% (t 1.51) | **−2.1% (t 0.02)** | +0.5% (t 0.63) |
| WSTR | +21.1% (t 2.58) | **+2.2% (t 1.31)** | +2.1% (t 0.72) |
| neutralized (baselines) | +14.7% (t 3.47) | **−0.3% (t 0.96)** | +3.9% (t 1.81) |
| neutralized (+size+sector) | +14.5% (t 3.66) | **+0.1% (t 1.62)** | −1.2% (t 0.25) |
| combo | +28.0% (t 4.27) | **+2.0% (t 1.52)** | +3.4% (t 1.11) |

**Across the fifteen years from 2005 to 2019 — 180 months, the majority of
the sample — every single signal returns approximately zero net.** Nothing
reaches t = 1.7. The impressive full-sample t-statistics of 3.4–3.9 are almost
entirely produced by a six-year window that ended two decades ago. This is
not a CNN-specific problem — WSTR shows the identical shape (+21% then +2%),
as expected of short-horizon cross-sectional effects that were arbitraged away
after decimalization. This is why [PREREGISTRATION.md](#the-pre-registration-protocol)
makes the full sample explicitly non-promotable.

## L.7 — Two Negative Diagnostics

Run before Stage 2, using only already-trained models — no new fitting, so
neither consumes a pre-registered variant slot. Both scored against the frozen
strawman on the primary 2005+ window.

**Rolling vs expanding retraining.** If the post-2005 decay were a regime
change, recency-weighted training (the `runs/rolling` trailing-5-year pathway)
should help. It does the opposite — FM incremental content of `p_up`:
expanding 0.330 (t 3.40) vs rolling 0.209 (t 1.90). As a portfolio, 2005+ net:
expanding-neutralized +0.88%, rolling-neutralized **−2.57%**. More data beats
recent data, decisively.

**Inverse-volatility weighting.** Equal-weight hands the most risk to the most
volatile names. 1/vol weighting gave a hair of Sharpe on the neutralized CNN
(0.434 → 0.445) and *hurt* the strawman. Neither reaches the pre-registered
bar (alpha t ≥ 2.0 and own net ≥ +2.0%/yr). Not the fix.

## L.8–L.11 — Dixon Geometry and the `rlcc` Signal

An independent run (`scripts/residual_geometry/phase0_20260809`) reported a
free "last-bar geometry" signal, `rlcc_or`, at +13.6% LS / t 5.86 — stronger
than anything found here. `rlcc` measures **where the close sat within its
daily range on the formation date, relative to the prior 19 days**; oriented
to buy stocks that closed low in their range. There is no look-ahead, but the
construction is the classic signature of **bid-ask bounce**.

### The implementation-lag test (L.8)

Compute the signal at the close of day *t*, enter at the close of day *t+1* —
enough to skip the bounce.

| Signal | IC entering at *t* | IC entering at *t+1* | **Retained** |
|---|---:|---:|---:|
| `rlcc_or` | 0.0319 (t 5.91) | 0.0124 (t 2.33) | **38.9%** |
| `geom_score` | 0.0328 (t 5.59) | 0.0200 (t 3.70) | 61.1% |
| `rev_5d` (WSTR) | 0.0215 (t 3.39) | 0.0118 (t 1.83) | 54.6% |
| **`p_up_resid`** | 0.0080 (t 2.76) | 0.0063 (t 2.10) | **79.0%** |
| `p_up_mean` (raw) | 0.0093 (t 2.42) | 0.0076 (t 1.84) | 81.7% |

`rlcc_or` loses 61% of its edge to a single day of delay. Retention rises
monotonically with liquidity, exactly as spread-driven artefacts do. `p_up_resid`
is the **most implementation-robust signal tested** — 79% of its edge survives
a day's delay — small, but not a microstructure illusion.

### Correction (L.10–L.11): the fair entry is the next morning's open

The t+1-*close* lag test discards the whole of day *t+1* — **the exact day a
one-day reversion signal predicts**. It is not a bounce filter; it removes the
effect being measured. The realistic implementation is the **next morning's
open**: the signal is known at the day-*t* close, you buy at the *t+1* open.

### `rlcc_or`, 2005+ (n=219), decile LS, net @8bp

| Entry | All names | Liquid half (top-50% $ volume) |
|---|---|---|
| *t* close (MOC-feasible) | +7.40%, Sharpe 0.94, t 3.82 | +4.69%, t 2.27 |
| **_t+1_ OPEN (realistic)** | **+6.18%, Sharpe 0.82, t 3.41** | **+3.89%, t 2.04** |
| *t+1* close (the flawed test) | +2.20%, Sharpe 0.43, t 1.75 | +1.14%, t 1.14 |

**Entering at the next open retains 84% of the edge and stays significant** —
including in the liquid half. Were this bid-ask bounce, the overnight gap would
have consumed it. It is genuine short-horizon reversion, concentrated in day
*t+1*, and it is capturable. At the fair entry, `rlcc` delivers **+6.18%/yr net
with t 3.41, 2005+** — the strongest implementable signal found anywhere in
this review, and a free piece of OHLC arithmetic that arrived from an outside
run, not from the CNN or GBDT.

This corrects L.10's claim that "on an implementable basis the post-2005 bar
is approximately zero." The t+1-*close* test was mis-specified; the mis-spec ran
through L.8–L.10. But Stage 2's verdict is unchanged: at the fair entry the
best Stage-2 candidate (GBDT L3) reaches alpha t 1.94 against a §5 bar of 2.0.

## L.9 — Stage 2 Label Bake-off: NO PROMOTION

Run under [PREREGISTRATION.md](#the-pre-registration-protocol), committed
before the data was touched again. Platform: xgboost on 21 features covering
the same information as the CNN's 20-day image (price path, daily ranges,
close location, volume texture, MA relationship), on the CNN's exact
expanding-window schedule. Purge audited at 40 trading days in every window.
Early stopping on **validation IC** for all labels.

<!-- openwiki: broken internal link [workflows.md#signal-evaluation-scriptsevaluate_signalpy] heading anchor "signal-evaluation-scriptsevaluate_signalpy" does not exist in "workflows.md". Fix the href or restore the target, then delete this comment. -->
[`scripts/gbdt_label_bakeoff.py`](workflows.md#signal-evaluation-scriptsevaluate_signalpy)
tests four pre-registered labels: L0 (binary sign, the production control),
L1 (market-excess sign), L2 (cross-sectional rank), L3 (factor-neutral rank).

### §5 evaluation — 2005+ (n=219), neutralized ensembles, net @8bp

| Signal | Net | Sharpe | t | Alpha vs strawman | alpha t | §5 |
|---|---:|---:|---:|---:|---:|---|
| L0_neut (production label) | +1.75% | 0.485 | 2.17 | +3.82% | 2.22 | no |
| L1_neut | +2.98% | 0.618 | 2.80 | +6.19% | **3.85** | **PASS** |
| L2_neut | +2.98% | 0.587 | 2.55 | +5.86% | 3.12 | **PASS** |
| L3_neut | +3.29% | 0.582 | 2.44 | +5.84% | 2.65 | **PASS** |

Three candidates satisfy §5's letter. Two things stop that being a result.

**It fails §6 (noise floor): no label beats the control.** Best case is
L3_neut − L0_neut = +1.54pp, against a floor of 1.27–2.83pp. **No
label-vs-label difference is distinguishable from seed noise.** Note L0 — the
current production label — already beats the strawman (alpha t 2.22) and beats
the CNN's own neutralized signal (alpha t 1.25, §L.7). Whatever the GBDT
gained, it did not come from the label.

**Roughly half of it is bid-ask bounce.** The implementation-lag test applied
to the bake-off: half the edge dies from one day's delay. The feature set
includes `clv_last` (the `rlcc` construction) and `r_lag1`; re-running without
those improves retention but drops returns below the strawman and reverses the
ordering. The apparent label effect was better exploitation of
microstructure, not better forecasting.

**Verdict: NO PROMOTION.** Per §5's pre-committed rule, Stage 2 stops here. The
lockbox (§8) remains sealed and unspent — there is no winner to spend it on.
Recorded as amendments A1 and A2 in PREREGISTRATION.md.

### What was actually learned

1. **The label was not the problem.** The Stage-1 diagnosis (binary sign trains
   a Sharpe-ranker: forward-vol IC 0.097, t 19.1 vs forward-return IC 0.010,
   t 2.60) is correct — but fixing it does not recover tradable content.
2. **The image representation is not earning its keep.** A GBDT on 21 hand
   features beats the CNN on the identical label (alpha t 2.22 vs 1.25) —
   strong evidence the CNN's vision machinery is unnecessary here.
3. **The CNN residual remains the most implementation-robust signal found**
   (79% retained vs 49–61% for the GBDT). Small, doesn't clear the post-2005
   bar — but alone among everything tested, not primarily a microstructure
   artefact.
4. **Most short-horizon cross-sectional "alpha" on this panel is bid-ask
   bounce.** It shows up in `rlcc_or` (39% retained), WSTR (55%), and the GBDT
   (49–61%). Any future work should treat the implementation-lag test as
   mandatory.

## L.12 — `rlcc` Parameter Sweep

<!-- openwiki: broken internal link [workflows.md#signal-evaluation-scriptsevaluate_signalpy] heading anchor "signal-evaluation-scriptsevaluate_signalpy" does not exist in "workflows.md". Fix the href or restore the target, then delete this comment. -->
[`scripts/rlcc_sweep.py`](workflows.md#signal-evaluation-scriptsevaluate_signalpy)
swept 144 variants before spending the NKY lockbox on `rlcc`. Anti-overfitting
design: every variant reported on a time split (tune 2005–2015, validate
2016–2026), ranked on the *worse* of the two halves. The parameter surface is
a **plateau, not a needle**:

| `n_last` | mean worst-half net | % positive in **both** halves |
|---:|---:|---:|
| 1 | +3.52% | 94% |
| **2** | **+4.62%** | **100%** |
| 3 | +1.69% | 89% |
| 5 | +0.09% | 47% |

| `lookback` | mean | % positive both |
|---:|---:|---:|
| 5 | +1.39% | 69% |
| 10 | +2.46% | 89% |
| 20 | +2.07% | 72% |
| **60** | **+4.01%** | **100%** |

58 of 144 variants (40%) clear +2% net with t > 1.5 in both halves — a fluke
produces one or two cells, not 58. The original construction (`n_last=1`,
`lookback≈20`) ranks 26th of 144, so the published setting was mid-plateau.

### The month-end effect

Building the actual daily overlapping implementation revealed the edge is a
**month-end effect**: 20-day LS by the formation day's distance to month end
shows +0.739% (t 3.72) at month-end, decaying to +0.048% (t 0.66) for days 11+
from month-end — **6.5× at month-end versus all other days**. This was not
selected for — the month-end grid was inherited from the CNN's cadence.
Plausible mechanism: month-end concentrates index rebalancing, window dressing
and fund flows, so flow-driven dislocation peaks there.

The strategy is therefore: **at each month-end, buy the decile that closed
lowest in their daily range relative to a 60-day baseline (2-bar signal), enter
at the next open, hold 20 days.** Twelve rebalances a year, ~89% one-sided
turnover. The L.11 headline (+6.18%/yr net @8bp, t 3.41, 2005+) stands.

## L.13 — International Replication: FAILS

<!-- openwiki: broken internal link [workflows.md#signal-evaluation-scriptsevaluate_signalpy] heading anchor "signal-evaluation-scriptsevaluate_signalpy" does not exist in "workflows.md". Fix the href or restore the target, then delete this comment. -->
Rather than spend the single NKY lockbox shot, [`scripts/rlcc_international.py`](workflows.md#signal-evaluation-scriptsevaluate_signalpy)
tested `rlcc` on **three developed markets never touched by this review**:
Australia (407 names), UK (525), Japan (224). Pre-committed before running —
the R1000-swept parameters transplanted unchanged. Success criterion: positive
with t ≥ 2.0 in at least two of three markets.

| Market | Names | Months | Gross | t | Net @8bp | Pass |
|---|---:|---:|---:|---:|---:|---|
| Australia | 407 | 191 | +8.38% | **2.09** | +6.68% | **YES** |
| UK500 | 525 | 254 | +2.16% | 1.35 | +0.49% | no |
| Japan | 224 | 254 | **−1.81%** | −0.49 | −3.53% | no |

**One of three. The pre-committed verdict is FAILS TO REPLICATE.**

The mechanism inverts abroad — the month-end concentration (6.5× on R1000)
replicates nowhere: Australia 1.35×, UK −0.83×, Japan −0.44×. In Japan the
signal works on ordinary days (t 3.13) and is *negative* at month-end. The
month-end effect is not established as a general phenomenon.

**The lockbox is still sealed and now unspendable on this candidate** — Japan
was one of the three markets tested, so the NKY panel is no longer untouched.
The honest status of the `rlcc` month-end book: **an interesting US result
that failed its first out-of-sample test.**

---

## The Pre-registration Protocol

`PREREGISTRATION.md` (written 2026-08-09, before any Stage-2 experiment was
run) is the binding protocol that prevents forking-paths search. Key rules:

- **§2 frozen strawman.** The baseline to beat is `combo` (neutralized CNN +
  WSTR), decile LS, simple returns, $5 floor, scored by `evaluate_signal.py`.
  Frozen: FULL 1999–2026 net +7.64% (t 3.66); **PRIMARY 2005+ net +2.40%** (t
  1.86). Checksum `sha256(round(series,10))[:16] = 59f55f36169346ca`, 282 months.
- **§3 evaluation windows.** **PRIMARY (promotion decision): 2005-01-01
  onward** (219 months). 2005–2019 and 2020+ reported separately but never
  decisive. **Full sample: reported, explicitly NOT usable for promotion** —
  full-sample statistics re-reward the dot-com era (see [L.6](#l6--everything-happened-before-2005)).
- **§4 primary metric.** Spanning **alpha over the frozen combo** (annualized
  intercept, NW t lag 3) — not raw CAGR. A candidate that merely rediscovers
  reversal will show high CAGR and no alpha.
- **§5 promotion rule (both required, 2005+).** (a) spanning alpha NW t ≥ 2.0,
  **and** (b) own net LS ≥ +2.0%/yr at 8bp. **If no candidate satisfies both,
  the pre-committed outcome is to stop.**
- **§6 noise floor.** Five seeds of identical config are scored first; no
  difference smaller than 2× that cross-seed SD counts as a finding.
- **§7 variant list is fixed here.** Label variants L0–L3 (Stage 2); horizon
  variants H5/H20 (Stage 3, only on the winning label — never run, no winner).

The strawman is enforced in code by
<!-- openwiki: broken internal link [testing.md#tests-test_strawman_frozenpy--strawman-baseline-freeze] heading anchor "tests-test_strawman_frozenpy--strawman-baseline-freeze" does not exist in "testing.md". Fix the href or restore the target, then delete this comment. -->
[`tests/test_strawman_frozen.py`](testing.md#tests-test_strawman_frozenpy--strawman-baseline-freeze):
it checks the checksum, length, and headline net/NW-t numbers, and asserts
`PREREGISTRATION.md` itself is present and contains its key markers. A
pre-registration nobody can verify is a promise, not a control.

---

## How Addendum L Revises the Headline

The [Domain & Strategy](domain.md) headline — +17.4% LS, Sharpe 1.22, NW t
+7.99 — is **arithmetically correct but practically misleading**, because:

1. **As deployed, raw `p_up` is worthless net of costs** (−0.02%/yr at 8bp).
   The Tracker paper book currently ranks on exactly this.
2. **The strong full-sample t-statistics lean on 1999–2004.** Across 2005–2019
   (180 months, the majority of the sample) every signal returns ~zero. The
   last six years are too noisy to call either way.
3. **The CNN carries genuine orthogonal content** (FM t 3.07–3.40, survives
   every control full-sample) that neutralization recovers — but since 2020
   fully neutralized it falls to +0.54% (t 0.25), indistinguishable from a
   size/sector bet.
4. **The strongest implementable signal found is `rlcc`** (+6.18% net, t 3.41,
   2005+, at the t+1-open entry) — a free OHLC arithmetic signal from an
   outside run, not the CNN — and it **failed its first out-of-sample
   replication** (1 of 3 markets).

The methodology is now sound and reusable; the diagnosis (raw `p_up`
contaminated by factor tilts; neutralization genuinely improves it) stands.
What does not survive is the case for spending further compute. The binding
constraint was never the model, the hyperparameters, or the training budget —
it is that the effect this project is trying to harvest largely stopped paying
twenty years ago. The remaining live question (does the neutralized signal do
anything in the next few years?) is answered by paper-tracking it, not by
another backtest or GPU run.

---

## Extension Points and Draft PRDs

Two further research directions are drafted (Phase 0 implemented) but have no
promotable result:

- **Visual Analog kNN** (`PRD_Visual_Analog_KNN.md`, `scripts/analog_knn/`):
  retrieve the *k* historically completed chart windows that look most similar
  in Pattern CNN embedding space, average those neighbors' returns, trade the
  cross-sectional rank. Asks whether *market memory of shape across names*
  predicts returns beyond the CNN's own `p_up` head and beyond reversal. Phase
  0 (extract embeddings, leakage-safe kNN, evaluate) is implemented with a
  leakage property test (`scripts/analog_knn/tests/test_leakage.py`).
- **Residual Pattern + Dixon Geometry + Attention Gate**
  (`PRD_Residual_Geometry_Attention.md`, `scripts/residual_geometry/`):
  harden `p_up` residualized against free baselines, add Dixon geometric
  features as residualization controls, and an FOMC attention gate as a cheap
  timing overlay. Phase 0 (`build_signals.py`, `run_phase0.py`) and a stress
  test (`stress_rlcc.py`) are implemented; `mark_multi_signal.py` publishes
  additional Tracker sleeves.

<!-- openwiki: broken internal link [workflows.md#signal-evaluation-scriptsevaluate_signalpy] heading anchor "signal-evaluation-scriptsevaluate_signalpy" does not exist in "workflows.md". Fix the href or restore the target, then delete this comment. -->
Both depend on the same [unified signal harness](workflows.md#signal-evaluation-scriptsevaluate_signalpy)
and are governed by the same "treat the implementation-lag test as mandatory"
<!-- openwiki: broken internal link [workflows.md#signal-evaluation-scriptsevaluate_signalpy] heading anchor "signal-evaluation-scriptsevaluate_signalpy" does not exist in "workflows.md". Fix the href or restore the target, then delete this comment. -->
discipline established in §L.8–L.11. See [Workflows](workflows.md#signal-evaluation-scriptsevaluate_signalpy)
for their script-level entry points.
