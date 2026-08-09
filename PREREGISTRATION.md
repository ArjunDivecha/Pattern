# Pre-registration — Stage 2: does a corrected training objective beat the strawman?

**Written 2026-08-09, BEFORE any Stage-2 experiment has been run.**
Committed so the promotion rule cannot be revised after seeing results.

This document is binding. Any deviation must be recorded as a dated amendment
at the bottom, with the reason, and any result obtained under a deviation is
labelled exploratory and is **not promotable**.

---

## 1. Why this exists

Stage 0/1 (README addenda L.3–L.6) produced a specific, mechanical finding: the
binary sign label trains the CNN to rank by `μ/σ`, not by `μ`. The forward test
is unambiguous — the CNN's score predicts *forward realized volatility* with
IC 0.097 (t 19.1) but *forward return* with IC 0.010 (t 2.60). It is
overwhelmingly a volatility forecaster deployed as a return signal.

Stage 2 asks whether retraining against a corrected objective recovers content
that the hijacked objective never learned.

The risk this document exists to control: we have already looked at every era
of this dataset dozens of times, across roughly a dozen signal variants. Any
further iteration without a fixed rule is forking-paths search, and would
produce a "winner" whether or not one exists.

## 2. The frozen strawman

The baseline to beat is **`combo` = equal-weight blend of the neutralized CNN
and weekly reversal (WSTR)**, decile long/short, portfolio (simple) returns,
$5 price floor, as scored by `scripts/evaluate_signal.py`.

Frozen values, computed 2026-08-09 and **not to be recomputed or re-tuned**:

| Window | n | Gross | Net @8bp | Sharpe | NW t |
|---|---:|---:|---:|---:|---:|
| FULL 1999–2026 | 282 | +9.48% | +7.64% | 0.763 | 3.66 |
| **PRIMARY 2005+** | **219** | **+4.17%** | **+2.40%** | **0.408** | **1.86** |
| 2005–2019 | 156 | +3.76% | +2.00% | 0.410 | 1.52 |
| 2020+ | 63 | +5.18% | +3.40% | 0.417 | 1.11 |

Turnover 0.8911 one-sided/month → 1.71%/yr drag at an 8bp half-spread.

- Source series: `runs/experiments/stage1b_eval/combo_resid_wstr/signal_card.xlsx`, sheet `floored_monthly`, column `ls_decile`
- Integrity check: `sha256(round(series,10))[:16] = 59f55f36169346ca`, 282 months

If that checksum ever fails to reproduce, the baseline has moved and every
comparison against it is void.

## 3. Evaluation windows

- **PRIMARY (promotion decision): 2005-01-01 onward.** 219 months.
- **Reported always, never decisive:** 2005–2019 and 2020+ separately, to
  confirm a result is not carried by one sub-era.
- **Full sample 1999–2026: reported, explicitly NOT usable for promotion.**
  Addendum L.6 established that essentially all historical performance predates
  2005; full-sample statistics re-reward the dot-com era and are misleading
  about anything deployable.

## 4. Primary metric

For candidate signal `X`:

1. Score `X` with `scripts/evaluate_signal.py` — decile LS, simple returns,
   $5 price floor. No alternative scoring path is admissible.
2. Apply factor neutralization (`scripts/neutralize_p_up.py`, `resid_all`
   specification) to `X` before scoring. Raw scores are known to be
   contaminated; every candidate is judged in its best honest form.
3. Regress `X`'s monthly return on the frozen combo's monthly return over the
   PRIMARY window. **The primary metric is the intercept (alpha), annualized,
   with a Newey-West t (lag 3).**

Alpha over the strawman — not raw CAGR. A candidate that merely rediscovers
reversal will show a high CAGR and no alpha, and must not be promoted.

## 5. Promotion rule (both conditions required)

A candidate is promoted to a full-panel retrain **only if, over 2005+:**

- **(a)** spanning alpha vs the frozen combo has **NW t ≥ 2.0**, and
- **(b)** the candidate's own **net LS ≥ +2.0%/yr at 8bp**.

Rationale for requiring both: (a) alone can be satisfied by a low-return,
low-correlation signal that is not worth trading; (b) alone is satisfied by the
strawman itself (+2.40%) and proves nothing. Note the combo's own alpha-vs-
itself is zero by construction, so (a) is a genuine hurdle, and the combo's
post-2005 t of 1.86 means "beat the combo numerically" would be a sub-t=2 bar —
which is why beating it is explicitly *not* the criterion.

**If no candidate satisfies both, the pre-committed outcome is to stop.** Write
the Investment Learnings verdict, leave the Tracker paper book on the best
available signal, and spend nothing further on this model. Failure is a
permitted and expected result of this protocol.

## 6. Noise floor — measured before any comparison

Five seeds of the identical configuration will be trained and scored first. The
cross-seed standard deviation of the primary metric is the noise floor, and is
recorded here as an amendment before any candidate comparison is interpreted.

**No difference smaller than 2× that standard deviation counts as a finding**,
regardless of its t-statistic.

## 7. The variant list is fixed here

Label variants (Stage 2, GBDT platform):

| ID | Label |
|---|---|
| L0 | binary sign of 20-day forward return (current production; replication control) |
| L1 | binary sign of market-excess forward return |
| L2 | cross-sectional rank of forward return, within formation date |
| L3 | cross-sectional rank of the factor-neutral forward return (residual on vol, size, sector, momentum, reversal) |

Horizon variants (Stage 3, run only on the winning label):

| ID | Horizon |
|---|---|
| H5 | 5-day, overlapping portfolios (form weekly, hold 4 weeks) |
| H10 | 10-day, overlapping |
| H20 | 20-day (current production) |

**Total pre-registered comparisons: 4 labels + 3 horizons = 7.** Every one is
reported, including losers. Any variant conceived after results are seen is
exploratory, is reported as such, and cannot be promoted under §5.

Multiple-testing note: with 7 pre-registered tests, an individual t of 2.0
corresponds to a family-wise error rate well above 5%. The t ≥ 2.0 bar in §5 is
therefore a **screening** threshold; the actual protection against a false
positive is §8, not the t-statistic.

## 8. The lockbox — the only untouched data we own

`data/nky_ohlcv_database.csv` (Japan / Nikkei panel) and
`runs/nky_expanding` have never been evaluated in this review and are hereby
**sealed**.

- No Stage-2 or Stage-3 experiment may read them.
- Exactly **one** candidate — the single winner under §5 — may be evaluated on
  NKY, exactly **once**, after the R1000 decision is final.
- That result is reported whatever it shows. A winner that fails on NKY is
  reported as failing.

This is the only genuinely out-of-sample test available besides waiting for the
future, and it is spent once.

## 9. What would invalidate a result

Recorded in advance, so they cannot be rationalized later:

- Changing the evaluation window after seeing results.
- Promoting a variant conceived post-hoc.
- Switching the primary metric, the scoring script, or the cost assumption.
- Reading NKY before the final single shot.
- Re-tuning the strawman, or its checksum failing to reproduce.
- Reporting a subset of the 7 pre-registered comparisons.

## 10. Honest prior

Stated in advance so the outcome cannot be re-narrated afterwards.

Success is unlikely. Every signal tested in Stage 1 — including plain weekly
reversal, which involves no model at all — went flat after 2005, which points
to a market-structure change rather than a modelling defect. Post-hoc
neutralization already bounds what tilt-removal is worth, and that bound is
approximately zero over 2005–2019 (t 0.96–1.62).

The one fact arguing the ceiling could be higher is the magnitude of the
objective hijack: at IC 0.097 (t 19.1) on forward volatility versus 0.010 on
forward return, essentially the model's entire learned capacity went to the
wrong task. Post-hoc neutralization can subtract a contaminated score's tilt;
it cannot reallocate what was never learned. That is the specific, mechanical
reason a retrain might exceed the neutralization bound — and it is the only one.

Expected outcome: no candidate passes §5, and the project is closed with a
documented verdict. That is a satisfactory result.

---

## Amendments

### A1 — 2026-08-09 — Noise floor recorded (§6), before any comparison interpreted

Five seeds of the identical GBDT configuration, decile LS CAGR over the primary
2005+ window:

| Label | Per-seed CAGRs (2005+) | Mean | SD |
|---|---|---:|---:|
| L0 (control) | 5.42, 1.93, 3.04, 3.50, 4.91 | +3.76% | **1.42** |
| L1 | 5.03, 5.56, 5.96, 3.14, 4.11 | +4.76% | 1.14 |
| L2 | 6.23, 6.40, 6.47, 3.89, 7.01 | +6.00% | 1.22 |
| L3 | 9.35, 5.85, 9.16, 5.85, 7.11 | +7.46% | 1.72 |

**Noise floor = 1.42pp (L0 seed SD). Per §6, nothing below 2× = 2.83pp counts
as a finding.**

Note the floor is measured on single seeds; a 5-seed ensemble has lower
variance, bounded below by 1.42/√5 = 0.63pp if seed errors were independent —
which they are not, since all seeds share data, features and splits. The true
ensemble floor lies between 0.63 and 1.42pp, and is not separately measurable
without many more ensembles. Both bounds are reported wherever the distinction
changes a conclusion.

### A2 — 2026-08-09 — Stage 2 outcome: NO PROMOTION

Recorded per §5's pre-committed stopping rule. Full detail in README addendum
L.9. Summary:

- On the standard evaluation (enter at formation close), several candidates
  satisfy §5's letter: L1_neut (alpha t 3.85), L2_neut (3.12), L3_neut (2.65),
  L3_ens (2.29), L1_ens (2.12).
- **But no label-vs-label difference clears the §6 noise floor.** Best case
  L3_neut − L0_neut = +1.54pp net, against a floor of 1.27–2.83pp. The label
  question is therefore *unresolved*, not answered affirmatively.
- An implementation-lag test (not pre-registered; added after README L.8, and
  strictly *more* stringent than §5) shows 39–51% of the GBDT's edge dies from
  a single day's entry delay. Re-running without same-bar microstructure
  features (`clv_last`, `r_lag1`, `gap_last`) leaves L0 at +2.86% (t 1.87) and
  L3 at +2.16% (t 1.10) — below the strawman, insignificant, and with L3 now
  *worse* than L0.

**Verdict: no candidate is promoted to a full-panel CNN retrain.** The
apparent label effect was better exploitation of bid-ask bounce, not better
forecasting.

The lockbox (§8) remains **sealed and unspent** — there is no winner to spend
it on.
