# MEMO — Putting Pattern on Tracker's daily board

**To:** Arjun
**From:** the Pattern repo
**Date:** 2026-08-04
**Re:** what I would do to track Pattern daily, and what has to be built first

---

## Bottom line

Tracker is **read-only** — it never runs a strategy's pipeline, it reads what
that strategy already published and flags gaps. So this is almost entirely a
**Pattern-side** job: Pattern has to start publishing a daily NAV. Tracker's
side is one registry edit and one `STRATEGY_PERF` entry.

I'd register it as **`mode: paper`, `cadence: daily`** — a monthly-rebalance
book marked every day, exactly the shape the two T2 sleeves already use.

Roughly a day of work, most of it in Pattern.

## Why now, when the current registry entry says not to

`strategies.yaml` currently says, correctly for when it was written:

> *"Value going forward is p_up as a feature into other sleeves, not a daily
> Tracker book. No daily monitoring required unless score_live is put on a
> schedule."*

Two things changed that argue for revisiting it:

1. **The backtest was found to be inflated ~2×** by sub-penny bankruptcy
   shells (README §J). The corrected trash tier is +12.8% gross full-sample,
   not +23.5%. After a correction that large, a *forward* record is worth
   considerably more than another backtest variant.
2. **The strategy can't be traded yet** — it needs single-stock shorting. That
   is precisely the argument *for* paper-tracking it now: when the constraint
   lifts, the choice is between a genuine multi-year out-of-sample track and a
   backtest that has already been revised once. Starting the clock is free.

So: not a live book, not a research dossier — a **paper book accumulating
honest OOS evidence.**

## What Tracker needs from Pattern

Tracker's contract (`docs/SCHEMA.md`, and the FDT/T2 precedent) is a NAV CSV
in the shape FDT publishes, which is why those books drop into the perf
template with no changes:

| Artifact | Path (in Pattern) | Contents |
|---|---|---|
| Daily NAV | `data/tracker/nav_daily.csv` | `date, nav, ew_cell` |
| Holdings | `data/tracker/holdings_latest.csv` | ticker, side, weight, p_up, entry date |
| Run meta | `data/tracker/meta.json` | `as_of`, n_positions, stale-name count, run status |

**Benchmark = the equal-weight return of the cell's own universe**, carried as
a column in the same file. Same reasoning as FDT reading its own `ew34`
column: identical calendar, identical price source, no join gaps, and the
excess line then isolates CNN selection skill from the cell's beta. House
convention is gross of costs.

Note `scripts/daily_from_monthly.py` is the right *precedent* but **cannot be
reused** — it asserts long-only and raises on negative weights. Pattern is
long/short. Pattern publishes its own NAV instead.

## The daily job in Pattern

```
1. splice_daily_update.py --csv        # refresh prices (built, validated)
2. score_live.py                       # w27 ensemble -> today's p_up   [UNVERIFIED]
3. mark_book.py                        # NEW - build cell, mark NAV, publish
```

Step 3 is the only genuinely new code. It must:

- read `data/r1000_tradeable_universe.csv` and **drop `tradeable=False` names**
  — a stale price silently poisons `dv_60d`/`vol_60d`/`mom_12_1` and can put a
  name in the wrong tertile;
- apply the **$5 price floor** (README §J) so the tracked book is the
  deployable one, not the contaminated one;
- hold the month's names fixed and mark daily, rebalancing on the 20-day
  cycle — same buy-and-hold-within-period mechanic as `daily_from_monthly.py`;
- append atomically so a crashed run can't corrupt the series.

Then in Tracker: change `pattern`'s mode/cadence in `strategies.yaml`, add an
adapter in `scripts/adapters.py`, add a `STRATEGY_PERF` entry, run
`build_perf_report.py --strategy pattern`.

## Which book to track

I'd publish **two sleeves**, because they answer different questions and paper
tracking is free:

| Sleeve | Cell | Since-2020 net | Why track it |
|---|---|---:|---|
| `pattern-trash` | Low-dv × High-vol × Low-mom | +22.9% (t 2.64) | the actual alpha; capacity $500k–$1M, 25bp spreads, hard borrow |
| `pattern-megacap` | High-dv × High-vol | +10.5% (t 2.84) | 2.5bp spreads, Sharpe 1.23, deepest capacity — the one that could clear your constraints |

The mega-cap sleeve is the more interesting of the two operationally, and also
the weaker evidentially: 63 months, and it did **not** work full-sample. That
combination is exactly what a forward track is for.

I would **not** track the §H.2 "workhorse" cell (Mid × High × Low). It is
negative since 2020 and doesn't survive the price floor either — it was a
full-sample artifact.

## What is not done

Honest status, so nothing here reads as further along than it is:

- **`score_live.py` has never been run.** I flagged this earlier and did not
  test it. The whole plan sits on it. **Verify this first** — if it doesn't
  work on the spliced panel, everything downstream slips.
- **No NAV publisher exists.** `mark_book.py` is unwritten.
- **The splice has run exactly once.** Validated (99.0% coverage, overlap
  correlation ≥0.9949), but I'd watch a few daily runs before trusting it
  unattended — particularly whether the 33-name stale list stays stable.
- **No launchd entry.** Pattern's job must land before Tracker's 13:45 PT run,
  and register in the NightWatch Overseer manifest so a silent failure alerts.

## Sequence

1. Smoke-test `score_live.py` on the spliced panel — **gate for everything else**
2. Write `mark_book.py`, publish both sleeves, backfill NAV from the corrected
   backtest so the page opens with history rather than one dot
3. Wire Tracker: registry entry, adapter, `STRATEGY_PERF`
4. Schedule in Pattern (before 13:45 PT), register with the Overseer
5. Watch a week before treating the series as real

Step 2's backfill deserves a flag on the page itself: everything before the
go-live date is **backtest**, everything after is **paper**. Same research-vs-
paper split Tracker already enforces on SemiAnalysis, and for the same reason —
the backtest leg has been revised once already and shouldn't be allowed to
quietly become the track record.
