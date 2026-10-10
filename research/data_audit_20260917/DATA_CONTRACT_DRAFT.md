# Research database definition — draft, 18 September 2026

Status: source and quality conventions documented; database is NOT certified for model training or reportable portfolio results. This document does not authorize a new training run.

## Source and units

Use only the existing Global Data Mart `equities.research_vintage_prices` June 30, 2026 vintage, resolved through the registry. Preserve raw rows and record source hashes. No Bloomberg requests. Index membership is not an eligibility requirement for this study. Study dates and security eligibility remain to be chosen.

OHLC values are historically adjusted, predominantly for splits with evidence of additional special-distribution adjustments. They are not verified nominal executable prices. Use the supplied gross-dividend total-return index ratios for diagnostic close-to-close returns; do not add dividends or apply splits again. Volume adjustment is not fully established, so nominal price floors and dollar-volume filters are not validated.

## Calendar and missingness rule

Define every horizon on a common session calendar, never by taking the next available N rows for each security. A 20-session forward horizon ends at calendar position t+20. A 20-session chart includes t-19 through t; the current renderer additionally needs 19 sessions of moving-average history (39 sessions total).

The delivered local calendar is PROVISIONAL: weekdays with at least 100 records having finite positive close and volume. This is empirical trading evidence, not an authoritative exchange calendar. Dates below that threshold, including ordinary holidays and potentially missing market dates, are exported for review. Dates before the resulting calendar starts are unsupported. Do not assume the threshold establishes the intended study start. Half-days count as sessions if included.

No forward-fill, interpolation, zero-return substitution, or compression across missing sessions. Keep off-calendar source observations with flags. Missing sessions, invalid bars, or unresolved identity boundaries invalidate a chart window until a complete new history accumulates.

The forward-completeness flag is a diagnostic requiring 21 valid return endpoints on consecutive sessions. It is NOT permission to select the investment universe using future data. Securities eligible at formation must remain in a portfolio accounting ledger even if their subsequent prices disappear. Missing terminal outcomes require explicit resolution or separately reported bounds; they cannot be silently deleted, renormalized away, or assigned zero. No portfolio engine has been implemented here.

Zero terminal prices and extreme moves remain in the raw data. A failed numerical validity check does not establish that an economic loss was unreal or justify excluding that firm retrospectively.

## Delivered layers

- `quality_layer_20260918T064856Z`: original row-level numeric and identity-boundary screening; its original window fields refer to observations, not sessions.
- `adjustment_audit_20260918T065713Z`: event diagnostics. Unresolved events are not automatically proven errors; event flags still need integration into the quality layer.
- `calendar_layer_*`: non-destructive, one-to-one session sidecar, calendar evidence and excluded-weekday review list. Its flags are necessary continuity checks, not sufficient overall eligibility checks.

## Decisions still open before freezing a training database

1. Choose study dates and eligible security types; assess coverage over that chosen period.
2. Validate the provisional calendar's excluded weekdays and unusual dates against adequate local evidence. A whole-market missing day cannot be distinguished from a holiday solely from absent price rows.
3. Integrate adjustment-event uncertainty and settle treatment of identity and terminal-price cases, with missing-outcome reporting.
4. Define portfolio execution timing, price units, and tradability separately from chart inputs and close-to-close research labels.

No arbitrary return clipping, price floor, source replacement, training, or backtesting is part of this step.

## Calendar implementation result

Full run: `calendar_layer_20260918T070935Z`. All 29,604,653 input rows retained; serialized output read back and compared with the computed frame. Original source SHA256 unchanged. Nineteen tests pass across calendar, quality and adjustment routines.

The empirical calendar contains 11,575 candidate sessions from 1980-07-28 through 2026-06-30. There are 407 excluded weekdays within that range requiring classification (not 407 proven errors). 460,780 observations follow at least one missing candidate session. 2,184,169 rows passed the previous 20-observation chart guard but fail the new session guard; this includes unsupported calendar dates and interruptions, not solely bad prices. No rows have been removed because of these flags.

This verifies the calendar-layer implementation, not the completeness of the source or correctness of every candidate session. Source coverage remains subject to GDM's uncleared research gate.
