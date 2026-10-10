# Pattern frozen daily research database — v1

This is the fixed local baseline for subsequent Pattern research. Version: `pattern_daily_v1_20260918`. Do not overwrite this version; changes require a new version and manifest. Resolve the selected version using `data/frozen/CURRENT.json` in the Pattern repository. All paths in the manifest's file inventory are relative to this directory. The directory is self-contained: consumption does not require GDM or Bloomberg.

## What is frozen

- `prices.parquet`: exact copy of the existing June 30, 2026 GDM vintage; all daily source observations and original values retained.
- `quality.parquet`: one-to-one screening flags keyed by `security,date`, with diagnostic returns. Rows are sorted by security/date.
- `calendar_quality.parquet`: one-to-one session positions, missing-session counts, off-calendar flags, and continuity segments.
- `candidate_sessions.parquet`: fixed empirical session calendar. `calendar_evidence.csv` and `excluded_weekdays_review.csv` record its construction evidence.
- `security_master.parquet`: static reference metadata, not a point-in-time identity history.
- `adjustment_cash_events.csv`, `adjustment_split_events.csv`, `adjustment_split_conflicts.csv`: adjustment diagnostic annotations; join by security/date only after accounting for event multiplicity. Unresolved does not automatically mean erroneous. These annotations do not exclude observations by default.
- `manifest.json`: source provenance, schemas, counts, and SHA256 checksums. `verification.json` records build-time integrity checks. `verify.py` can independently rerun verification against the delivered files.

## Practical rules for using this database

Choose lookback L and forecast horizon H independently for each experiment. Neither is fixed to 20. Do not change the database to change L or H. Portfolio deciles and multi-class or ranking targets are experiment outputs, not stored database assumptions.

For a signal after session t closes, the visible chart uses L sessions t-L+1 through t. The close-to-close H-session research target is TRI(t+H)/TRI(t)-1. TRI is the supplied gross-dividend total-return index. Add no dividends or split factors a second time. Execution at a later price is a separate portfolio assumption.

If a moving average of M sessions is drawn throughout the chart, fetch L+M-1 sessions of history. Record M and the resulting effective input history in the experiment configuration. The original moving-average convention remains available; there is no fixed M in this database.

Use `current_renderer_bar_ok` for the existing renderer's input observations: finite positive internally consistent OHLC, finite nonnegative volume, and finite positive TRI. Zero volume is flagged but is not automatically excluded. Require complete input history within a security and uninterrupted calendar segment. Do not fill missing bars or compress the horizon by counting only available rows. These rules screen numerical defects; they do not certify every retained price.

Use the common calendar positions to find t+H. Require valid target endpoints for an ordinary TRI return and flag incomplete forward histories. Record counts of missing or unresolved labels; do not silently turn them into zero. Formation eligibility must use information available at formation only. Future completeness flags must never retrospectively remove holdings from portfolio returns. Retain and disclose unresolved terminal holdings rather than dropping or renormalizing them away.

`observed_*_return` and `screened_*_return` in the quality table are adjacent-observation diagnostics. They are NOT guaranteed daily returns and are not the H-session training target. `return_endpoint_ok` is a conservative numerical screen, not a determination of economic worthlessness.

No membership restriction, price floor, liquidity threshold, return clipping, or currently-active-security restriction is applied. Obvious bad rows remain stored with flags; consumers screen inputs rather than destructively deleting source records.

## Accepted limitations of v1

The calendar is an existing-data approximation: weekdays with at least 100 finite positive-close, positive-volume records. It spans 1980-07-28 to 2026-06-30. All earlier source history remains stored but lacks calendar support for these session-based windows. There are 407 excluded weekdays inside the calendar range, including holidays and potentially missing dates; exhaustive classification is not required for this practical v1 freeze. This start date is a calendar-support boundary, not a chosen model training start.

OHLC is historically adjusted, including evidence of special-distribution adjustments. It is not verified nominal executable pricing. Volume units and dollar-volume measures are not fully reconciled. Security metadata is static; a >180-calendar-day gap starts an unresolved identity boundary, without proving that shorter histories share one issuer. Terminal outcomes and extreme prices are not exhaustively resolved. Existing GDM coverage gates remain uncleared; freezing this research snapshot does not establish survivorship completeness or independent price certification.

These limitations are accepted and recorded for reproducibility. The frozen database is ready as the shared research input; model designs, horizons, eligibility choices and execution assumptions must still be recorded per experiment.

## Verification

Run the repository's Python interpreter with `verify.py <absolute path to this directory>`. This reads every inventoried file's checksum, compares all daily keys across the three daily tables, checks screening behavior and confirms that no fixed 20/20 window fields were frozen. Read-only file permissions discourage accidental modification; checksums detect changes. They are not a cryptographic access-control system.
