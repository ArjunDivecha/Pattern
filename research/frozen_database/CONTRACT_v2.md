# Pattern frozen daily research database — v2

Version `pattern_daily_v2_20261010`. It is the second fixed local baseline for Pattern research. Version 1 (`pattern_daily_v1_20260918`) is kept unchanged beside it. Resolve the selected version through `data/frozen/CURRENT.json`. Paths in the manifest are relative to this directory. Using the directory needs neither the Global Data Mart nor Bloomberg.

## What changed from v1

The prices come from the rebuilt Global Data Mart June 30, 2026 vintage, covering 9,470 priced names over 37.6 million rows. That vintage is survivorship-certified for the Russell 3000 from January 2000 to June 2026. The years 1996 to 1999 are about 79% covered and are not certified. Version 1 had 6,751 names and no survivorship claim.

The session calendar is now the real New York Stock Exchange calendar, from `exchange_calendars` (XNYS). It replaces the old rule of counting any weekday with at least 100 active names as a session. Rows dated on non-sessions are kept and flagged `off_exchange_calendar`, and each one is listed in `off_calendar_rows.parquet`.

Version 2 adds five tables:

- point-in-time Russell 3000 membership;
- traded (nominal) prices;
- delisting outcomes;
- identity segments;
- the Global Data Mart vintage manifest.

## Files

- `prices.parquet`: an exact copy of the vintage. Prices are split-adjusted, not dividend-adjusted. `total_return_index_gross_dividends` (TRI) is the total-return series.
- `nominal_prices.parquet`: one row per price row, giving the traded price on that day (`close_nominal`).
  - Rows marked `bloomberg_unadjusted` (5.7M, the October 2026 names) are Bloomberg's own unadjusted price.
  - Rows marked `reconstructed_from_splits` (31.9M) are rebuilt by undoing the recorded splits.
  - Tested on the October names, the reconstruction lands within 1% on 87.7% of rows. When it misses, it usually reads low because a split record is missing.
  - Since 2000 it puts 1.8% of rows on the wrong side of a $5 floor, against 8.6% for split-adjusted prices.
  - The exact check is in `gdm_research_extras_summary.json`.
- `quality.parquet`: one row per price row with the v1.0 screening flags, the same rules as v1, recomputed. It adds `off_exchange_calendar` and `bar_ok_on_calendar`, which is `current_renderer_bar_ok` and on a session.
- `calendar_quality.parquet`: one row per price row with:
  - `session_index`, the position in `sessions.parquet`, null when off-calendar;
  - `missing_sessions_before`;
  - `calendar_segment`.
- `calendar_segment` changes only at a missing session or at a gap over 180 days. An off-calendar row does not break a segment, because session-indexed windows never read it.
- `sessions.parquet`: the exchange calendar from the first to the last price date, with `session_index`.
- `calendar_evidence.parquet`: observed and active names on every session.
- `off_calendar_rows.parquet`: every price row on a non-session.
- `membership_pit.parquet`: Russell 3000 month-end membership from `INDX_MWEIGHT_HIST`, January 1996 to June 2026. All 11,011 names, including members the Data Mart never priced.
- `membership_vintage.parquet`: the vintage's own copy, restricted to its universe.
- `universe.parquet`: the vintage universe of 9,510 names, with cohort flags.
- `security_master.parquet`: static reference metadata. It is not point-in-time.
- `delisting_outcomes.parquet`: one row per name.
  - Outcome is active, acquired, bankrupt, delisted_other, liquidated or other, from Bloomberg `MARKET_STATUS` and the bankruptcy ticker suffix.
  - It carries the last price date and last price.
  - Deal terms are present only for names where `mna_history_pulled` is true.
- `identity_flags.parquet`: the segment table. A gap over 180 days starts a new segment, and a gap over 365 days sets `possible_different_issuer`.
- `gdm_manifest.json`, `gdm_research_extras_summary.json`: source provenance from the Data Mart.
- `build_summary.json`: build-time counts and cross-checks.
- `manifest.json`: hashes, schemas and row counts.
- `verification.json`: the result of `verify_v2.py` at build time.

## Rules for using this database

These carry over from v1.

Choose lookback L, horizon H and moving-average length M per experiment, and record them. The database stores no window, label or decile. A signal after the close of session t sees sessions t−L+1 through t, and needs L+M−1 sessions of history for a full moving average. The target is TRI(t+H)/TRI(t)−1, using session positions to find t+H. Require every input and endpoint bar to pass `bar_ok_on_calendar` and to sit in one `calendar_segment`. Do not fill missing bars or count only available rows.

No membership restriction, price floor, liquidity threshold, return clipping or active-only restriction is applied here. These are experiment choices.

- **Membership.** Use `membership_pit` at the month end on or before formation, never a later one.
- **Price floor.** Apply it to `close_nominal`, never to the split-adjusted `close`, which leaks future splits. Because the reconstruction is not exact, report results with the floor applied both ways until Bloomberg unadjusted prices exist for the older names.
- **Terminal returns.** A delisted holding must not silently vanish from a portfolio.
  - For `bankrupt`, the price series stops at the last listed price, and the economic outcome is usually near zero. The experiment must state the terminal value it assumes.
  - For `acquired`, the last price is close to the deal price.
- **Formation.** Eligibility must use information known at formation. Future completeness flags must never remove holdings after the fact.

## Known limits

Prices are split-adjusted and include special-distribution adjustments. Traded prices for names outside the October cohort are reconstructions. Security metadata is static. Identity within a segment is not proven. Deal terms are missing for most acquired names. Coverage before 2000 is incomplete. These limits are accepted and recorded.

## Verification

Run `verify_v2.py <absolute path to this directory>` with the repository's interpreter. It checks every file hash, and that the four daily tables share exactly the same keys. It also checks the screening flags, the calendar ordering and segment contiguity. Read-only permissions discourage edits, and the hashes detect them.
