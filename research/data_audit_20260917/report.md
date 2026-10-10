# Global Data Mart: 30-case local price audit

Date: 2026-09-17. Scope: existing local files only. No Bloomberg API calls, no source changes, no eligibility filters applied. This report does not use external web evidence. A few public-source searches were made before the user's local-only clarification; they are excluded from the findings.

## Findings

- Ten corporate-action controls pass limited local consistency checks: five splits have matching stored events and adjusted price continuity; five dividends reconcile to total-return arithmetic within 0.002 basis points.
- Five selected OHLC cases contain internal contradictions. Example: ASTH on 2013-11-07 has open/high/low 100.10 but close 5.35. KITDQ on 2005-08-24 has high/low 73.5011/61.2509 but close 6.3438. The correct replacement prices are NOT established.
- Five selected nonpositive observations cannot be used as ordinary positive-price inputs. A zero could represent a terminal event or a placeholder; the local evidence does not decide which. Do not delete a terminal loss from a portfolio ledger.
- All five selected extreme jumps start from a stored close of 0.000001. Their arithmetic is reproducible; actual trading, executable size, and source correctness are unverified.
- Five selected continuity breaks span roughly 25-30 years. Latest security-master identifiers do not establish historical continuity. The numerical returns across them are diagnostic artifacts, not valid holding-period returns.

## Adjustment provenance

The existing producer calls `bbg.hist_batch` for six price fields without explicit adjustment parameters (GDM `sources/equities/ingest/build_research_vintage.py`, `pull_price_payload`). The existing wrapper (`OpusBloomberg/bbg.py`, `hist_batch`) sets dates and periodicity but does not explicitly set `adjustmentFollowDPDF`, `adjustmentSplit`, `adjustmentNormal`, or `adjustmentAbnormal`. The effective historical settings are therefore not proved by the code inspected. Control events empirically support split-adjusted closes and dividend-inclusive TRI for those cases only. Volume conventions remain unresolved. Current code is not proof of the exact code version/settings used at every original retrieval.

The producer also drops rows with missing close before assembly. Consequently, zero missing closes in the final panel does not prove complete observation coverage.

## Proposed rules for your review — not applied

1. Give every source row a quality status and reason; retain raw evidence.
2. Quarantine internally inconsistent OHLC and nonpositive price inputs from chart construction pending resolution; flag any feature/label window they affect. Do not silently omit portfolio losses or exits.
3. Prevent returns and chart windows from spanning unresolved security-identity breaks. A 180-day gap is a screening threshold, not proof of a new issuer.
4. Preserve extreme positive-price observations, flag microprice/stale/zero-volume cases, and decide tradability separately. Do not winsorize returns to improve results.
5. Treat split-adjusted chart prices and total-return outcomes as distinct series; avoid applying dividends or splits twice. Obtain a documented nominal-price basis before interpreting a historical dollar-price filter.
6. The local event table named dividends also contains Stock Split rows that overlap splits.parquet. Filter by event type and deduplicate; never add a split ratio as a cash dividend.

## Evidence boundaries and sampling

This is a purposeful exception sample, not an error-rate estimate. Five distinct securities per anomaly group: largest absolute TRI returns with positive endpoints and gaps at most seven days; latest nonpositive observations; largest relative OHLC violations among closes at least 5 (for case selection only); longest gaps above 180 days. Tied dates follow input/sort order. Controls are five fixed known split dates and the last 2024 regular-cash ex-date for five fixed large-cap names. The sample uses all available history, including pre-1996 observations; it does not define the future study period. There are no separately sampled stale-price runs: zero-volume checks and surrounding observations are only preliminary stale-price evidence.

Each case includes up to three observations before and after the event. Windows are truncated at series boundaries; observation spacing is not necessarily daily. No matching event in a seven-day window is not proof that no corporate action occurred. Static security metadata is explicitly not point-in-time. Cached prices are same-source corroboration, not independent market validation.

## Files

- `reviewed_cases.csv`: final 30 classifications and calculations.
- `surrounding_prices.csv`: observed windows, including diagnostic returns.
- `nearby_events.csv`: stored events within seven calendar days.
- `security_identity.csv`: latest stored identifiers and names.
- `cache_reconciliation.csv`: comparison against per-security retrieval cache.
- `audit_metadata.json`: input file metadata and validation boundaries.
- `audit.py`, `render_report.py`: reproducible local-only audit.

## Cases

### C01 — Extreme return: TWER US Equity, 2022-12-23

**Unresolved microprice observation**

Positive-endpoint arithmetic produces the jump. Stored observations alone do not establish an executable trade or erroneous quote; do not clip or call it verified market performance.

TOWERSTREAM CORP; FIGI BBG000BB5XS4; ID EQ0000000002796346. Static metadata, not historical identity proof.

```text
      date     open     high      low    close  volume  total_return_index_gross_dividends  gap_days
2022-11-18 0.000001 0.000001 0.000001 0.000001   267.0                            0.000001       3.0
2022-12-13 0.150000 0.150000 0.150000 0.150000   505.0                            0.150000      25.0
2022-12-21 0.000001 0.000001 0.000001 0.000001  2362.0                            0.000001       8.0
2022-12-23 0.100000 0.100000 0.100000 0.100000   233.0                            0.100000       2.0
2023-01-06 0.150000 0.150000 0.150000 0.150000  9922.0                            0.150000      14.0
2023-09-12 0.105000 0.105000 0.105000 0.105000  1011.0                            0.105000     249.0
2024-08-01 0.000001 0.000001 0.000001 0.000001   127.0                            0.000001     324.0
```


### C02 — Extreme return: GTTNQ US Equity, 2021-11-04

**Unresolved microprice observation**

Positive-endpoint arithmetic produces the jump. Stored observations alone do not establish an executable trade or erroneous quote; do not clip or call it verified market performance.

GTT COMMUNICATIONS INC; FIGI BBG000QWDVT6; ID EQ0000000001765595. Static metadata, not historical identity proof.

```text
      date   open    high      low    close   volume  total_return_index_gross_dividends  gap_days
2021-11-01 0.0556 0.07000 0.050000 0.070000  71782.0                            0.070000       3.0
2021-11-02 0.0500 0.06000 0.050000 0.060000  25998.0                            0.060000       1.0
2021-11-03 0.0500 0.06000 0.000001 0.000001 411736.0                            0.000001       1.0
2021-11-04 0.0424 0.05070 0.000100 0.050000 139285.0                            0.050000       1.0
2021-11-05 0.0450 0.05990 0.010500 0.011300  83435.0                            0.011300       1.0
2021-11-08 0.0212 0.05565 0.011300 0.011400  31811.0                            0.011400       3.0
2021-11-09 0.0312 0.06000 0.011500 0.012100 153139.0                            0.012100       1.0
```


### C03 — Extreme return: LICYQ US Equity, 2025-07-24

**Unresolved microprice observation**

Positive-endpoint arithmetic produces the jump. Stored observations alone do not establish an executable trade or erroneous quote; do not clip or call it verified market performance.

LI-CYCLE HOLDINGS CORP; FIGI BBG0122LQVR9; ID EQ0000000089900282. Static metadata, not historical identity proof.

```text
      date     open     high      low    close   volume  total_return_index_gross_dividends  gap_days
2025-07-21 0.010000 0.031000 0.010000 0.011000 115799.0                            0.011000       3.0
2025-07-22 0.001100 0.051900 0.001100 0.001200  48987.0                            0.001200       1.0
2025-07-23 0.001100 0.020000 0.000001 0.000001 189580.0                            0.000001       1.0
2025-07-24 0.001100 0.031000 0.000100 0.030000 246216.0                            0.030000       1.0
2025-07-25 0.005000 0.005000 0.000001 0.000001  17121.0                            0.000001       1.0
2025-07-28 0.000001 0.000001 0.000001 0.000001   4232.0                            0.000001       3.0
2025-07-29 0.000100 0.000100 0.000001 0.000001  34906.0                            0.000001       1.0
```


### C04 — Extreme return: HGGG US Equity, 2022-04-20

**Unresolved microprice observation**

Positive-endpoint arithmetic produces the jump. Stored observations alone do not establish an executable trade or erroneous quote; do not clip or call it verified market performance.

HHGREGG INC; FIGI BBG000R6LW66; ID EQ0000000003446011. Static metadata, not historical identity proof.

```text
      date     open     high      low    close  volume  total_return_index_gross_dividends  gap_days
2022-01-19 0.000100 0.002000 0.000100 0.002000 28120.0                            0.002000      20.0
2022-04-12 0.000001 0.000001 0.000001 0.000001   500.0                            0.000001      83.0
2022-04-14 0.000001 0.000001 0.000001 0.000001   100.0                            0.000001       2.0
2022-04-20 0.017900 0.017900 0.017900 0.017900   500.0                            0.017900       6.0
2022-04-21 0.017800 0.017800 0.017800 0.017800  2000.0                            0.017800       1.0
2022-04-22 0.000100 0.000100 0.000100 0.000100  1000.0                            0.000100       1.0
2022-05-03 0.000100 0.000100 0.000100 0.000100   150.0                            0.000100      11.0
```


### C05 — Extreme return: CDRBQ US Equity, 2018-12-18

**Unresolved microprice observation**

Positive-endpoint arithmetic produces the jump. Stored observations alone do not establish an executable trade or erroneous quote; do not clip or call it verified market performance.

CODE REBEL CORP; FIGI BBG008F92QD2; ID EQ0000000042109683. Static metadata, not historical identity proof.

```text
      date     open     high      low    close  volume  total_return_index_gross_dividends  gap_days
2018-09-14 0.000010 0.000010 0.000010 0.000010  2700.0                            0.000010       2.0
2018-11-13 0.000001 0.000001 0.000001 0.000001   200.0                            0.000001      60.0
2018-12-12 0.000001 0.000001 0.000001 0.000001   100.0                            0.000001      29.0
2018-12-18 0.010100 0.010100 0.010000 0.010100 17080.0                            0.010100       6.0
2018-12-20 0.009900 0.009900 0.009900 0.009900  2980.0                            0.009900       2.0
2018-12-21 0.000001 0.000001 0.000001 0.000001   360.0                            0.000001       1.0
2019-01-09 0.000100 0.000100 0.000100 0.000100   300.0                            0.000100      19.0
```


### C06 — Nonpositive price: INFIQ US Equity, 2026-06-30

**Confirmed unusable observation; source cause unresolved**

Nonpositive close/TRI cannot support an ordinary positive-price stock return. Could be source placeholder or terminal security event; do not infer worthless delisting or replace with zero return.

INFINITY PHARMACEUTICALS INC; FIGI BBG000CN3TG1; ID EQ0097401800001000. Static metadata, not historical identity proof.

```text
      date  open  high  low  close  volume  total_return_index_gross_dividends  gap_days
2026-06-22   0.0   0.0  0.0    0.0   314.0                                 0.0       4.0
2026-06-23   0.0   0.0  0.0    0.0 20502.0                                 0.0       1.0
2026-06-24   0.0   0.0  0.0    0.0   200.0                                 0.0       1.0
2026-06-30   0.0   0.0  0.0    0.0  6809.0                                 0.0       6.0
```


### C07 — Nonpositive price: AILEQ US Equity, 2026-06-30

**Confirmed unusable observation; source cause unresolved**

Nonpositive close/TRI cannot support an ordinary positive-price stock return. Could be source placeholder or terminal security event; do not infer worthless delisting or replace with zero return.

ILEARNINGENGINES INC; FIGI BBG00Z9114K8; ID EQ0000000089319862. Static metadata, not historical identity proof.

```text
      date   open   high  low  close  volume  total_return_index_gross_dividends  gap_days
2026-05-22 0.0001 0.0001  0.0    0.0 11492.0                                 0.0      15.0
2026-06-17 0.0000 0.0000  0.0    0.0   100.0                                 0.0      26.0
2026-06-25 0.0000 0.3037  0.0    0.0   676.0                                 0.0       8.0
2026-06-30 0.0000 0.0000  0.0    0.0   200.0                                 0.0       5.0
```


### C08 — Nonpositive price: YTENQ US Equity, 2026-06-30

**Confirmed unusable observation; source cause unresolved**

Nonpositive close/TRI cannot support an ordinary positive-price stock return. Could be source placeholder or terminal security event; do not infer worthless delisting or replace with zero return.

YIELD10 BIOSCIENCE INC; FIGI BBG000J1XKK3; ID EQ0732034000001000. Static metadata, not historical identity proof.

```text
      date   open   high    low  close  volume  total_return_index_gross_dividends  gap_days
2026-03-23 0.0000 0.0000 0.0000 0.0000   166.0                              0.0000      33.0
2026-05-11 0.0001 0.0001 0.0001 0.0001   169.0                              0.0001      49.0
2026-06-01 0.0001 0.0001 0.0001 0.0001   117.0                              0.0001      21.0
2026-06-30 0.0000 0.0000 0.0000 0.0000   291.0                              0.0000      29.0
```


### C09 — Nonpositive price: ASAPQ US Equity, 2026-06-30

**Confirmed unusable observation; source cause unresolved**

Nonpositive close/TRI cannot support an ordinary positive-price stock return. Could be source placeholder or terminal security event; do not infer worthless delisting or replace with zero return.

WAITR HOLDINGS INC; FIGI BBG00D0L7DK6; ID EQ0000000049883865. Static metadata, not historical identity proof.

```text
      date  open  high  low  close  volume  total_return_index_gross_dividends  gap_days
2026-06-16   0.0   0.0  0.0    0.0  3697.0                                 0.0       7.0
2026-06-22   0.0   0.0  0.0    0.0   235.0                                 0.0       6.0
2026-06-24   0.0   0.0  0.0    0.0   357.0                                 0.0       2.0
2026-06-30   0.0   0.0  0.0    0.0 30647.0                                 0.0       6.0
```


### C10 — Nonpositive price: ATHXQ US Equity, 2026-06-30

**Confirmed unusable observation; source cause unresolved**

Nonpositive close/TRI cannot support an ordinary positive-price stock return. Could be source placeholder or terminal security event; do not infer worthless delisting or replace with zero return.

ATHERSYS INC; FIGI BBG000QBZBS3; ID EQ0000000002971932. Static metadata, not historical identity proof.

```text
      date   open   high    low  close  volume  total_return_index_gross_dividends  gap_days
2026-06-17 0.0001 0.0001 0.0001 0.0001  2909.0                              0.0001       5.0
2026-06-25 0.0000 0.0000 0.0000 0.0000  8156.0                              0.0000       8.0
2026-06-29 0.0000 0.0000 0.0000 0.0000  2014.0                              0.0000       4.0
2026-06-30 0.0000 0.0000 0.0000 0.0000  2140.0                              0.0000       1.0
```


### C11 — OHLC inconsistency: ASTH US Equity, 2013-11-07

**Confirmed internal inconsistency; source cause unresolved**

Open/close lies outside supplied high-low interval. Adjustment mismatch, session definitions, or vendor record error remain alternatives.

ASTRANA HEALTH INC; FIGI BBG000H7ZK57; ID EQ0018788400001000. Static metadata, not historical identity proof.

```text
      date   open   high    low  close  volume  total_return_index_gross_dividends  gap_days
2013-10-25   5.50   5.50   5.50   5.50    70.0                                5.50       3.0
2013-10-29   5.50   5.50   5.50   5.50    50.0                                5.50       4.0
2013-11-04   5.35   5.35   5.35   5.35    50.0                                5.35       6.0
2013-11-07 100.10 100.10 100.10   5.35    50.0                                5.35       3.0
2013-11-08   3.81   5.45   3.81   5.35  1480.0                                5.35       1.0
2013-11-11   5.45   5.45   5.45   5.45   180.0                                5.45       3.0
2013-11-12   5.45   5.45   5.45   5.45   140.0                                5.45       1.0
```


### C12 — OHLC inconsistency: CRXM US Equity, 1998-10-13

**Confirmed internal inconsistency; source cause unresolved**

Open/close lies outside supplied high-low interval. Adjustment mismatch, session definitions, or vendor record error remain alternatives.

TAXUS CARDIUM PHARMACEUTICAL; FIGI BBG000LYJ894; ID EQ0015907400001000. Static metadata, not historical identity proof.

```text
      date       open       high       low     close  volume  total_return_index_gross_dividends  gap_days
1998-10-08 154.398693 154.398693  0.964992 77.199346    28.0                           96.116878       1.0
1998-10-09  38.599673 115.799020 19.299837 77.199346     5.0                           96.116878       1.0
1998-10-12  77.199346  77.199346 19.299837 77.199346     8.0                           96.116878       3.0
1998-10-13 964.991830  96.499183 19.299837 96.499183    10.0                          120.146098       1.0
1998-10-14  96.499183  96.499183 38.599673 38.599673    27.0                           48.058439       1.0
1998-10-15  38.599673 192.998366 19.299837 57.899510    37.0                           72.087659       1.0
1998-10-16  77.199346  77.199346  7.719935 77.199346    23.0                           96.116878       1.0
```


### C13 — OHLC inconsistency: KITDQ US Equity, 2005-08-24

**Confirmed internal inconsistency; source cause unresolved**

Open/close lies outside supplied high-low interval. Adjustment mismatch, session definitions, or vendor record error remain alternatives.

PIKSEL INC; FIGI BBG000BV9SD3; ID EQ0116709900001000. Static metadata, not historical identity proof.

```text
      date    open    high     low   close  volume  total_return_index_gross_dividends  gap_days
2005-08-19 36.5318 47.9070 36.5318 45.5007  1480.0                             45.5007       1.0
2005-08-22 47.2507 50.7508 33.2505 47.2507  1390.0                             47.2507       3.0
2005-08-23 49.0007 73.5011 49.0007 64.7510  4890.0                             64.7510       1.0
2005-08-24 68.2510 73.5011 61.2509  6.3438  2360.0                              6.3438       1.0
2005-08-25 63.8760 66.5010 61.2509 66.5010   506.0                             66.5010       1.0
2005-08-26 70.0011 77.0012 70.0011 75.0324  1733.0                             75.0324       1.0
2005-08-29 77.0012 77.0012 71.7511 71.7511   174.0                             71.7511       3.0
```


### C14 — OHLC inconsistency: PRK US Equity, 1993-06-07

**Confirmed internal inconsistency; source cause unresolved**

Open/close lies outside supplied high-low interval. Adjustment mismatch, session definitions, or vendor record error remain alternatives.

PARK NATIONAL CORP; FIGI BBG000DCV3T6; ID EQ0010598200001000. Static metadata, not historical identity proof.

```text
      date    open    high     low   close  volume  total_return_index_gross_dividends  gap_days
1993-05-27 33.4391 33.6590 33.4391 33.6590  3528.0                             37.9682       2.0
1993-06-01 34.3190 34.3190 34.3190 34.3190   221.0                             38.7127       5.0
1993-06-02 33.6590 34.3190 33.6590 34.3190  1323.0                             38.7127       1.0
1993-06-07 33.6590 33.6590 33.6590  8.2498   221.0                              9.3059       5.0
1993-06-10 32.9991 33.8240 32.9991 33.8240   441.0                             38.1543       3.0
1993-06-14 32.9991 32.9991 32.9991 32.9991   662.0                             37.2237       4.0
1993-06-15 33.6590 33.6590 32.9991 33.6590  1544.0                             37.9682       1.0
```


### C15 — OHLC inconsistency: INDB US Equity, 1989-12-12

**Confirmed internal inconsistency; source cause unresolved**

Open/close lies outside supplied high-low interval. Adjustment mismatch, session definitions, or vendor record error remain alternatives.

INDEPENDENT BANK CORP/MA; FIGI BBG000BM0597; ID EQ0010468900001000. Static metadata, not historical identity proof.

```text
      date  open   high    low  close  volume  total_return_index_gross_dividends  gap_days
1989-12-07 8.000  8.375  8.000  8.250   700.0                              8.8235       1.0
1989-12-08 8.125  8.125  8.125  8.125   100.0                              8.6898       1.0
1989-12-11 8.125  8.375  8.125  8.250  5000.0                              8.8235       3.0
1989-12-12 8.250 16.750 16.500  8.250  3600.0                              8.8235       1.0
1989-12-13 8.250  8.250  8.250  8.250  2800.0                              8.8235       1.0
1989-12-15 8.250  8.250  8.250  8.250  4500.0                              8.8235       2.0
1989-12-18 8.250  8.750  8.250  8.750  4300.0                              9.4545       3.0
```

Stored nearby events:

```text
event_table    Ex-Date  Dividend Amount Dividend Type
  dividends 1989-12-18             0.09  Regular Cash
```


### C16 — Long gap: NPB US Equity, 2025-02-13

**Unresolved identity/continuity**

Large calendar gap is verified; latest static FIGI/name does not prove both segments belong to one issuer. Do not bridge until continuity is established.

NORTHPOINTE BANCSHARES INC; FIGI BBG000QKT8D3; ID EQ0000000001440668. Static metadata, not historical identity proof.

```text
      date   open   high    low  close    volume  total_return_index_gross_dividends  gap_days
1994-09-06 10.875 10.875 10.875 10.875     800.0                              10.875       4.0
1994-09-07 10.875 11.000 10.875 11.000    3600.0                              11.000       1.0
1994-09-08 11.000 11.000 10.875 10.875    4100.0                              10.875       1.0
2025-02-13    NaN    NaN    NaN 14.500       NaN                              14.500   11116.0
2025-02-14 14.600 14.700 14.500 14.500 2706961.0                              14.500       1.0
2025-02-18 14.500 14.750 14.500 14.590  182819.0                              14.590       4.0
2025-02-19 14.500 14.570 14.250 14.540  236246.0                              14.540       1.0
```


### C17 — Long gap: CBNA US Equity, 2024-10-03

**Unresolved identity/continuity**

Large calendar gap is verified; latest static FIGI/name does not prove both segments belong to one issuer. Do not bridge until continuity is established.

CHAIN BRIDGE BANCORP INC-A; FIGI BBG000V8CRG7; ID EQ0000000005582337. Static metadata, not historical identity proof.

```text
      date  open  high   low  close   volume  total_return_index_gross_dividends  gap_days
1995-06-26  0.73  0.73  0.73   0.73   2500.0                                0.73       NaN
2024-10-03   NaN   NaN   NaN  22.00      NaN                               22.00   10692.0
2024-10-04 22.00 22.00 21.01  21.20 345105.0                               21.20       1.0
2024-10-07 21.11 21.15 20.75  20.75  76148.0                               20.75       3.0
2024-10-08 21.70 21.70 20.75  20.79  23282.0                               20.79       1.0
```


### C18 — Long gap: CCP US Equity, 2015-08-14

**Unresolved identity/continuity**

Large calendar gap is verified; latest static FIGI/name does not prove both segments belong to one issuer. Do not bridge until continuity is established.

CARE CAPITAL PROPERTIES INC; FIGI BBG008G5SGL4; ID EQ0000000042217190. Static metadata, not historical identity proof.

```text
      date   open   high    low  close    volume  total_return_index_gross_dividends  gap_days
1988-07-11  5.875  5.875  5.750  5.750       NaN                               5.750       4.0
1988-07-12  5.875  5.875  5.875  5.875       NaN                               5.875       1.0
1988-07-13  5.750  5.750  5.625  5.625       NaN                               5.625       1.0
2015-08-14 30.000 31.560 30.000 31.500   35559.0                              31.500    9893.0
2015-08-17 31.990 37.680 31.500 34.050 8273155.0                              34.050       3.0
2015-08-18 34.050 36.440 33.550 35.580 5244880.0                              35.580       1.0
2015-08-19 34.830 35.650 34.110 35.610 3112528.0                              35.610       1.0
```


### C19 — Long gap: CRCM US Equity, 2014-01-23

**Unresolved identity/continuity**

Large calendar gap is verified; latest static FIGI/name does not prove both segments belong to one issuer. Do not bridge until continuity is established.

CARE.COM INC; FIGI BBG0026ZHV68; ID EQ0000000020618051. Static metadata, not historical identity proof.

```text
      date   open   high    low  close    volume  total_return_index_gross_dividends  gap_days
1988-03-09 17.375 17.500 16.375  16.50       NaN                               16.50       1.0
1988-03-10 16.375 16.625 15.250  15.75       NaN                               15.75       1.0
1988-03-11 15.375 16.500 15.125  15.75       NaN                               15.75       1.0
2014-01-23    NaN    NaN    NaN  17.00       NaN                               17.00    9449.0
2014-01-24 21.210 24.400 21.210  24.30 7471826.0                               24.30       1.0
2014-01-27 25.120 27.820 24.900  24.90 1393615.0                               24.90       3.0
2014-01-28 25.250 25.900 24.220  25.50  698004.0                               25.50       1.0
```


### C20 — Long gap: LDI US Equity, 2021-02-10

**Unresolved identity/continuity**

Large calendar gap is verified; latest static FIGI/name does not prove both segments belong to one issuer. Do not bridge until continuity is established.

LOANDEPOT INC -CLASS A; FIGI BBG00B5Z6KK9; ID EQ0000000045736479. Static metadata, not historical identity proof.

```text
      date    open    high     low   close    volume  total_return_index_gross_dividends  gap_days
1995-04-07  2.4871  2.4871  2.4871  2.4871    5000.0                              2.5625      36.0
1995-06-01  3.5790  3.5790  3.5790  3.5790     500.0                              3.6875      55.0
1995-09-20  5.8840  5.8840  5.8840  5.8840     200.0                              6.0625     111.0
2021-02-10     NaN     NaN     NaN 13.5879       NaN                             14.0000    9275.0
2021-02-11 13.5879 23.0905 13.5879 21.3524 7896516.0                             22.0000       1.0
2021-02-12 24.0214 38.6769 22.4297 30.5533 9535177.0                             31.4800       1.0
2021-02-16 30.1845 33.9697 25.2346 25.6714 2413748.0                             26.4500       4.0
```


### C21 — Split control: AAPL US Equity, 2020-08-31

**Pass: stored split event consistent with adjusted continuity**

Stored event records 4-for-1 split on this date. Previous close 124.808, event-day close 129.04; ordinary price return 3.3912%. History is consistent with split-adjustment, not raw nominal prices. Do not apply the split a second time. Volume adjustment and full-history cash/special-distribution settings remain unverified.

APPLE INC; FIGI BBG000B9XRY4; ID EQ0010169500001000. Static metadata, not historical identity proof.

```text
      date     open     high      low    close      volume  total_return_index_gross_dividends  gap_days
2020-08-26 126.1791 126.9925 125.0825 126.5225 163022268.0                            160.4389       1.0
2020-08-27 127.1425 127.4850 123.8325 125.0100 155552384.0                            158.5209       1.0
2020-08-28 126.0125 126.4425 124.5775 124.8075 187629916.0                            158.2642       1.0
2020-08-31 127.5800 131.0000 126.0000 129.0400 225768339.0                            163.6312       3.0
2020-09-01 132.7600 134.8000 130.5300 134.1800 152470142.0                            170.1491       1.0
2020-09-02 137.5900 137.9800 127.0000 131.4000 200118991.0                            166.6239       1.0
2020-09-03 126.9100 128.8400 120.5000 120.8800 257599640.0                            153.2838       1.0
```

Stored nearby events:

```text
event_table    Ex-Date  Dividend Amount Dividend Type
     splits 2020-08-31              4.0   Stock Split
  dividends 2020-08-31              4.0   Stock Split
```


### C22 — Split control: TSLA US Equity, 2020-08-31

**Pass: stored split event consistent with adjusted continuity**

Stored event records 5-for-1 split on this date. Previous close 147.56, event-day close 166.107; ordinary price return 12.5689%. History is consistent with split-adjustment, not raw nominal prices. Do not apply the split a second time. Volume adjustment and full-history cash/special-distribution settings remain unverified.

TESLA INC; FIGI BBG000N9MNX3; ID EQ0000000003531703. Static metadata, not historical identity proof.

```text
      date     open     high      low    close      volume  total_return_index_gross_dividends  gap_days
2020-08-26 137.3333 144.4000 136.9086 143.5447 213590730.0                            143.5447       1.0
2020-08-27 145.3640 153.0400 142.8333 149.2500 355395645.0                            149.2500       1.0
2020-08-28 153.0080 154.5660 145.7680 147.5600 301217640.0                            147.5600       1.0
2020-08-31 148.2033 166.7133 146.7033 166.1067 355246701.0                            166.1067       3.0
2020-09-01 167.3800 167.4967 156.8367 158.3500 270358257.0                            158.3500       1.0
2020-09-02 159.6633 159.6800 135.0402 149.1233 288842937.0                            149.1233       1.0
2020-09-03 135.7433 143.9333 134.0000 135.6667 262788258.0                            135.6667       1.0
```

Stored nearby events:

```text
event_table    Ex-Date  Dividend Amount Dividend Type
     splits 2020-08-31              5.0   Stock Split
  dividends 2020-08-31              5.0   Stock Split
```


### C23 — Split control: NVDA US Equity, 2024-06-10

**Pass: stored split event consistent with adjusted continuity**

Stored event records 10-for-1 split on this date. Previous close 120.888, event-day close 121.79; ordinary price return 0.7461%. History is consistent with split-adjustment, not raw nominal prices. Do not apply the split a second time. Volume adjustment and full-history cash/special-distribution settings remain unverified.

NVIDIA CORP; FIGI BBG000BBJQV0; ID EQ0031365700001000. Static metadata, not historical identity proof.

```text
      date    open     high     low   close      volume  total_return_index_gross_dividends  gap_days
2024-06-05 118.371 122.4495 117.468 122.440 528401780.0                            133.4596       1.0
2024-06-06 124.048 125.5870 118.320 120.998 664696190.0                            131.8879       1.0
2024-06-07 119.770 121.6917 118.022 120.888 412385800.0                            131.7680       1.0
2024-06-10 120.370 123.1000 117.010 121.790 314160362.0                            132.7511       3.0
2024-06-11 121.770 122.8700 118.740 120.910 222551158.0                            131.8028       1.0
2024-06-12 123.060 126.8800 122.570 125.200 299595001.0                            136.4793       1.0
2024-06-13 129.390 129.8000 127.160 129.610 260704537.0                            141.2866       1.0
```

Stored nearby events:

```text
event_table    Ex-Date  Dividend Amount Dividend Type
     splits 2024-06-10            10.00   Stock Split
  dividends 2024-06-11             0.01  Regular Cash
  dividends 2024-06-10            10.00   Stock Split
```


### C24 — Split control: AMZN US Equity, 2022-06-06

**Pass: stored split event consistent with adjusted continuity**

Stored event records 20-for-1 split on this date. Previous close 122.35, event-day close 124.79; ordinary price return 1.9943%. History is consistent with split-adjustment, not raw nominal prices. Do not apply the split a second time. Volume adjustment and full-history cash/special-distribution settings remain unverified.

AMAZON.COM INC; FIGI BBG000BVPV84; ID EQ0021695200001000. Static metadata, not historical identity proof.

```text
      date    open    high     low   close      volume  total_return_index_gross_dividends  gap_days
2022-06-01 122.256 125.179 120.622 121.684 127528980.0                             121.684       1.0
2022-06-02 121.684 125.610 120.045 125.511 100560680.0                             125.511       1.0
2022-06-03 124.200 124.400 121.046 122.350  97603320.0                             122.350       1.0
2022-06-06 125.245 128.990 123.810 124.790 135269024.0                             124.790       3.0
2022-06-07 122.005 124.100 120.630 123.000  85156712.0                             123.000       1.0
2022-06-08 122.610 123.750 120.750 121.180  64926594.0                             121.180       1.0
2022-06-09 119.990 121.300 116.100 116.150  67029842.0                             116.150       1.0
```

Stored nearby events:

```text
event_table    Ex-Date  Dividend Amount Dividend Type
     splits 2022-06-06             20.0   Stock Split
  dividends 2022-06-06             20.0   Stock Split
```


### C25 — Split control: GOOGL US Equity, 2022-07-18

**Pass: stored split event consistent with adjusted continuity**

Stored event records 20-for-1 split on this date. Previous close 111.778, event-day close 109.03; ordinary price return -2.4584%. History is consistent with split-adjustment, not raw nominal prices. Do not apply the split a second time. Volume adjustment and full-history cash/special-distribution settings remain unverified.

ALPHABET INC-CL A; FIGI BBG009S39JX6; ID EQ0000000044666717. Static metadata, not historical identity proof.

```text
      date    open    high     low   close     volume  total_return_index_gross_dividends  gap_days
2022-07-13 111.701 114.196 111.202 111.354 43722020.0                             111.354       1.0
2022-07-14 110.211 111.093 108.373 110.367 37003040.0                             110.367       1.0
2022-07-15 112.001 113.141 110.900 111.778 46769120.0                             111.778       1.0
2022-07-18 112.640 113.680 108.370 109.030 43164588.0                             109.030       3.0
2022-07-19 110.850 114.020 109.560 113.810 36530945.0                             113.810       1.0
2022-07-20 113.170 115.400 112.480 113.900 35600426.0                             113.900       1.0
2022-07-21 114.280 114.550 111.113 114.340 32676166.0                             114.340       1.0
```

Stored nearby events:

```text
event_table    Ex-Date  Dividend Amount Dividend Type
     splits 2022-07-18             20.0   Stock Split
  dividends 2022-07-18             20.0   Stock Split
```


### C26 — Dividend control: AAPL US Equity, 2024-11-08

**Pass: stored cash dividend reconciles to total return**

TRI return minus (close + cash dividend)/previous close - 1 = 0.001334 basis points. This validates arithmetic for this event only, using the same provider's stored records.

APPLE INC; FIGI BBG000B9XRY4; ID EQ0010169500001000. Static metadata, not historical identity proof.

```text
      date    open    high     low  close     volume  total_return_index_gross_dividends  gap_days
2024-11-05 221.795 223.950 221.140 223.45 28111338.0                            289.9383       1.0
2024-11-06 222.610 226.065 221.190 222.72 54561121.0                            288.9911       1.0
2024-11-07 224.625 227.875 224.570 227.48 42137691.0                            295.1674       1.0
2024-11-08 227.170 228.660 226.405 226.96 38328824.0                            294.8171       1.0
2024-11-11 225.000 225.700 221.500 224.23 42005602.0                            291.2709       3.0
2024-11-12 224.550 225.590 223.355 224.23 40398299.0                            291.2709       1.0
2024-11-13 224.010 226.650 222.760 225.12 48566217.0                            292.4270       1.0
```

Stored nearby events:

```text
event_table    Ex-Date  Dividend Amount Dividend Type
  dividends 2024-11-08             0.25  Regular Cash
```


### C27 — Dividend control: MSFT US Equity, 2024-11-21

**Pass: stored cash dividend reconciles to total return**

TRI return minus (close + cash dividend)/previous close - 1 = 0.001099 basis points. This validates arithmetic for this event only, using the same provider's stored records.

MICROSOFT CORP; FIGI BBG000BPH459; ID EQ0010174300001000. Static metadata, not historical identity proof.

```text
      date    open     high      low  close     volume  total_return_index_gross_dividends  gap_days
2024-11-18 414.870 418.4037 412.1000 415.76 24742013.0                            672.8516       3.0
2024-11-19 413.110 417.9400 411.5500 417.79 18133529.0                            676.1369       1.0
2024-11-20 416.870 417.2900 410.5800 415.49 19191655.0                            672.4146       1.0
2024-11-21 419.500 419.7800 410.2887 412.87 20780162.0                            669.5178       1.0
2024-11-22 411.365 417.4000 411.0600 417.00 24814626.0                            676.2150       1.0
2024-11-25 418.380 421.0800 414.8500 418.79 27691090.0                            679.1177       3.0
2024-11-26 419.585 429.0414 418.8500 427.99 23458889.0                            694.0366       1.0
```

Stored nearby events:

```text
event_table    Ex-Date  Dividend Amount Dividend Type
  dividends 2024-11-21             0.83  Regular Cash
```


### C28 — Dividend control: JNJ US Equity, 2024-11-26

**Pass: stored cash dividend reconciles to total return**

TRI return minus (close + cash dividend)/previous close - 1 = 0.000946 basis points. This validates arithmetic for this event only, using the same provider's stored records.

JOHNSON & JOHNSON; FIGI BBG000BMHYD1; ID EQ0010082800001000. Static metadata, not historical identity proof.

```text
      date   open    high      low  close     volume  total_return_index_gross_dividends  gap_days
2024-11-21 153.62 156.060 152.8100 155.50 10141056.0                            483.9617       1.0
2024-11-22 155.90 157.115 154.1100 155.17  8265961.0                            482.9346       1.0
2024-11-25 155.17 157.035 155.1400 155.78 12256242.0                            484.8331       3.0
2024-11-26 155.16 155.250 153.1600 154.52  5683499.0                            484.7709       1.0
2024-11-27 154.63 156.630 154.6000 155.40  4140444.0                            487.5317       1.0
2024-11-29 154.89 155.670 154.1690 155.01  5687821.0                            486.3081       2.0
2024-12-02 154.75 155.000 153.9258 154.80  6946494.0                            485.6493       3.0
```

Stored nearby events:

```text
event_table    Ex-Date  Dividend Amount Dividend Type
  dividends 2024-11-26             1.24  Regular Cash
```


### C29 — Dividend control: KO US Equity, 2024-11-29

**Pass: stored cash dividend reconciles to total return**

TRI return minus (close + cash dividend)/previous close - 1 = 0.001895 basis points. This validates arithmetic for this event only, using the same provider's stored records.

COCA-COLA CO/THE; FIGI BBG000BMX289; ID EQ0010035500001000. Static metadata, not historical identity proof.

```text
      date   open  high    low  close     volume  total_return_index_gross_dividends  gap_days
2024-11-25 64.320 64.71 63.840  64.38 36805252.0                            247.5783       3.0
2024-11-26 64.310 64.64 64.060  64.55 14473271.0                            248.2321       1.0
2024-11-27 64.960 65.10 64.250  64.43 13263403.0                            247.7706       1.0
2024-11-29 64.020 64.25 63.770  64.08 10648354.0                            248.2898       2.0
2024-12-02 64.050 64.19 63.345  63.65 16500095.0                            246.6237       3.0
2024-12-03 63.720 64.08 63.025  63.52 24874490.0                            246.1199       1.0
2024-12-04 63.275 63.35 61.900  62.21 28137660.0                            241.0441       1.0
```

Stored nearby events:

```text
event_table    Ex-Date  Dividend Amount Dividend Type
  dividends 2024-11-29            0.485  Regular Cash
```


### C30 — Dividend control: XOM US Equity, 2024-11-14

**Pass: stored cash dividend reconciles to total return**

TRI return minus (close + cash dividend)/previous close - 1 = -0.001075 basis points. This validates arithmetic for this event only, using the same provider's stored records.

EXXONMOBIL HOLDINGS CORP; FIGI BBG023CY9MM1; ID EQ0000000408566008. Static metadata, not historical identity proof.

```text
      date    open   high      low  close     volume  total_return_index_gross_dividends  gap_days
2024-11-11 121.000 121.62 120.0700 120.47 11969663.0                            758.4824       3.0
2024-11-12 121.130 121.31 119.8365 120.35 11972802.0                            757.7269       1.0
2024-11-13 120.565 122.05 118.8000 121.47 15127877.0                            764.7785       1.0
2024-11-14 121.660 121.88 120.3300 120.56 13041845.0                            765.2821       1.0
2024-11-15 120.400 121.24 119.1300 119.31 19051713.0                            757.3475       1.0
2024-11-18 119.790 120.62 119.2700 120.31 14243701.0                            763.6952       3.0
2024-11-19 119.750 119.75 118.2000 118.63 11584382.0                            753.0310       1.0
```

Stored nearby events:

```text
event_table    Ex-Date  Dividend Amount Dividend Type
  dividends 2024-11-14             0.99  Regular Cash
```

