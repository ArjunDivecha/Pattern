# Horizon study v1 — pilot and grid specification

Primary evaluation: equal-weight top and bottom n-tile forward returns separately, and their difference. Deciles are primary; quintiles and top/bottom 5% are prespecified sensitivity views. Daily cross-sectional Spearman IC and explicit non-overlapping holding-period spread Sharpe support screening. Never select models solely on peak Sharpe.

Grid: image lookback [5,10,20,40,60,120] × forward horizon [1,5,10,20,40,60] sessions. Daily bars, daily signals, three pixels per bar, height 64. MA length defaults to lookback, consuming 2L-1 sessions; independently configurable. No temporal compression in this stage.

Frozen source: data/frozen/CURRENT.json. Record resolved dataset ID and manifest checksum. Numerical input screening is historical only. Preserve all valid-input test scores, including missing forward outcomes; assign tiles before examining outcomes. A tile-date with any unknown outcome is unresolved, not renormalized. Binary direction/CE is the baseline target and loss; multi-class/ranking comparisons are later experiments.

Pilot only: I10/R5, MA10, deterministic stock sample drawn from valid bars at the first input date; 120 stocks, one seed, reduced [8,16] two-block CNN, two epochs, CPU. A small operational test, not signal evidence or model selection. Use chronological disjoint train/validation/test windows and enforce max earlier target date < min later input start. Train-only normalization and class balancing. No random time splits. The pilot uses 2020 and does not reserve an ultimate study holdout.

Report all ten tiles, top/bottom/spread, IC, missing outcomes, counts and quintile/5% sensitivity. For a supporting holding-period simulation, form equal-weight $1 long and $1 short cohorts every H calendar sessions and hold to t+H; arithmetic TRI returns. No overlapping-return annualization. Spread Sharpe uses sqrt(252/H) on non-overlapping H-session spreads, explicitly on $1-per-leg exposure; not NAV CAGR. Cost sensitivity assumes 10bp per dollar traded per transaction, or 40bp per complete long/short cohort. Borrow costs and next-open execution are not modeled. Missing tile outcomes block the corresponding portfolio statistic.

Before the full 36-cell study: freeze matched evaluation samples, full architecture and training budget, seed list, execution and cost assumptions. The pilot does not launch or approve that full compute run. Final selection should emphasize stable top/bottom separation by year, tail contributions and costs; do not cherry-pick sensitivity tails or test years.


## Annual expanding-window schedule (implemented)

`walkforward_config.json` and `annual_schedule.csv` define the shared development schedule. Source input history begins in 1996; the first eligible common training signal is 1996-12-09 after 239 sessions of warm-up. Annual development test years are 2010–2022. Reserve 2023–2025 for final evaluation after choosing the specification; do not inspect their prediction outcomes during development. Exclude partial 2026 as a test year. This reserve is for this new study, not a claim that those dates have never appeared in earlier project research.

Use a 252-session validation block. Before each later split, remove 298 earlier signal sessions, protecting the largest grid history (239 sessions) and horizon (60). Earlier last target is strictly before later first input. Apply this conservative schedule to all grid cells, even shorter ones. Consequently validation is earlier than the immediately preceding calendar year and training data is less recent. This is the cost of requiring fully disjoint input/target intervals across the entire grid; it is not a claim that every alternative validation design requires such a long gap. A future change to that convention requires a new schedule version before comparing results.

Development signals in 2022 stop on October 5 so that even 60-session outcomes end before the reserved block. Other development test years retain the full year's calendar dates. Missing stock histories can still change usable stock samples; common-stock/date comparison remains a full-grid implementation requirement.

Successful operational run: `runs/walkforward_smoke_20260918T073832Z`. Two separate fits for 2010 and 2011 with sparse training/validation date samples and only the first 30 signal sessions per test year. 100 securities chosen from information present on the initial 1996 date. This is a machinery check, not the full-universe study; training history expands in year two. Saved models reproduce predictions after reload. Combined scores have unique security/date keys. Neither reserved-year prices nor outcomes were loaded by this run.

## Evaluation priority update

User-directed gross-only phase: ignore transaction costs and implementability for now. Lead with top/bottom n-tile separation, matched-date legs, full decile ordering, consistency through time and stock/date concentration. Rank IC and gross Sharpe are supporting diagnostics. Keep earlier cost-sensitivity artifacts as historical records; new gross reports do not calculate costs. Missing outcomes and questionable source observations remain relevant to signal validity.

The next controlled extension scores all sessions of 2010 and 2011 with the existing I10/R5 checkpoints and the same initial-date stock sample. It changes evaluation length only. Separate the previously inspected first 30 sessions from the remaining dates. This is not a new training run or a full-universe study.
