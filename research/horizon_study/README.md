# Frozen-data pilot

Successful run: `runs/frozen_pilot_20260918T073242Z` (supersedes the earlier attempt whose Markdown formatting failed).

Run `.venv/bin/python research/horizon_study/run_pilot.py --lookback 10 --horizon 5 --ma 10` from the repository root. The pilot accepts independent dimensions within its deliberately short split schedule; `pattern.data.frozen.samples` supports the broader study dimensions. Legacy `pattern train` still uses its original loader; use this explicit frozen-data runner for the new study.

The run contains source snapshots, configuration, versions, selected securities, input/target dates per split, checkpoint, epoch losses, all out-of-sample scores (including unknown outcomes), all-ten-decile results, quintile and 5% tails, IC and non-overlapping cohort cost sensitivity. `report.md` leads with the decile results. `verification.json` records the successful run and checks. `SPEC.md` documents the staged study and decisions still needed before the full grid.

The pilot tests the pipeline; its small model, short test period, and six holding cohorts cannot establish investment performance. The source, outcome and portfolio-accounting caveats remain visible in its report.
