# Fast development lab

The old annual training queue remains stopped. This directory starts the small-sample development workflow.

## Hardware benchmark

`benchmark_mac.py` measures actual forward/backward/Adam updates on the Mac MPS GPU using the full 708,418-parameter CNN. It selects 100,000 observations: 500 stocks on each of 200 dates stratified across quarters in 1997–2022, with sample seed 71. Images come from the existing frozen I20/MA20 cache; R5 direction supplies the timing labels. No external data is fetched.

The benchmark compares disk-backed, host-memory and GPU-resident inputs, batch sizes 128–1024, CPU thread counts, synchronization frequency and autocast precision. Input/logit equivalence and three-step FP32 update equivalence are checked. Short screening runs are followed by four full 100,000-observation timing passes. If the selected setting equals baseline, these are repeated measurements of the same configuration, not an optimization gain.

Outputs live under `runs/mac_development_benchmark_*`: sample manifest, yearly coverage, parity checks, screening measurements, full-epoch timings and recommendation. Timing excludes sample preparation, image rendering, validation and checkpoint writing. Disk measurements may benefit from the operating-system file cache. Cases run sequentially, so thermal or background-load variation can confound small differences. Throughput does not establish GPU utilization percentage or optimal concurrent-model scheduling. AMP finite checks do not establish equivalent learning quality.

## Research experiments

The hardware sample is not a validated research split. Before comparing predictive quality, use separate chronological training and validation date pools with horizon-aware purging, sample many stocks on each selected date, fit normalization on training only, and preserve the reserved final period. Do not randomly split overlapping stock-date windows.

Compare lookback, forecast horizon and bar frequency independently. Reuse stock-date samples across compatible configurations and report coverage differences. Start with one seed for screening, repeat promising choices across seeds and periods. Evaluate top-versus-bottom n-tile spreads and the full n-tile profile first; use cross-sectional rank IC and Sharpe as supporting filters. Current research focuses on gross signal, without transaction costs.

## Executable development harness

Run with the project Python:

```sh
.venv/bin/python research/development_lab/run.py --config research/development_lab/config.json --run-dir runs/<new-name>
```

On this Mac, MPS execution requires the normal host environment rather than the restricted sandbox. The command fails instead of silently falling back to CPU. It requires a fresh output directory and does not resume or alter the stopped annual study.

The initial configuration uses:

| Partition | Period | Sampling | Role |
|---|---|---|---|
| Training | 1997–2012 | 200 stratified dates × 500 stocks | Gradient updates |
| Validation | July 2014–June 2016 | 40 stratified dates × 500 stocks | Checkpoint selection by loss |
| Comparison | 2018–early October 2022 | 500 stocks at each regularly scheduled formation date | Development comparison metrics |
| Reserved | 2023 onward | Not read for images or labels | Later independent evaluation |

The intervening gaps protect a common maximum 239-session history and 60-session label horizon. Both initial experiments share the same candidate training and validation stock-dates. Unknown outcomes are excluded only when a label is needed; this means the actual number of gradient-update observations can differ slightly by horizon. Samples use no forward-return eligibility filter. Class balance stays natural. Image normalization is fitted on the labeled training subset only.

The first run checks I20/R5 and I20/R20 for two epochs and one seed. It is a workflow check, not a sufficient training budget or architecture search. Their arithmetic spreads cover different holding periods and must not be compared as identical-period returns. Their nonoverlapping schedules also contain different numbers of observations. Results by year make that coverage visible.

### Metrics

- Top and bottom decile arithmetic returns, their spread, the full tile profile, cross-sectional Spearman IC and results by formation year.
- A regular H-session formation schedule, $1 long and $1 short, gross TRI endpoint P&L, with no costs and zero cash rate. Cohorts abut but do not overlap.
- Annualized holding-period Sharpe is `mean(spread) / sample_std(spread) * sqrt(252/H)`. This is not daily marked-to-market NAV Sharpe and can differ from it, especially across horizons.
- **Observed-only** returns reweight known outcomes after assigning all stocks to tiles. They are conditional diagnostics and can be biased by missing outcomes. The report separately discloses unknown fractions by tile and withholds **strict Sharpe** if any top/bottom cohort includes an unknown return. No missing outcome is set to zero, and no stock is removed before ranking because of its future return.
- Comparison results are for tuning and are not described as independent test results. Repeated experiments can overfit them; the reserved period remains separate.

### Controls available now

`config.json` controls training budget, seed list, learning rate, batch size, model channels/blocks/dropout and tile count. Each experiment selects a cached label horizon and can remove volume while keeping the OHLC price area unchanged. A new compatible cache can supply another image width/lookback; cache identity and source-manifest identity are checked.

The first cache contains daily I20 images with MA20 and R5/R20 labels. **MA removal, other lookbacks and weekly representations need new rendering packs; the current runner does not pretend that masking an existing MA image can recover OHLC-only input.** Their renderers and matched-sample coverage checks are the next extension. Seed stability requires multiple seeds; one-seed startup results do not measure it.

Each output contains config and source hashes, a fixed sample manifest, yearly coverage, compact images, training-only normalization, epoch logs, best checkpoints, reload-verified scores, per-year metrics and per-stock decile assignments. `comparison.csv` collects all completed fits; `status.json` records completion or failure.

## Longer I20/R5 run and live W&B

`long_I20_R5.json` runs seeds 0/1/2 freshly initialized, up to 50 epochs, with patience 5. It reuses the exact previous sample and images, retains the same optimizer/learning rate, and logs to the existing `arjun-divecha-dancing-elephant/pattern-cnn` project. W&B shows training/validation loss and accuracy each epoch, timing and learning rate; the comparison-period deciles, rank IC, Sharpe and yearly metrics are published after each fit, not used to select epochs.

Launch with automatic report generation:

```sh
.venv/bin/python research/development_lab/run_budget.py --config research/development_lab/long_I20_R5.json --run-dir runs/development_I20_R5_long_20260918
```

Each seed now writes:

- `last.pt`: atomic full checkpoint at the last completed epoch, including model, Adam state, epoch, loss history, patience state, RNG states, best weights and a data/config signature.
- `best_checkpoint.pt`: full state at the best validation epoch.
- `best.pt`: inference weights for that best epoch, retained for compatibility.
- `wandb.json`: the live run URL and stable run ID.

After stopping the process, rerun the same command with `--resume` to restart from `last.pt`. An interrupted partial epoch is replayed. The maximum epoch cap may be increased; other configuration and source changes are rejected. Early-stopped seeds retain their patience state. The trainer seeds sampling/dropout deterministically at each epoch boundary. Resume equivalence is tested on CPU and the Mac GPU; changing hardware/software is not a promise of bitwise equivalence.

Older weights-only checkpoints are not silently treated as resumable training state. Checkpoints remain local; W&B receives scalar and aggregate-table telemetry, not the raw stock panel or image pack.
