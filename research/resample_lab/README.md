# Independent training sample study

Six additional fits: daily I20/R5 and monthly M12/R1, each using fixed sample/model seeds 101, 202, 303. Each fit selects 100,000 input-eligible training candidates; unknown labels are excluded by the unchanged trainer. Both sampling and initialization vary, so differences measure their combined effect, not a pure sampling effect.

Validation and comparison rows, labels, and image bytes are checked against the original runs. Existing split guards apply. Daily draws select 200 time-stratified dates and 500 stocks per date; monthly draws span all 180 original training months. Samples may overlap naturally. Each uses the existing architecture, MA/volume, learning rate, 50-epoch maximum, patience 5, resumable checkpoints, and W&B logging. No sample refresh between epochs.

Run artifacts: `runs/resample_stability_20260918`. Each frequency has a queue status JSON and log, per-draw audit, and pairwise overlap audit. One queue per frequency waits for its existing training process, then runs its three new fits sequentially. A failed fit stops that frequency's queue and records failure. Original jobs are not changed.
