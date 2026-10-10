# Annual expanding versus five-year rolling comparison

Runs: `runs/annual_expanding_rolling_50k_20260918`.

60 fits: evaluation years 2013–2022, two history choices, seeds 101/202/303. The cache stops October 5, 2022, so 2022 is explicitly partial. 2023 onward is untouched. This is historical development walk-forward evaluation, not live trading.

Expanding starts in 1997; rolling starts January 1 five years before evaluation. Each fit draws 100 time-stratified dates and 500 stocks per date (50,000 input-eligible candidates); unknown labels are excluded from training. Training formation ends 44 calendar-session positions before the first evaluation formation. This ensures the final five-day training outcome precedes the earliest input underlying MA20 in the first evaluation chart. Annual models are fixed throughout each evaluation year.

Binary absolute-direction labels use next-session open to fifth-session close: entry close/open multiplied by subsequent close-to-close TRI. Entry-day overnight return and entry-day dividend entitlement are excluded. Historical adjusted OHLC remains a research proxy, not a certified fill price. Missing forward outcomes remain unknown.

20 epochs exactly, no validation or early stopping, final checkpoint only. ChartCNN I20/MA20/volume, Adam 1e-5, batch 256. Three workers run seeds in parallel; each processes annual history choices sequentially. Checkpoints and per-epoch metrics are saved, with W&B logging. Each annual evaluation uses the same 500 stocks per five-session formation for every method and seed.

The preflight validates temporal separation and identical evaluation frames across the six fits in each year. Individual results appear inside each job directory; aggregate comparison.csv is written after all workers exit. Available-outcome metrics retain the inherited missing-return limitation. Costs are excluded by user instruction.
