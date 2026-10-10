# Monthly Pattern side study

The experiment predicts the next calendar month's total return direction from 12 completed calendar-month charts. Scores rank stocks into deciles. It uses only the existing frozen daily database; no Bloomberg or other vendor calls.

Each monthly bar takes the first, maximum, minimum and last **stored adjusted daily close**, plus summed stored daily volume. These are close-derived bars, not intraday OHLC extrema. MA200 is the trailing 200 daily closes at each month-end. The renderer draws these 12 precomputed MA points in the same price coordinate system and includes the MA in its vertical bounds. It does not re-adjust prices via TRI. Future returns are gross-dividend TRI endpoint ratios, without re-adding dividends.

Complete months and contiguous calendar/identity segments are required for inputs. Invalid daily closes or a missing session invalidate the affected MA history. Unknown forward outcomes remain in the sampled panel and decile assignment; only supervised loss excludes missing labels. No price floor, clipping, index-membership filter or cost model is added. Monthly volume uses the frozen volume convention, whose historical split adjustment is not independently certified.

## Fixed development sample

- Training: 1997–2011, 100,000 candidates sampled evenly across 180 months.
- Validation: 2014–2015, 20,000 candidates across 24 months; selects checkpoints.
- Comparison: January 2018–November 2022, 500 stocks per month, 29,500 candidates; next-month targets end in December 2022.
- 2023 onward stays reserved. Actual MA-history start dates are checked against the previous partition's latest target, protecting the extra history before the first chart month.

Images are uint8, 1×64×36, with three pixel columns per monthly bar. The same three-block ChartCNN is trained from scratch with natural class balance, train-only normalization, Adam at 1e-5, batch 256, seeds 0/1/2, up to 50 epochs, patience 5. Each epoch has full model/optimizer/RNG checkpoints. W&B logs training and validation loss/accuracy and final monthly evaluation results.

Evaluation uses top/bottom deciles, spread, all deciles, monthly rank IC, and nonoverlapping monthly Sharpe annualized by sqrt(12). Observed-only statistics renormalize known outcomes within already-assigned tiles and can be biased by missing outcomes. Strict Sharpe is unavailable if any extreme-leg cohort has unknown outcomes. Formation uses month-end close as a gross signal study, not an execution assumption. Yearly results are grouped by formation year.

Two historical-feature baselines use exactly the same comparison stock-months: adjusted-close momentum from t−12 to t−1 month-end, and current close / MA200 − 1.

## Commands

```sh
.venv/bin/python research/monthly_lab/build.py --run-dir runs/monthly_M12_R1_v1_20260918
.venv/bin/python research/monthly_lab/train.py --run-dir runs/monthly_M12_R1_v1_20260918
```

After stopping an interrupted training process, the second command accepts `--resume`; the tested shared trainer restores each seed's last complete epoch. Old weights-only files are not treated as a resumable optimizer state. The monthly run imports the existing trainer without changing the currently running daily experiment.

Build outputs include monthly bars, eligibility counts by month, sample manifest, yearly sample coverage, compact image pack and example charts. Training writes scores, per-stock decile assignments, monthly/yearly metrics, simple-baseline results and a final report automatically.
