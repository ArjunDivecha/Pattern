# Canonical research data baseline

Use `data/frozen/CURRENT.json` under the Pattern repository to resolve the frozen daily database. The current snapshot is `pattern_daily_v1_20260918`. Its README and manifest are the final data definition for this version; earlier audit drafts are superseded for operational use.

All experiments must record the dataset ID and manifest checksum, along with independent lookback, forecast horizon and moving-average settings. The original production training/backtest loaders have not been migrated in this data-freeze step. They must explicitly consume this snapshot before a new research run; legacy `data/r1000_*` files are not this baseline.

`freeze.py` builds a new, self-contained snapshot from existing local inputs and refuses to overwrite its version. `verify.py` checks the delivered snapshot without querying GDM. `CONTRACT.md` is the contract copied into the frozen snapshot.
