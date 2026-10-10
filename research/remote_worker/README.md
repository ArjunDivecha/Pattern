# MacBook Air worker

SSH: `air@arjuns-macbook-air.taild3d657.ts.net` (100.112.25.121).
Remote working directory: `/Users/air/pattern-worker`.
M5 / ARM64, 32 GiB memory. Python 3.14.7 virtualenv in `.venv`, installed with isolated bootstrap uv. Core versions matched to main host: torch 2.11.0, numpy 2.4.4, pandas 3.0.2, pydantic 2.13.3. MPS availability verified.

`benchmark.py` reads a compact images.npy and sample.parquet. It runs three 50k-candidate binary CNN training epochs, with no validation, metrics selection or production checkpoint. This is a throughput test only. No credentials were copied; W&B is installed but not authenticated on the Air.

Launch from remote directory with `caffeinate -i .venv/bin/python -u benchmark.py`. Keep laptop plugged in and lid open. Outputs `benchmark.json`. The benchmark is not a completed study fit; all official jobs remain on the main Mac until separately assigned to prevent duplication.
