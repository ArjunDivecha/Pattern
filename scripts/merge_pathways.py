"""
=============================================================================
SCRIPT NAME: merge_pathways.py
=============================================================================

INPUT FILES:
- {expanding_run}/predictions.parquet
- {rolling_run}/predictions.parquet

OUTPUT FILES:
- {out_dir}/pathway_comparison.parquet   — one row per (ticker, end_date)
  with p_up_mean_expanding, p_up_mean_rolling, std_*, rank_*, decile_*,
  forward_return, label (the label is identical across pathways, kept once).
- {out_dir}/pathway_comparison_summary.csv — high-level diagnostic: per-window
  correlation, rank correlation, decile disagreement rate.

DESCRIPTION:
Merges the expanding- and rolling-pathway prediction parquets on
(ticker, end_date) so downstream analysis can compare ensemble-mean
probability, rank, and decile assignment for the same (stock, formation date)
from the two retraining schemes.

Ticker ID is preserved (no Bloomberg metadata enrichment here — that can be
layered on later using this parquet as a key).

USAGE:
  python scripts/merge_pathways.py \
      --expanding /data/Pattern/runs/expanding/<ts_hash> \
      --rolling   /data/Pattern/runs/rolling/<ts_hash>  \
      --out-dir   /data/Pattern/runs/comparison
=============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEEP_COLS = ["ticker", "end_date", "forward_return", "label",
             "p_up_mean", "p_up_std", "logit_down_mean", "logit_up_mean",
             "rank_pct", "decile", "window"]


def _load_preds(run_dir: Path, suffix: str) -> pd.DataFrame:
    path = run_dir / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions at {path}")
    df = pd.read_parquet(path)
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[KEEP_COLS].copy()
    rename = {c: f"{c}_{suffix}" for c in df.columns
              if c not in ("ticker", "end_date", "forward_return", "label")}
    return df.rename(columns=rename)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanding", required=True, type=Path)
    ap.add_argument("--rolling",   required=True, type=Path)
    ap.add_argument("--out-dir",   required=True, type=Path)
    args = ap.parse_args()

    exp = _load_preds(args.expanding, "expanding")
    rol = _load_preds(args.rolling,   "rolling")

    # forward_return / label should agree; keep one copy for each but check.
    merged = exp.merge(
        rol,
        on=["ticker", "end_date", "forward_return", "label"],
        how="outer",
        indicator=True,
    )
    n_both = int((merged["_merge"] == "both").sum())
    n_exp_only = int((merged["_merge"] == "left_only").sum())
    n_rol_only = int((merged["_merge"] == "right_only").sum())
    print(f"Merged:  both={n_both:,}  expanding_only={n_exp_only:,}  "
          f"rolling_only={n_rol_only:,}")
    merged = merged.drop(columns="_merge")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = args.out_dir / "pathway_comparison.parquet"
    merged.to_parquet(out_parquet, index=False)
    print(f"wrote {out_parquet}  rows={len(merged):,}")

    # Diagnostic summary — only on rows present in both pathways.
    both = merged.dropna(subset=["p_up_mean_expanding", "p_up_mean_rolling"]).copy()
    if len(both) == 0:
        print("No overlap — skipping summary.")
        return

    def _summary(g: pd.DataFrame) -> pd.Series:
        pe = g["p_up_mean_expanding"].to_numpy()
        pr = g["p_up_mean_rolling"].to_numpy()
        corr = float(np.corrcoef(pe, pr)[0, 1]) if len(g) > 1 else np.nan
        de = g["decile_expanding"]
        dr = g["decile_rolling"]
        mask = de.notna() & dr.notna()
        if mask.any():
            disagree = float((de[mask].astype(int) != dr[mask].astype(int)).mean())
        else:
            disagree = float("nan")
        return pd.Series({
            "n": len(g),
            "corr_p_up": corr,
            "decile_disagreement": disagree,
        })

    summary = both.groupby("end_date").apply(_summary).reset_index()
    out_csv = args.out_dir / "pathway_comparison_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  dates={len(summary)}  "
          f"mean corr={summary['corr_p_up'].mean():.3f}  "
          f"mean decile disagree={summary['decile_disagreement'].mean():.3f}")


if __name__ == "__main__":
    main()
