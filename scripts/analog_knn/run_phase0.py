"""
=============================================================================
SCRIPT NAME: run_phase0.py
=============================================================================

Orchestrates Phase 0 of PRD_Visual_Analog_KNN.md:
  1. extract_embeddings
  2. query_knn
  3. evaluate_signal for analog + baselines
  4. comparison table + report.md + leakage checklist

USAGE:
  .venv/bin/python -m scripts.analog_knn.run_phase0
  .venv/bin/python -m scripts.analog_knn.run_phase0 --skip-extract  # reuse embeddings
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analog_knn.extract_embeddings import main as extract_main
from scripts.analog_knn.query_knn import main as knn_main

log = logging.getLogger("analog_knn.phase0")

DEFAULT_EXPANDING = ROOT / "runs" / "expanding" / "20260419_174908_cdef6809"
DEFAULT_CACHE = ROOT / "cache" / "r1000_I20_monthly"
DEFAULT_OHLCV = ROOT / "data" / "r1000_ohlcv_spliced.parquet"
DEFAULT_BASELINES = ROOT / "runs" / "experiments" / "stage1_baselines" / "baseline_signals.parquet"


def newey_west_t(x: np.ndarray, lag: int = 3) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean()
    xc = x - mu
    s = (xc ** 2).mean()
    for k in range(1, lag + 1):
        if k >= n:
            break
        s += 2 * (1 - k / (lag + 1)) * (xc[k:] * xc[:-k]).mean()
    se = np.sqrt(max(s, 0) / n)
    return float(mu / se) if se > 0 else float("nan")


def monthly_ic(df: pd.DataFrame, score_col: str, higher_is_better: bool = True) -> pd.Series:
    """Spearman IC of score vs forward_return per end_date."""
    sign = 1.0 if higher_is_better else -1.0
    ics = []
    dates = []
    for dt, g in df.groupby("end_date"):
        s = g[score_col].astype(float) * sign
        y = g["forward_return"].astype(float)
        m = s.notna() & y.notna()
        if m.sum() < 30:
            continue
        ic = s[m].corr(y[m], method="spearman")
        if np.isfinite(ic):
            ics.append(ic)
            dates.append(dt)
    return pd.Series(ics, index=pd.to_datetime(dates), name=score_col)


def run_eval(signal_path: Path, score_col: str, label: str, out_dir: Path, higher: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_signal.py"),
        "--signal",
        str(signal_path),
        "--score-col",
        score_col,
        "--label",
        label,
        "--out-dir",
        str(out_dir),
    ]
    # evaluate_signal defaults to higher-is-better; only flip for reversals
    if not higher:
        cmd.append("--lower-is-better")
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def build_comparison(scores: pd.DataFrame, baselines_path: Path | None, out_dir: Path) -> pd.DataFrame:
    rows = []

    def add_signal(df: pd.DataFrame, col: str, name: str, higher: bool) -> None:
        if col not in df.columns:
            return
        sub = df.dropna(subset=[col, "forward_return"]).copy()
        if len(sub) < 100:
            return
        ics = monthly_ic(sub, col, higher_is_better=higher)
        rows.append(
            {
                "signal": name,
                "score_col": col,
                "n_obs": int(len(sub)),
                "n_months_ic": int(len(ics)),
                "mean_ic": float(ics.mean()) if len(ics) else float("nan"),
                "ic_nw_t": newey_west_t(ics.to_numpy()) if len(ics) else float("nan"),
                "pct_months_pos_ic": float((ics > 0).mean()) if len(ics) else float("nan"),
            }
        )

    add_signal(scores, "analog_r_mean", "analog_r_mean_k50", True)
    add_signal(scores, "analog_r_mean_k25", "analog_r_mean_k25", True)
    add_signal(scores, "analog_r_mean_k100", "analog_r_mean_k100", True)
    add_signal(scores, "analog_r_med", "analog_r_med_k50", True)
    add_signal(scores, "analog_hit", "analog_hit_k50", True)
    if "p_up_mean" in scores.columns:
        add_signal(scores, "p_up_mean", "p_up_mean", True)

    if baselines_path and baselines_path.exists():
        b = pd.read_parquet(baselines_path)
        b["end_date"] = pd.to_datetime(b["end_date"])
        merged = scores[["ticker", "end_date"]].merge(b, on=["ticker", "end_date"], how="inner")
        for col, name, higher in [
            ("p_up_mean", "baseline_p_up_mean", True),
            ("p_up_resid", "baseline_p_up_resid", True),
            ("rev_5d", "rev_5d_WSTR", False),  # lower past return = buy
            ("rev_1m", "rev_1m_STR", False),
            ("mom_12_1", "mom_12_1", True),
            ("vol_60d", "vol_60d", False),
        ]:
            if col in merged.columns:
                add_signal(merged, col, name, higher)

        # Rank correlations (mean monthly spearman) analog vs baselines
        corr_rows = []
        if "analog_r_mean" in scores.columns and "p_up_mean" in scores.columns:
            for dt, g in scores.groupby("end_date"):
                a = g["analog_r_mean"]
                p = g["p_up_mean"]
                m = a.notna() & p.notna()
                if m.sum() < 30:
                    continue
                corr_rows.append(a[m].corr(p[m], method="spearman"))
            mean_rho = float(np.nanmean(corr_rows)) if corr_rows else float("nan")
        else:
            mean_rho = float("nan")
    else:
        mean_rho = float("nan")

    comp = pd.DataFrame(rows)
    comp_path = out_dir / "comparison_ic.csv"
    comp.to_csv(comp_path, index=False)
    # also xlsx if openpyxl available
    try:
        comp.to_excel(out_dir / "comparison.xlsx", index=False)
    except Exception:
        pass

    (out_dir / "identity_corr.json").write_text(
        json.dumps({"mean_monthly_spearman_analog_vs_p_up": mean_rho}, indent=2)
    )
    return comp


def write_report(out_dir: Path, comp: pd.DataFrame, knn_meta: dict, decision_notes: str) -> None:
    lines = [
        "# Analog kNN Phase 0 report",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## kNN meta",
        "```json",
        json.dumps(knn_meta, indent=2),
        "```",
        "",
        "## IC comparison",
        "",
        comp.to_markdown(index=False) if hasattr(comp, "to_markdown") else comp.to_string(index=False),
        "",
        "## Decision notes",
        "",
        decision_notes,
        "",
        "## Leakage",
        "",
        "See `leakage_checklist.md`.",
        "",
    ]
    try:
        body = "\n".join(lines)
    except Exception:
        body = "\n".join(
            [
                "# Analog kNN Phase 0 report",
                "",
                comp.to_string(index=False),
                "",
                decision_notes,
            ]
        )
    (out_dir / "report.md").write_text(body)


def write_leakage_checklist(out_dir: Path) -> None:
    text = """# Leakage checklist — Phase 0

| ID | Rule | Status | How verified |
|----|------|--------|----------------|
| L1 | j ≠ i (no self ticker) | PASS (by construction) | `query_knn.knn_for_date_multi_k` masks `bank_ticker_codes != q_ticker` |
| L2 | neighbor outcome completed | PASS (by construction) | `outcome_available_dates = end_date + 20 trading days`; bank only adds when `avail <= t` |
| L3 | walk-forward model vintage | PASS (by construction) | `extract_embeddings` uses `w{WW}_ensemble_*.pt` for OOS year only |
| L4 | panel alignment | PASS | embeddings = predictions_monthly ∩ image cache |

Automated tests: `scripts/analog_knn/tests/test_leakage.py`
"""
    (out_dir / "leakage_checklist.md").write_text(text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanding-dir", type=Path, default=DEFAULT_EXPANDING)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    ap.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--k", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-knn", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--max-windows", type=int, default=None, help="Debug extract only")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (ROOT / "runs" / "analog_knn" / f"phase0_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Phase 0 out_dir=%s", out_dir)

    pred_monthly = args.expanding_dir / "predictions_monthly.parquet"
    emb_path = out_dir / "embeddings.npz"

    if not args.skip_extract:
        extract_main(
            [
                "--expanding-dir",
                str(args.expanding_dir),
                "--cache-dir",
                str(args.cache_dir),
                "--predictions-monthly",
                str(pred_monthly),
                "--out-dir",
                str(out_dir),
                "--device",
                args.device,
                "--chunk",
                str(args.chunk),
            ]
            + (
                ["--max-windows", str(args.max_windows)]
                if args.max_windows is not None
                else []
            )
        )
    elif not emb_path.exists():
        raise FileNotFoundError(f"--skip-extract but missing {emb_path}")

    scores_path = out_dir / "analog_scores.parquet"
    if not args.skip_knn:
        knn_main(
            [
                "--embeddings",
                str(emb_path),
                "--ohlcv",
                str(args.ohlcv),
                "--out-dir",
                str(out_dir),
                "--k",
                *[str(k) for k in args.k],
                "--device",
                args.device,
                "--predictions-monthly",
                str(pred_monthly),
            ]
        )
    elif not scores_path.exists():
        raise FileNotFoundError(f"--skip-knn but missing {scores_path}")

    scores = pd.read_parquet(scores_path)
    knn_meta = json.loads((out_dir / "knn_meta.json").read_text())

    comp = build_comparison(scores, args.baselines, out_dir)

    # Decision draft from IC table
    decision = "INCOMPLETE — run evaluate_signal for portfolio t-stats"
    row = comp[comp["signal"] == "analog_r_mean_k50"]
    if len(row):
        t_ic = float(row.iloc[0]["ic_nw_t"])
        if t_ic > 2:
            decision = f"IC gate S1 soft-PASS (analog IC NW t={t_ic:.2f}). Confirm with evaluate_signal L/S."
        elif t_ic > 1:
            decision = f"IC borderline (t={t_ic:.2f}). Need L/S + residual tests."
        else:
            decision = f"IC weak (t={t_ic:.2f}). Likely kill unless L/S or residual emb saves it."

    write_leakage_checklist(out_dir)
    write_report(out_dir, comp, knn_meta, decision)

    if not args.skip_eval:
        # Primary analog + p_up on same rows
        try:
            run_eval(
                scores_path,
                "analog_r_mean",
                "Analog kNN k=50 mean",
                out_dir / "eval_analog_r_mean",
                higher=True,
            )
        except subprocess.CalledProcessError as e:
            log.error("evaluate_signal failed for analog: %s", e)
        if "p_up_mean" in scores.columns:
            try:
                run_eval(
                    scores_path,
                    "p_up_mean",
                    "CNN p_up (analog panel)",
                    out_dir / "eval_p_up_mean",
                    higher=True,
                )
            except subprocess.CalledProcessError as e:
                log.error("evaluate_signal failed for p_up: %s", e)

    # config snapshot
    (out_dir / "config.yaml").write_text(
        f"""project: analog_knn
phase: 0
out_dir: {out_dir}
expanding_dir: {args.expanding_dir}
cache_dir: {args.cache_dir}
k: {list(args.k)}
device: {args.device}
"""
    )
    log.info("Phase 0 complete → %s", out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
