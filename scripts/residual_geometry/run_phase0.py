"""
Phase 0 orchestrator: build residual/geometry/FOMC signals + evaluate_signal cards + report.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.residual_geometry.build_signals import main as build_main

log = logging.getLogger("residual_geometry.phase0")


def run_eval(signal: Path, col: str, label: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_signal.py"),
        "--signal", str(signal),
        "--score-col", col,
        "--label", label,
        "--out-dir", str(out),
    ]
    log.info("%s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (ROOT / "runs" / "residual_geometry" / f"phase0_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sig = out_dir / "residual_signals.parquet"
    if not args.skip_build:
        build_main(["--out-dir", str(out_dir)])
    elif not sig.exists():
        raise FileNotFoundError(sig)

    if not args.skip_eval:
        evals = [
            ("p_up_mean", "CNN p_up"),
            ("p_up_resid", "p_up residual vs baselines"),
            ("p_up_resid_geom", "p_up residual vs geometry"),
            ("p_up_resid_full", "p_up residual baselines+geometry"),
            ("geom_score", "Dixon geometry blend"),
            ("rlcc_or", "Dixon RLCC oriented"),
            ("p_up_resid_nofomc", "p_up residual no-FOMC months"),
        ]
        for col, label in evals:
            try:
                run_eval(sig, col, label, out_dir / f"eval_{col}")
            except subprocess.CalledProcessError as e:
                log.error("eval failed %s: %s", col, e)

    # report
    ic = pd.read_csv(out_dir / "feature_ic.csv") if (out_dir / "feature_ic.csv").exists() else None
    cards = {}
    for p in sorted(out_dir.glob("eval_*/signal_card.json")):
        cards[p.parent.name] = json.loads(p.read_text())

    lines = [
        "# Residual + Geometry + FOMC — Phase 0 report",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## IC table",
        "",
        "```",
        ic.to_string(index=False) if ic is not None else "(missing)",
        "```",
        "",
        "## Portfolio ($5 floor, simple returns)",
        "",
        "| Signal | LS decile CAGR | LS decile t | LS 50/50 CAGR | LS 50/50 t |",
        "|--------|---------------:|------------:|--------------:|-----------:|",
    ]
    for name, card in cards.items():
        flo = card.get("variants", {}).get("floored", {})
        d = flo.get("ls_decile_full", {})
        h = flo.get("ls_5050_full", {})
        lines.append(
            f"| {card.get('signal', name)} | {d.get('cagr_pct', float('nan'))} | "
            f"{d.get('nw_t', float('nan'))} | {h.get('cagr_pct', float('nan'))} | "
            f"{h.get('nw_t', float('nan'))} |"
        )

    # decision
    decision = "See IC and portfolio tables."
    if ic is not None and "p_up_resid" in ic["signal"].values:
        t_resid = float(ic.loc[ic.signal == "p_up_resid", "IC_t_NW"].iloc[0])
        if t_resid > 2:
            decision = f"S1 soft-PASS on IC (p_up_resid t={t_resid:.2f}). Check L/S; export as feature if L/S also ok."
        elif t_resid > 1.5:
            decision = f"Borderline residual IC t={t_resid:.2f}. Rely on L/S cards."
        else:
            decision = f"S1 FAIL on IC (p_up_resid t={t_resid:.2f}). Residual path weak on this panel."

    lines += ["", "## Decision", "", decision, ""]
    (out_dir / "report.md").write_text("\n".join(lines))
    log.info("Phase 0 done → %s", out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
