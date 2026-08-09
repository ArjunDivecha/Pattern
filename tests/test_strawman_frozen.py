"""
=============================================================================
SCRIPT NAME: tests/test_strawman_frozen.py
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/stage1b_eval/combo_resid_wstr/signal_card.xlsx
      sheet `floored_monthly`, column `ls_decile` -- the frozen strawman series

OUTPUT FILES: none

DESCRIPTION:
Enforces PREREGISTRATION.md section 2 in code rather than in prose.

Stage 2 compares every candidate against a frozen baseline: the `combo`
(neutralized CNN + weekly reversal) decile long/short book. If that series
silently changes -- a rescored signal, a different price panel, an edited
harness -- then every spanning regression against it is measuring the wrong
thing, and nothing in the comparison would look wrong.

So the baseline carries a checksum, and this test is what actually checks it.
A pre-registration nobody can verify is a promise, not a control.

Skips (rather than fails) when the artefact is absent, so a fresh clone without
run outputs still has a green suite; the assertions bite whenever the file is
present, which is whenever it matters.
=============================================================================
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STRAWMAN = (ROOT / "runs" / "experiments" / "stage1b_eval" / "combo_resid_wstr"
            / "signal_card.xlsx")

# Frozen 2026-08-09, PREREGISTRATION.md section 2
EXPECTED_SHA16 = "59f55f36169346ca"
EXPECTED_MONTHS = 282
# (window label, start, end, net@8bp, NW t) -- tolerance is generous because
# only a real change to the series should trip this, not float noise
EXPECTED_NET = {
    "full":  (None,         None,         7.64, 3.66),
    "2005+": ("2005-01-01", None,         2.40, 1.86),
}
TURNOVER = 0.8911
HALF_SPREAD_BP = 8.0


def _series() -> pd.Series:
    d = pd.read_excel(STRAWMAN, sheet_name="floored_monthly")
    d["end_date"] = pd.to_datetime(d["end_date"])
    return d.set_index("end_date")["ls_decile"].dropna()


def _nwt(x: np.ndarray, lag: int = 3) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); mu = x.mean(); xc = x - mu
    s = (xc ** 2).mean()
    for k in range(1, lag + 1):
        if k >= n:
            break
        s += 2 * (1 - k / (lag + 1)) * (xc[k:] * xc[:-k]).mean()
    return float(mu / np.sqrt(max(s, 0) / n))


pytestmark = pytest.mark.skipif(
    not STRAWMAN.exists(),
    reason="strawman artefact absent (run outputs are gitignored)")


def test_strawman_checksum_and_length():
    s = _series()
    sha = hashlib.sha256(np.round(s.to_numpy(), 10).tobytes()).hexdigest()[:16]
    assert len(s) == EXPECTED_MONTHS, (
        f"strawman has {len(s)} months, frozen at {EXPECTED_MONTHS}. "
        "The baseline moved — every Stage-2 comparison against it is void "
        "until PREREGISTRATION.md section 2 is re-frozen deliberately.")
    assert sha == EXPECTED_SHA16, (
        f"strawman checksum {sha} != frozen {EXPECTED_SHA16}. "
        "See PREREGISTRATION.md section 9: a moved baseline invalidates results.")


@pytest.mark.parametrize("label", list(EXPECTED_NET))
def test_strawman_headline_numbers(label):
    start, end, exp_net, exp_t = EXPECTED_NET[label]
    s = _series()
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    if end:
        s = s[s.index <= pd.Timestamp(end)]
    drag = 2 * TURNOVER * HALF_SPREAD_BP / 1e4
    net = ((1 + s - drag).prod() ** (12 / len(s)) - 1) * 100
    assert net == pytest.approx(exp_net, abs=0.05), \
        f"{label} net {net:.2f}% != frozen {exp_net}%"
    assert _nwt(s.to_numpy()) == pytest.approx(exp_t, abs=0.05), \
        f"{label} NW t {_nwt(s.to_numpy()):.2f} != frozen {exp_t}"


def test_preregistration_document_present():
    """The protocol must ship with the code it constrains."""
    doc = ROOT / "PREREGISTRATION.md"
    assert doc.exists(), "PREREGISTRATION.md is missing"
    txt = doc.read_text()
    for marker in (EXPECTED_SHA16, "Promotion rule", "lockbox",
                   "nky_ohlcv_database"):
        assert marker.lower() in txt.lower(), \
            f"PREREGISTRATION.md no longer contains '{marker}'"
