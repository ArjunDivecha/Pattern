"""
=============================================================================
SCRIPT NAME: neutralize_p_up.py
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/experiments/stage1_baselines/baseline_signals.parquet
      Built by scripts/baseline_gauntlet.py. One row per (ticker, end_date)
      with p_up_mean, forward_return (20d LOG), and the baselines.

OUTPUT FILES:
- {out_dir}/neutralized_signals.parquet   every neutralization variant as its
                                          own column, ready for
                                          scripts/evaluate_signal.py --score-col
- {out_dir}/fama_macbeth.xlsx             FM regression tables
- {out_dir}/neutralize_summary.json       machine-readable

DESCRIPTION:
Follow-up to Gate G1 (README addendum L). The gauntlet found that ranking on
p_up's residual after regressing out the baselines BEATS ranking on raw p_up
(+4.19%/yr t 3.01 vs +1.68%/yr t 1.05). That implies the CNN carries a small
genuine edge wrapped in harmful factor exposure. This script asks three
questions the portfolio test alone cannot answer:

1. FAMA-MACBETH -- is the incremental content statistically real?
   Per formation date, regress the forward return on p_up alone, then on p_up
   PLUS every baseline. Average the coefficients across months and test with a
   Newey-West t. If p_up's multivariate coefficient stays significant, it
   predicts returns that the baselines do not explain. This is the textbook
   incremental-value test and is stricter than any portfolio sort.
   All regressors are cross-sectionally rank-normalized to [-0.5, 0.5] each
   month, so coefficients are directly comparable across signals and immune to
   the very different raw scales (p_up ~0.5 +/- 0.02; vol ~0.4 +/- 0.3).

2. WHICH EXPOSURE IS DOING THE DAMAGE -- neutralize against subsets
   (vol only / rev_1m only / both / all) so the improvement can be attributed
   rather than assumed. The G1 write-up hypothesized the low-vol tilt; this
   tests it.

3. IS IT IMPLEMENTABLE -- the G1 residual used CONTEMPORANEOUS cross-sectional
   betas. That is not look-ahead (the forward return never enters, and every
   input is known at the formation date, so it is executable in real time), but
   the betas are fit on the same cross-section being ranked. The strictly
   causal alternative fits betas only on months < t. If the result survives
   that, it is robust to the objection.

USAGE:
  python3 scripts/neutralize_p_up.py --out-dir runs/experiments/stage1b_neutralize

NOTES:
- Portfolio scoring is deliberately delegated to scripts/evaluate_signal.py so
  every number stays on the one measuring stick.
- Returns used here are SIMPLE (expm1 of the stored 20-day log return), per
  README addendum K.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIG = (ROOT / "runs" / "experiments" / "stage1_baselines"
               / "baseline_signals.parquet")

# oriented so that HIGHER = expected buy, matching baseline_gauntlet.py
BASELINES = ["rev_1m", "rev_5d", "mom_12_1", "vol_60d"]
SIGN = {"rev_1m": -1, "rev_5d": -1, "mom_12_1": +1, "vol_60d": -1}

NEUTRALIZE_SETS = {
    "resid_vol":        ["vol_60d"],
    "resid_rev1m":      ["rev_1m"],
    "resid_vol_rev1m":  ["vol_60d", "rev_1m"],
    "resid_all":        BASELINES,
}


def newey_west_t(x: np.ndarray, lag: int = 3) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean(); xc = x - mu
    s = (xc ** 2).mean()
    for k in range(1, lag + 1):
        if k >= n: break
        s += 2 * (1 - k / (lag + 1)) * (xc[k:] * xc[:-k]).mean()
    se = np.sqrt(max(s, 0) / n)
    return float(mu / se) if se > 0 else float("nan")


def rank_norm(s: pd.Series) -> pd.Series:
    """Cross-sectional rank mapped to [-0.5, 0.5]. Scale-free and outlier-proof."""
    n = s.notna().sum()
    if n < 3:
        return pd.Series(np.nan, index=s.index)
    return s.rank(method="average", pct=True) - 0.5


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    if ok.sum() < X.shape[1] + 10:
        return None
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=Path, default=DEFAULT_SIG)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-names", type=int, default=50)
    ap.add_argument("--min-history-months", type=int, default=36,
                    help="months of history before causal betas are usable")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.signals)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["r_simple"] = np.expm1(df["forward_return"])
    print(f"rows {len(df):,}  months {df.end_date.nunique()}  "
          f"{df.end_date.min().date()} -> {df.end_date.max().date()}")

    # ── rank-normalize everything, per formation date ───────────────────────
    cols = ["p_up_mean"] + BASELINES
    for c in cols:
        oriented = df[c] * (SIGN.get(c, 1))
        df[f"rn_{c}"] = (oriented.groupby(df["end_date"], group_keys=False)
                         .apply(rank_norm) if False else
                         df.assign(_o=oriented).groupby("end_date")["_o"]
                         .transform(lambda s: s.rank(pct=True) - 0.5))

    dates = sorted(df["end_date"].unique())

    # ── 1. Fama-MacBeth ─────────────────────────────────────────────────────
    fm_uni, fm_multi = [], []
    for dt, g in df.groupby("end_date"):
        g = g.dropna(subset=["r_simple", "rn_p_up_mean"] + [f"rn_{b}" for b in BASELINES])
        if len(g) < args.min_names:
            continue
        y = g["r_simple"].to_numpy()

        Xu = np.column_stack([np.ones(len(g)), g["rn_p_up_mean"].to_numpy()])
        bu = ols(Xu, y)
        if bu is not None:
            fm_uni.append({"end_date": dt, "p_up": bu[1]})

        Xm = np.column_stack([np.ones(len(g)), g["rn_p_up_mean"].to_numpy()]
                             + [g[f"rn_{b}"].to_numpy() for b in BASELINES])
        bm = ols(Xm, y)
        if bm is not None:
            fm_multi.append({"end_date": dt,
                             **dict(zip(["p_up"] + BASELINES, bm[1:]))})

    uni = pd.DataFrame(fm_uni)
    multi = pd.DataFrame(fm_multi)

    def fm_table(d: pd.DataFrame, label: str) -> pd.DataFrame:
        out = []
        for c in [c for c in d.columns if c != "end_date"]:
            v = d[c].to_numpy()
            out.append({"spec": label, "regressor": c,
                        "mean_coef_pct": round(float(np.nanmean(v) * 100), 4),
                        "nw_t": round(newey_west_t(v), 2),
                        "months": int(np.isfinite(v).sum())})
        return pd.DataFrame(out)

    fm = pd.concat([fm_table(uni, "univariate"),
                    fm_table(multi, "multivariate")], ignore_index=True)
    print("\n=== 1. Fama-MacBeth: 20d forward SIMPLE return on rank-normalized signals ===")
    print("    coefficient = return spread (%) from bottom to top of the signal's rank range")
    print(fm.to_string(index=False))

    # ── 2. neutralization variants, contemporaneous betas ───────────────────
    print("\n=== 2. Building neutralization variants (contemporaneous betas) ===")
    for name, subset in NEUTRALIZE_SETS.items():
        vals = np.full(len(df), np.nan)
        for dt, g in df.groupby("end_date"):
            need = ["rn_p_up_mean"] + [f"rn_{b}" for b in subset]
            gg = g.dropna(subset=need)
            if len(gg) < args.min_names:
                continue
            X = np.column_stack([np.ones(len(gg))]
                                + [gg[f"rn_{b}"].to_numpy() for b in subset])
            y = gg["rn_p_up_mean"].to_numpy()
            b = ols(X, y)
            if b is None:
                continue
            vals[gg.index.to_numpy()] = y - X @ b
        df[name] = vals
        print(f"  {name:18s} {np.isfinite(vals).sum():,} rows")

    # ── 3. strictly causal betas (fit on months < t only) ───────────────────
    print("\n=== 3. Strictly causal neutralization (expanding-window betas) ===")
    hist_X, hist_y = [], []
    causal = np.full(len(df), np.nan)
    n_used = 0
    for dt in dates:
        g = df[df["end_date"] == dt]
        need = ["rn_p_up_mean"] + [f"rn_{b}" for b in BASELINES]
        gg = g.dropna(subset=need)
        if len(gg) >= args.min_names:
            Xt = np.column_stack([np.ones(len(gg))]
                                 + [gg[f"rn_{b}"].to_numpy() for b in BASELINES])
            yt = gg["rn_p_up_mean"].to_numpy()
            # predict with betas from STRICTLY EARLIER months only
            if len(hist_X) >= args.min_history_months:
                Xh = np.vstack(hist_X); yh = np.concatenate(hist_y)
                bh = ols(Xh, yh)
                if bh is not None:
                    causal[gg.index.to_numpy()] = yt - Xt @ bh
                    n_used += 1
            hist_X.append(Xt); hist_y.append(yt)
    df["resid_all_causal"] = causal
    print(f"  usable months {n_used} of {len(dates)} "
          f"(first {args.min_history_months} reserved for burn-in)")

    variants = list(NEUTRALIZE_SETS) + ["resid_all_causal"]

    # correlation of each variant with raw p_up -- how much was removed
    print("\n=== How much of raw p_up survives each neutralization? ===")
    corr_rows = []
    for v in variants:
        c = (df.dropna(subset=[v, "rn_p_up_mean"])
               .groupby("end_date")
               .apply(lambda g: g[v].corr(g["rn_p_up_mean"], method="spearman"),
                      include_groups=False))
        corr_rows.append({"variant": v, "mean_rank_corr_with_p_up": round(float(c.mean()), 4)})
    corr_tab = pd.DataFrame(corr_rows)
    print(corr_tab.to_string(index=False))

    keep = ["ticker", "end_date", "forward_return", "p_up_mean"] + BASELINES + variants
    df[keep].to_parquet(args.out_dir / "neutralized_signals.parquet", index=False)

    with pd.ExcelWriter(args.out_dir / "fama_macbeth.xlsx", engine="openpyxl") as xw:
        fm.to_excel(xw, sheet_name="fama_macbeth", index=False)
        corr_tab.to_excel(xw, sheet_name="corr_with_p_up", index=False)
        multi.to_excel(xw, sheet_name="fm_multi_monthly", index=False)

    (args.out_dir / "neutralize_summary.json").write_text(json.dumps({
        "fama_macbeth": fm.to_dict("records"),
        "corr_with_p_up": corr_tab.to_dict("records"),
        "variants": variants,
    }, indent=2, default=str))
    print(f"\nwrote {args.out_dir / 'neutralized_signals.parquet'}")
    print(f"wrote {args.out_dir / 'fama_macbeth.xlsx'}")


if __name__ == "__main__":
    main()
