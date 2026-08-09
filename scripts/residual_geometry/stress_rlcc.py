"""
RLCC / geometry stress test (next step after residual Phase 0).

Tests:
  1. Residualize rlcc_or / geom_score on WSTR+STR (is it just weekly reversal?)
  2. Liquid slices: full floored, top-2/3 DV, top 500 by DV
  3. Sub-periods: Full / 2015+ / 2020+
  4. Double-sort: RLCC tercile spread inside rev_5d quintiles

INPUT:
  runs/residual_geometry/phase0_20260809/residual_signals.parquet
  data/r1000_ohlcv_spliced.parquet

OUTPUT:
  runs/residual_geometry/stress_rlcc_<ts>/  report.md, tables
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_signal import (  # type: ignore
    build_books,
    newey_west_t,
    series_stats,
)

log = logging.getLogger("stress_rlcc")

DEFAULT_SIG = (
    ROOT / "runs" / "residual_geometry" / "phase0_20260809" / "residual_signals.parquet"
)
DEFAULT_OHLCV = ROOT / "data" / "r1000_ohlcv_spliced.parquet"
MIN_PRICE = 5.0
MIN_NAMES = 50
PERIODS = {
    "full": None,
    "since_2015": "2015-01-01",
    "since_2020": "2020-01-01",
}


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    if not sd or not np.isfinite(sd) or sd <= 0:
        return s * 0.0
    return (s - s.mean()) / sd


def cs_residual(y: pd.Series, X_list: list[pd.Series]) -> np.ndarray:
    n = len(y)
    yz = zscore(y).to_numpy(float)
    cols = [zscore(x).to_numpy(float) for x in X_list]
    X = np.column_stack([np.ones(n)] + cols)
    resid = np.full(n, np.nan)
    ok = np.isfinite(X).all(1) & np.isfinite(yz)
    if ok.sum() <= X.shape[1] + 3:
        return resid
    beta, *_ = np.linalg.lstsq(X[ok], yz[ok], rcond=None)
    resid[ok] = yz[ok] - X[ok] @ beta
    return resid


def add_dv(ohlcv_path: Path) -> pd.DataFrame:
    o = pd.read_parquet(
        ohlcv_path, columns=["Ticker", "Date", "Close", "Volume"]
    )
    o["Date"] = pd.to_datetime(o["Date"])
    o = o.sort_values(["Ticker", "Date"])
    o["dv"] = o["Close"] * o["Volume"]
    o["dv_60d"] = o.groupby("Ticker")["dv"].transform(
        lambda s: s.rolling(60, min_periods=15).mean()
    )
    return o.rename(columns={"Ticker": "ticker", "Date": "end_date"})[
        ["ticker", "end_date", "Close", "dv_60d"]
    ]


def monthly_ic(df: pd.DataFrame, score: str) -> pd.Series:
    ics = []
    idx = []
    for dt, g in df.groupby("end_date"):
        s = g[score]
        y = g["forward_return"]
        m = s.notna() & y.notna()
        if m.sum() < 30:
            continue
        ic = s[m].corr(y[m], method="spearman")
        if np.isfinite(ic):
            ics.append(ic)
            idx.append(dt)
    return pd.Series(ics, index=pd.to_datetime(idx), name=score)


def portfolio_row(df: pd.DataFrame, score: str, period: str, since: str | None) -> dict:
    d = df.dropna(subset=[score, "r_simple", "forward_return"]).copy()
    if since:
        d = d[d["end_date"] >= pd.Timestamp(since)]
    book, meta = build_books(d, score, n_deciles=10, min_names=MIN_NAMES)
    if book.empty:
        return {
            "score": score,
            "period": period,
            "months": 0,
            "mean_names": float("nan"),
            "ls_decile_cagr": float("nan"),
            "ls_decile_t": float("nan"),
            "ls_5050_cagr": float("nan"),
            "ls_5050_t": float("nan"),
            "skipped": meta.get("skipped_months", 0),
        }
    # mean names
    mean_n = float(book["n"].mean())
    out = {
        "score": score,
        "period": period,
        "months": int(len(book)),
        "mean_names": round(mean_n, 1),
        "skipped": meta.get("skipped_months", 0),
    }
    for col, key in [("ls_decile", "ls_decile"), ("ls_5050", "ls_5050")]:
        st = series_stats(book[col])
        out[f"{key}_cagr"] = st["cagr_pct"]
        out[f"{key}_t"] = st["nw_t"]
        out[f"{key}_sharpe"] = st["sharpe"]
    return out


def double_sort_rlcc_in_wstr(df: pd.DataFrame) -> pd.DataFrame:
    """Inside each rev_5d quintile (oriented: low past ret = buy), RLCC tercile spread."""
    rows = []
    for dt, g in df.groupby("end_date"):
        g = g.dropna(subset=["rlcc_or", "rev_5d_or", "r_simple"])
        if len(g) < 100:
            continue
        g = g.assign(
            wstr_q=pd.qcut(
                g["rev_5d_or"].rank(method="first"), 5, labels=range(1, 6)
            )
        )
        for q, sub in g.groupby("wstr_q", observed=True):
            if len(sub) < 15:
                continue
            r = sub["rlcc_or"].rank(method="first", pct=True)
            hi = sub.loc[r > 2 / 3, "r_simple"]
            lo = sub.loc[r <= 1 / 3, "r_simple"]
            if len(hi) >= 3 and len(lo) >= 3:
                rows.append(
                    {
                        "end_date": dt,
                        "wstr_q": int(q),
                        "spread": hi.mean() - lo.mean(),
                    }
                )
    if not rows:
        return pd.DataFrame()
    ds = pd.DataFrame(rows)
    tab = (
        ds.groupby("wstr_q")
        .agg(months=("spread", "size"), mean_spread=("spread", "mean"))
        .reset_index()
    )
    tab["mean_spread_pct"] = (tab["mean_spread"] * 100).round(4)
    tab["nw_t"] = [
        round(newey_west_t(ds.loc[ds.wstr_q == q, "spread"].to_numpy()), 2)
        for q in tab["wstr_q"]
    ]
    # q1 = strongest WSTR long (biggest losers), q5 = winners
    tab["note"] = tab["wstr_q"].map(
        {
            1: "WSTR longs (recent losers)",
            2: "WSTR q2",
            3: "WSTR mid",
            4: "WSTR q4",
            5: "WSTR shorts (recent winners)",
        }
    )
    return tab


def liquid_masks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return named universe slices (already price-floored)."""
    out = {"all_p5": df.copy()}
    # top 2/3 dollar volume within month
    parts = []
    for dt, g in df.groupby("end_date"):
        g = g.dropna(subset=["dv_60d"])
        if len(g) < MIN_NAMES:
            continue
        thr = g["dv_60d"].quantile(1 / 3)
        parts.append(g[g["dv_60d"] >= thr])
    out["top2thirds_dv"] = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]

    parts = []
    for dt, g in df.groupby("end_date"):
        g = g.dropna(subset=["dv_60d"])
        if len(g) < 50:
            continue
        top = g.nlargest(min(500, len(g)), "dv_60d")
        parts.append(top)
    out["top500_dv"] = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]

    # high DV tertile only
    parts = []
    for dt, g in df.groupby("end_date"):
        g = g.dropna(subset=["dv_60d"])
        if len(g) < MIN_NAMES:
            continue
        thr = g["dv_60d"].quantile(2 / 3)
        parts.append(g[g["dv_60d"] >= thr])
    out["high_dv_tertile"] = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=Path, default=DEFAULT_SIG)
    ap.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        ROOT / "runs" / "residual_geometry" / f"stress_rlcc_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("out_dir=%s", out_dir)

    sig = pd.read_parquet(args.signals)
    sig["end_date"] = pd.to_datetime(sig["end_date"])
    dv = add_dv(args.ohlcv)
    df = sig.merge(dv, on=["ticker", "end_date"], how="left")
    df = df[df["Close"] >= MIN_PRICE].copy()
    df["r_simple"] = np.expm1(df["forward_return"])
    # ensure oriented WSTR exists
    if "rev_5d_or" not in df.columns:
        df["rev_5d_or"] = -df["rev_5d"]
    if "rev_1m_or" not in df.columns:
        df["rev_1m_or"] = -df["rev_1m"]

    # ── 1. residualize geometry on WSTR/STR ────────────────────────────────
    log.info("Residualizing RLCC / geom on WSTR+STR …")
    parts = []
    for dt, g in df.groupby("end_date"):
        if len(g) < MIN_NAMES:
            continue
        g = g.copy()
        g["rlcc_resid_wstr"] = cs_residual(
            g["rlcc_or"], [g["rev_5d_or"], g["rev_1m_or"]]
        )
        g["geom_resid_wstr"] = cs_residual(
            g["geom_score"], [g["rev_5d_or"], g["rev_1m_or"]]
        )
        # also residualize on WSTR only
        g["rlcc_resid_wstr_only"] = cs_residual(g["rlcc_or"], [g["rev_5d_or"]])
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)

    scores = [
        "rlcc_or",
        "geom_score",
        "rev_5d_or",
        "p_up_resid",
        "p_up_resid_full",
        "rlcc_resid_wstr",
        "rlcc_resid_wstr_only",
        "geom_resid_wstr",
    ]

    # ── 2–3. IC + portfolio by universe × period ───────────────────────────
    universes = liquid_masks(df)
    ic_rows = []
    port_rows = []

    for uname, udf in universes.items():
        log.info("Universe %s  n=%s  months=%s", uname, f"{len(udf):,}", udf.end_date.nunique())
        for score in scores:
            if score not in udf.columns:
                continue
            for pname, since in PERIODS.items():
                d = udf if since is None else udf[udf["end_date"] >= pd.Timestamp(since)]
                ics = monthly_ic(d, score)
                ic_rows.append(
                    {
                        "universe": uname,
                        "score": score,
                        "period": pname,
                        "n_months": int(len(ics)),
                        "mean_ic": float(ics.mean()) if len(ics) else float("nan"),
                        "ic_nw_t": newey_west_t(ics.to_numpy()) if len(ics) else float("nan"),
                        "pct_pos": float((ics > 0).mean()) if len(ics) else float("nan"),
                    }
                )
                port_rows.append(
                    {
                        "universe": uname,
                        **portfolio_row(d, score, pname, since),
                    }
                )

    ic_tab = pd.DataFrame(ic_rows)
    port_tab = pd.DataFrame(port_rows)
    ic_tab.to_csv(out_dir / "ic_by_universe_period.csv", index=False)
    port_tab.to_csv(out_dir / "portfolio_by_universe_period.csv", index=False)

    # ── 4. double sort ─────────────────────────────────────────────────────
    ds = double_sort_rlcc_in_wstr(df)
    if len(ds):
        ds.to_csv(out_dir / "double_sort_rlcc_in_wstr.csv", index=False)

    # ── decision helpers ───────────────────────────────────────────────────
    def get_port(universe: str, score: str, period: str = "full") -> pd.Series:
        m = (
            (port_tab.universe == universe)
            & (port_tab.score == score)
            & (port_tab.period == period)
        )
        return port_tab.loc[m].iloc[0] if m.any() else pd.Series(dtype=float)

    def get_ic(universe: str, score: str, period: str = "full") -> pd.Series:
        m = (
            (ic_tab.universe == universe)
            & (ic_tab.score == score)
            & (ic_tab.period == period)
        )
        return ic_tab.loc[m].iloc[0] if m.any() else pd.Series(dtype=float)

    # key comparisons
    keys = []
    for u in ["all_p5", "top2thirds_dv", "top500_dv", "high_dv_tertile"]:
        for s in [
            "rlcc_or",
            "rlcc_resid_wstr",
            "rlcc_resid_wstr_only",
            "geom_score",
            "geom_resid_wstr",
            "rev_5d_or",
            "p_up_resid",
        ]:
            for p in ["full", "since_2015", "since_2020"]:
                pr = get_port(u, s, p)
                ir = get_ic(u, s, p)
                if pr.empty:
                    continue
                keys.append(
                    {
                        "universe": u,
                        "score": s,
                        "period": p,
                        "ic_t": ir.get("ic_nw_t", float("nan")),
                        "mean_ic": ir.get("mean_ic", float("nan")),
                        "ls_decile_cagr": pr.get("ls_decile_cagr", float("nan")),
                        "ls_decile_t": pr.get("ls_decile_t", float("nan")),
                        "ls_5050_t": pr.get("ls_5050_t", float("nan")),
                        "months": pr.get("months", 0),
                        "mean_names": pr.get("mean_names", float("nan")),
                    }
                )
    key_tab = pd.DataFrame(keys)
    key_tab.to_csv(out_dir / "key_results.csv", index=False)

    # verdicts
    r_full = get_port("all_p5", "rlcc_resid_wstr", "full")
    r_liq = get_port("top500_dv", "rlcc_resid_wstr", "full")
    r_hi = get_port("high_dv_tertile", "rlcc_resid_wstr", "full")
    r_raw_liq = get_port("top500_dv", "rlcc_or", "full")
    r_raw = get_port("all_p5", "rlcc_or", "full")
    wstr = get_port("all_p5", "rev_5d_or", "full")

    t_resid = float(r_full.get("ls_decile_t", float("nan"))) if len(r_full) else float("nan")
    t_resid_liq = float(r_liq.get("ls_decile_t", float("nan"))) if len(r_liq) else float("nan")
    t_raw = float(r_raw.get("ls_decile_t", float("nan"))) if len(r_raw) else float("nan")

    if t_resid <= 1.5 or not np.isfinite(t_resid):
        verdict = (
            "KILL as new factor: RLCC residual on WSTR+STR has weak L/S "
            f"(decile t={t_resid:.2f}). Treat RLCC as WSTR-family refinement only."
        )
    elif t_resid_liq >= 2.0:
        verdict = (
            f"KEEP as free technical feature: RLCC residual lives (t={t_resid:.2f}) "
            f"AND survives top500 DV (t={t_resid_liq:.2f})."
        )
    elif float(r_raw_liq.get("ls_decile_t", 0) or 0) >= 2:
        verdict = (
            f"RLCC residual on WSTR is soft (t={t_resid:.2f}) but raw RLCC still "
            f"works on liquid top500 (t={float(r_raw_liq.get('ls_decile_t', float('nan'))):.2f}). "
            "Likely better packaging of short-term reversal — use carefully."
        )
    else:
        verdict = (
            f"FEATURE-ONLY / thin: residual t={t_resid:.2f}, liquid residual t={t_resid_liq:.2f}. "
            "Do not build a book; optional multi-signal input only."
        )

    # report
    def fmt_block(title: str, scores_show: list[str], universe: str) -> list[str]:
        lines = [f"### {title}", "", "| Score | Period | IC t | LS dec t | LS 50/50 t | CAGR dec | Months | Names |",
                 "|-------|--------|-----:|---------:|-----------:|---------:|-------:|------:|"]
        sub = key_tab[(key_tab.universe == universe) & (key_tab.score.isin(scores_show))]
        for _, r in sub.sort_values(["score", "period"]).iterrows():
            lines.append(
                f"| {r.score} | {r.period} | {r.ic_t:.2f} | {r.ls_decile_t:.2f} | "
                f"{r.ls_5050_t:.2f} | {r.ls_decile_cagr:.2f}% | {int(r.months)} | {r.mean_names:.0f} |"
            )
        lines.append("")
        return lines

    lines = [
        "# RLCC / geometry stress test",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Source signals: `{args.signals}`",
        f"$5 price floor applied. Simple-return L/S via evaluate_signal.build_books.",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"- Raw RLCC all_p5 LS decile t = **{t_raw:.2f}**",
        f"- RLCC residual on WSTR+STR all_p5 LS decile t = **{t_resid:.2f}**",
        f"- RLCC residual top500 DV LS decile t = **{t_resid_liq:.2f}**",
        f"- WSTR (rev_5d_or) all_p5 LS decile t = **{float(wstr.get('ls_decile_t', float('nan'))):.2f}**",
        "",
    ]
    lines += fmt_block(
        "All names ($5 floor)",
        ["rlcc_or", "rlcc_resid_wstr", "rlcc_resid_wstr_only", "geom_score", "geom_resid_wstr", "rev_5d_or", "p_up_resid"],
        "all_p5",
    )
    lines += fmt_block(
        "Top 500 by 60d dollar volume",
        ["rlcc_or", "rlcc_resid_wstr", "rev_5d_or", "p_up_resid", "geom_score"],
        "top500_dv",
    )
    lines += fmt_block(
        "High DV tertile only",
        ["rlcc_or", "rlcc_resid_wstr", "rev_5d_or", "p_up_resid"],
        "high_dv_tertile",
    )
    lines += fmt_block(
        "Top 2/3 DV",
        ["rlcc_or", "rlcc_resid_wstr", "rev_5d_or"],
        "top2thirds_dv",
    )

    lines += ["## Double-sort: RLCC spread inside WSTR quintiles", ""]
    if len(ds):
        lines.append("| WSTR q | Months | Mean spread % | NW t | Note |")
        lines.append("|-------:|-------:|--------------:|-----:|------|")
        for _, r in ds.iterrows():
            lines.append(
                f"| {int(r.wstr_q)} | {int(r.months)} | {r.mean_spread_pct:.3f} | "
                f"{r.nw_t:.2f} | {r.note} |"
            )
        lines.append("")
        lines.append(
            "If RLCC spreads stay significant across all WSTR quintiles, "
            "RLCC is not *only* a re-sort of weekly reversal."
        )
    else:
        lines.append("(no double-sort rows)")

    lines += [
        "",
        "## Files",
        "",
        "- `key_results.csv` — compact grid",
        "- `ic_by_universe_period.csv`",
        "- `portfolio_by_universe_period.csv`",
        "- `double_sort_rlcc_in_wstr.csv`",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))

    # also save enriched signals for reuse
    df[
        [
            c
            for c in [
                "ticker",
                "end_date",
                "forward_return",
                "rlcc_or",
                "geom_score",
                "rev_5d_or",
                "rev_1m_or",
                "p_up_resid",
                "p_up_resid_full",
                "rlcc_resid_wstr",
                "rlcc_resid_wstr_only",
                "geom_resid_wstr",
                "dv_60d",
                "Close",
            ]
            if c in df.columns
        ]
    ].to_parquet(out_dir / "stress_signals.parquet", index=False)

    meta = {
        "verdict": verdict,
        "t_raw_rlcc": t_raw,
        "t_rlcc_resid_wstr": t_resid,
        "t_rlcc_resid_top500": t_resid_liq,
        "t_wstr": float(wstr.get("ls_decile_t", float("nan"))),
    }
    (out_dir / "verdict.json").write_text(json.dumps(meta, indent=2))
    log.info("Verdict: %s", verdict)
    log.info("Done → %s", out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
