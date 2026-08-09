"""
=============================================================================
SCRIPT NAME: mark_multi_signal.py
=============================================================================

Wire liquid-gated RLCC residual + Pattern residual into paper sleeves for
Tracker, and measure incremental IR of the multi-signal combo.

SLEEVES (all: $5 floor, top-500 DV universe, 50/50 dollar-neutral L/S,
monthly rebalance, daily mark — same arithmetic as mark_book.py):

  pattern-rlcc-liquid   score = rlcc_resid_wstr
  pattern-pup-resid-liq score = p_up_resid  (expected weak on liquid)
  pattern-combo-liquid  score = z(rlcc_resid_wstr) + z(p_up_resid)

INPUT:
  runs/residual_geometry/stress_rlcc_*/stress_signals.parquet  (preferred)
  or residual_geometry/phase0_*/residual_signals.parquet (+ rebuild resid)
  data/r1000_ohlcv_spliced.parquet

OUTPUT:
  data/tracker/<sleeve>/{nav_daily.csv,holdings_latest.csv,meta.json}
  data/tracker/multi_signal_sleeves.json
  runs/residual_geometry/multi_signal_<ts>/incremental_ir_report.md

USAGE:
  python3 -m scripts.residual_geometry.mark_multi_signal
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_signal import newey_west_t, series_stats  # noqa: E402

log = logging.getLogger("multi_signal")

OHLCV = ROOT / "data" / "r1000_ohlcv_spliced.parquet"
OUT_ROOT = ROOT / "data" / "tracker"
MIN_PRICE = 5.0
TOP_N_DV = 500

SLEEVES = {
    "pattern-rlcc-liquid": {
        "name": "RLCC residual (liquid top500)",
        "score": "rlcc_resid_wstr",
        "notes": "Dixon last-bar geometry residualized on WSTR+STR; top 500 DV",
    },
    "pattern-pup-resid-liq": {
        "name": "Pattern p_up residual (liquid top500)",
        "score": "p_up_resid",
        "notes": "CNN residual vs baselines; expected weak on liquid slice",
    },
    "pattern-combo-liquid": {
        "name": "RLCC residual + p_up residual (liquid)",
        "score": "combo_score",
        "notes": "Equal-weight CS z of rlcc_resid_wstr and p_up_resid",
    },
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


def load_scores(path: Path | None) -> pd.DataFrame:
    if path and path.exists():
        df = pd.read_parquet(path)
        df["end_date"] = pd.to_datetime(df["end_date"])
        return df

    # prefer stress_signals
    stress = sorted(
        (ROOT / "runs" / "residual_geometry").glob("stress_rlcc_*/stress_signals.parquet")
    )
    if stress:
        log.info("Loading %s", stress[-1])
        df = pd.read_parquet(stress[-1])
        df["end_date"] = pd.to_datetime(df["end_date"])
        return df

    phase0 = (
        ROOT / "runs" / "residual_geometry" / "phase0_20260809" / "residual_signals.parquet"
    )
    if not phase0.exists():
        raise FileNotFoundError("No residual/stress signal files found")
    log.info("Loading %s and rebuilding WSTR residuals", phase0)
    df = pd.read_parquet(phase0)
    df["end_date"] = pd.to_datetime(df["end_date"])
    if "rev_5d_or" not in df.columns:
        df["rev_5d_or"] = -df["rev_5d"]
    if "rev_1m_or" not in df.columns:
        df["rev_1m_or"] = -df["rev_1m"]
    parts = []
    for _, g in df.groupby("end_date"):
        g = g.copy()
        g["rlcc_resid_wstr"] = cs_residual(
            g["rlcc_or"], [g["rev_5d_or"], g["rev_1m_or"]]
        )
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def liquid_topn(df: pd.DataFrame, n: int = TOP_N_DV) -> pd.DataFrame:
    parts = []
    for dt, g in df.groupby("end_date"):
        g = g.dropna(subset=["dv_60d", "Close"])
        g = g[g["Close"] >= MIN_PRICE]
        if len(g) < 50:
            continue
        parts.append(g.nlargest(min(n, len(g)), "dv_60d"))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def add_combo(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for dt, g in df.groupby("end_date"):
        g = g.copy()
        a = g["rlcc_resid_wstr"]
        b = g["p_up_resid"]
        # if one missing, combo falls back to the available leg
        za = zscore(a.fillna(a.median()) if a.notna().sum() > 10 else a * 0)
        zb = zscore(b.fillna(b.median()) if b.notna().sum() > 10 else b * 0)
        both = a.notna() & b.notna()
        combo = np.full(len(g), np.nan)
        # when both present: average z; when only one: that z
        combo[both.to_numpy()] = (
            0.5 * za[both].to_numpy() + 0.5 * zb[both].to_numpy()
        )
        only_a = a.notna() & b.isna()
        only_b = b.notna() & a.isna()
        combo[only_a.to_numpy()] = za[only_a].to_numpy()
        combo[only_b.to_numpy()] = zb[only_b].to_numpy()
        g["combo_score"] = combo
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def build_sleeve_nav(
    preds: pd.DataFrame,
    score_col: str,
    px: pd.DataFrame,
    trading_days: np.ndarray,
    bt_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """50/50 L/S by score_col, daily mark — mirrors mark_book.build_sleeve."""
    sel = preds.dropna(subset=[score_col]).copy()
    rebals = sorted(sel["end_date"].unique())
    if len(trading_days):
        cur = pd.Timestamp(trading_days.max()).to_period("M")
        rebals = [d for d in rebals if pd.Timestamp(d).to_period("M") != cur]
    if not rebals:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    nav = 100.0
    bench = 100.0
    last_book = None

    for i, rb in enumerate(rebals):
        g = sel[sel["end_date"] == rb]
        if len(g) < 4:
            continue
        r = g[score_col].rank(method="first") / len(g)
        longs = g.loc[r > 0.5, "ticker"].tolist()
        shorts = g.loc[r <= 0.5, "ticker"].tolist()
        allnames = longs + shorts
        if len(longs) < 2 or len(shorts) < 2:
            continue

        end = rebals[i + 1] if i + 1 < len(rebals) else trading_days.max()
        days = trading_days[(trading_days > rb) & (trading_days <= end)]
        if len(days) == 0:
            continue

        try:
            p0 = px.loc[rb, allnames]
        except KeyError:
            continue
        p0 = p0[p0.notna() & (p0 > 0)]
        L = [t for t in longs if t in p0.index]
        S = [t for t in shorts if t in p0.index]
        if len(L) < 2 or len(S) < 2:
            continue

        nav_start, bench_start = nav, bench
        for d in days:
            if d not in px.index:
                continue
            pt = px.loc[d, p0.index]
            rel = (pt / p0) - 1.0
            rl = rel[L].mean(skipna=True)
            rs = rel[S].mean(skipna=True)
            rew = rel.mean(skipna=True)
            if not np.isfinite(rl) or not np.isfinite(rs):
                continue
            ls = rl - rs
            nav = nav_start * (1.0 + ls)
            bench = bench_start * (1.0 + (rew if np.isfinite(rew) else 0.0))
            rows.append(
                {
                    "date": d,
                    "nav": nav,
                    "ew_cell": bench,
                    "leg_long": rl,
                    "leg_short": rs,
                    "n_long": len(L),
                    "n_short": len(S),
                    "track": "backtest" if d <= bt_end else "paper",
                }
            )

        last_book = {
            "rebal": rb,
            "L": L,
            "S": S,
            "p0": p0,
            "scores": g.set_index("ticker")[score_col],
        }

    nav_df = pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date")
    if not nav_df.empty:
        nav_df["ret_1d_pct"] = nav_df["nav"].pct_change(fill_method=None) * 100
        nav_df["cum_ret_pct"] = (nav_df["nav"] / 100.0 - 1.0) * 100
        nav_df["bench_cum_ret_pct"] = (nav_df["ew_cell"] / 100.0 - 1.0) * 100
        nav_df["n_positions"] = nav_df["n_long"] + nav_df["n_short"]
        nav_df = nav_df[
            [
                "date",
                "nav",
                "ew_cell",
                "ret_1d_pct",
                "cum_ret_pct",
                "bench_cum_ret_pct",
                "n_positions",
                "n_long",
                "n_short",
                "leg_long",
                "leg_short",
                "track",
            ]
        ]

    hold = pd.DataFrame()
    if last_book is not None and not nav_df.empty:
        d_last = nav_df["date"].max()
        pt = px.loc[d_last, last_book["p0"].index]
        recs = []
        for side, names in (("long", last_book["L"]), ("short", last_book["S"])):
            w = (1.0 / len(names)) * (1 if side == "long" else -1)
            for t in names:
                recs.append(
                    {
                        "ticker": t,
                        "side": side,
                        "weight": w,
                        "score": float(last_book["scores"].get(t, np.nan)),
                        "entry_date": pd.Timestamp(last_book["rebal"]).date(),
                        "px_entry": float(last_book["p0"][t]),
                        "px_last": float(pt.get(t, np.nan)),
                    }
                )
        hold = pd.DataFrame(recs)
        hold["ret_since_entry"] = hold["px_last"] / hold["px_entry"] - 1.0
        hold = hold.sort_values(["side", "score"], ascending=[True, False])

    return nav_df, hold


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".csv.tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def monthly_ls_returns(preds: pd.DataFrame, score: str) -> pd.Series:
    """Per rebalance-date 20d simple LS 50/50 return (formation date series)."""
    rows = []
    for dt, g in preds.groupby("end_date"):
        g = g.dropna(subset=[score, "forward_return"])
        if len(g) < 50:
            continue
        r = g[score].rank(method="first") / len(g)
        hi = g.loc[r > 0.5, "forward_return"]
        lo = g.loc[r <= 0.5, "forward_return"]
        if len(hi) < 5 or len(lo) < 5:
            continue
        # simple portfolio arithmetic
        ls = np.expm1(hi).mean() - np.expm1(lo).mean()
        rows.append((dt, ls))
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series({d: v for d, v in rows}).sort_index()


def incremental_ir_report(
    preds: pd.DataFrame, out_dir: Path, navs: dict[str, pd.DataFrame]
) -> str:
    """Compare A, B, combo monthly L/S + daily NAV IR."""
    a = monthly_ls_returns(preds, "rlcc_resid_wstr")
    b = monthly_ls_returns(preds, "p_up_resid")
    c = monthly_ls_returns(preds, "combo_score")
    idx = a.index.intersection(b.index).intersection(c.index)
    a, b, c = a.loc[idx], b.loc[idx], c.loc[idx]

    def ir(s: pd.Series) -> dict:
        st = series_stats(s)
        return {
            "months": st["months"],
            "cagr_pct": st["cagr_pct"],
            "sharpe": st["sharpe"],
            "nw_t": st["nw_t"],
            "mean_pct": st["mean_pct"],
        }

    # regression: combo ~ alpha + beta * A  (incremental over RLCC alone)
    def ols_alpha(y: pd.Series, x: pd.Series) -> dict:
        m = y.notna() & x.notna()
        yv, xv = y[m].to_numpy(), x[m].to_numpy()
        if len(yv) < 24:
            return {"alpha_ann_pct": None, "beta": None, "alpha_t": None, "n": int(len(yv))}
        X = np.column_stack([np.ones(len(xv)), xv])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        # NW t on residual mean as alpha
        alpha_m = float(beta[0])
        # SE of alpha from OLS (rough)
        s2 = (resid ** 2).sum() / max(len(yv) - 2, 1)
        xtx_inv = np.linalg.inv(X.T @ X)
        se = float(np.sqrt(s2 * xtx_inv[0, 0]))
        t = alpha_m / se if se > 0 else float("nan")
        return {
            "alpha_ann_pct": round(alpha_m * 12 * 100, 3),
            "beta": round(float(beta[1]), 3),
            "alpha_t": round(t, 3),
            "n": int(len(yv)),
        }

    rho_ab = float(a.corr(b)) if len(a) > 5 else float("nan")
    alpha_c_a = ols_alpha(c, a)
    alpha_c_b = ols_alpha(c, b)
    alpha_a_b = ols_alpha(a, b)  # does RLCC add over p_up residual?

    # daily NAV IR if available
    daily_rows = []
    for sid, nav in navs.items():
        if nav is None or nav.empty:
            continue
        r = nav["nav"].pct_change(fill_method=None).dropna()
        yrs = (nav["date"].max() - nav["date"].min()).days / 365.25
        cagr = (
            ((nav["nav"].iloc[-1] / 100.0) ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan
        )
        sh = (
            (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
        )
        daily_rows.append(
            {
                "sleeve": sid,
                "days": len(nav),
                "cagr_pct": round(float(cagr), 3) if np.isfinite(cagr) else None,
                "sharpe": round(float(sh), 3) if np.isfinite(sh) else None,
                "final_nav": round(float(nav["nav"].iloc[-1]), 3),
            }
        )

    lines = [
        "# Multi-signal liquid sleeves — incremental IR",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Universe: top {TOP_N_DV} by 60d dollar volume, ${MIN_PRICE:.0f} floor, 50/50 L/S",
        "",
        "## Monthly L/S (simple return, 50/50)",
        "",
        "| Book | Months | Mean %/mo | CAGR % | Sharpe | NW t |",
        "|------|-------:|----------:|-------:|-------:|-----:|",
    ]
    for name, s in [
        ("rlcc_resid_wstr (A)", a),
        ("p_up_resid (B)", b),
        ("combo z(A)+z(B)", c),
    ]:
        st = ir(s)
        lines.append(
            f"| {name} | {st['months']} | {st['mean_pct']} | {st['cagr_pct']} | "
            f"{st['sharpe']} | {st['nw_t']} |"
        )

    lines += [
        "",
        f"**Corr(A, B) monthly LS** = **{rho_ab:.3f}**",
        "",
        "## Incremental alpha (monthly OLS)",
        "",
        f"| Test | α ann % | β | α t | n |",
        f"|------|--------:|--:|----:|--:|",
        f"| combo ~ A (RLCC alone) | {alpha_c_a['alpha_ann_pct']} | {alpha_c_a['beta']} | {alpha_c_a['alpha_t']} | {alpha_c_a['n']} |",
        f"| combo ~ B (p_up resid alone) | {alpha_c_b['alpha_ann_pct']} | {alpha_c_b['beta']} | {alpha_c_b['alpha_t']} | {alpha_c_b['n']} |",
        f"| A ~ B (does RLCC add over p_up resid?) | {alpha_a_b['alpha_ann_pct']} | {alpha_a_b['beta']} | {alpha_a_b['alpha_t']} | {alpha_a_b['n']} |",
        "",
        "Interpretation:",
        "- If combo ~ A has α≈0, p_up residual adds nothing on liquid to RLCC.",
        "- If A ~ B has large α, RLCC dominates residual CNN on liquid.",
        "",
        "## Daily-mark NAV (Tracker series)",
        "",
        "| Sleeve | Days | CAGR % | Sharpe | Final NAV |",
        "|--------|-----:|-------:|-------:|----------:|",
    ]
    for r in daily_rows:
        lines.append(
            f"| {r['sleeve']} | {r['days']} | {r['cagr_pct']} | {r['sharpe']} | {r['final_nav']} |"
        )

    # decision
    a_t = ir(a).get("nw_t") or 0
    c_t = ir(c).get("nw_t") or 0
    alpha_t = alpha_c_a.get("alpha_t") or 0
    if a_t >= 2 and (alpha_t is None or abs(float(alpha_t)) < 1.5):
        decision = (
            "**Ship RLCC residual liquid sleeve as primary.** "
            "Combo does not add significant alpha over A alone on this panel."
        )
    elif c_t > a_t and float(alpha_t or 0) >= 1.5:
        decision = (
            "**Ship combo sleeve.** p_up residual adds measurable alpha to RLCC on liquid."
        )
    else:
        decision = (
            "**Ship RLCC residual liquid; keep p_up residual as research feature only.**"
        )

    lines += ["", "## Decision", "", decision, ""]
    text = "\n".join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "incremental_ir_report.md").write_text(text)
    (out_dir / "incremental_ir.json").write_text(
        json.dumps(
            {
                "rho_ab": rho_ab,
                "A": ir(a),
                "B": ir(b),
                "combo": ir(c),
                "combo_vs_A": alpha_c_a,
                "combo_vs_B": alpha_c_b,
                "A_vs_B": alpha_a_b,
                "daily": daily_rows,
                "decision": decision,
            },
            indent=2,
        )
    )
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=Path, default=None)
    ap.add_argument("--ohlcv", type=Path, default=OHLCV)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--top-n", type=int, default=TOP_N_DV)
    ap.add_argument("--report-dir", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scores = load_scores(args.signals)
    # need dv_60d
    if "dv_60d" not in scores.columns or "Close" not in scores.columns:
        o = pd.read_parquet(
            args.ohlcv, columns=["Ticker", "Date", "Close", "Volume"]
        )
        o["Date"] = pd.to_datetime(o["Date"])
        o = o.sort_values(["Ticker", "Date"])
        o["dv"] = o["Close"] * o["Volume"]
        o["dv_60d"] = o.groupby("Ticker")["dv"].transform(
            lambda s: s.rolling(60, min_periods=15).mean()
        )
        dv = o.rename(columns={"Ticker": "ticker", "Date": "end_date"})[
            ["ticker", "end_date", "Close", "dv_60d"]
        ]
        scores = scores.drop(columns=[c for c in ["Close", "dv_60d"] if c in scores.columns])
        scores = scores.merge(dv, on=["ticker", "end_date"], how="left")

    scores = scores.dropna(subset=["forward_return"])
    liq = liquid_topn(scores, n=args.top_n)
    liq = add_combo(liq)
    log.info("Liquid panel: %s rows, %s months", f"{len(liq):,}", liq.end_date.nunique())

    # price pivot for daily mark
    o = pd.read_parquet(args.ohlcv)
    o["Date"] = pd.to_datetime(o["Date"])
    need = set(liq["ticker"])
    op = o[o["Ticker"].isin(need)]
    px = (
        op.pivot_table(index="Date", columns="Ticker", values="AdjClose", aggfunc="last")
        .sort_index()
    )
    trading_days = px.index.to_numpy()
    bt_end = liq["end_date"].max()

    navs: dict[str, pd.DataFrame] = {}
    summary = {}
    for sid, spec in SLEEVES.items():
        log.info("Building sleeve %s (score=%s)", sid, spec["score"])
        nav_df, hold = build_sleeve_nav(
            liq, spec["score"], px, trading_days, bt_end
        )
        if nav_df.empty:
            log.warning("  empty — skip")
            continue
        dest = args.out_root / sid
        atomic_write_csv(nav_df, dest / "nav_daily.csv")
        if not hold.empty:
            # Tracker holdings may expect p_up_mean column name — provide alias
            h = hold.rename(columns={"score": "p_up_mean"})
            atomic_write_csv(h, dest / "holdings_latest.csv")

        r = nav_df["nav"].pct_change(fill_method=None).dropna()
        yrs = (nav_df["date"].max() - nav_df["date"].min()).days / 365.25
        cagr = (
            ((nav_df["nav"].iloc[-1] / 100.0) ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan
        )
        sharpe = (
            (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
        )
        paper = nav_df[nav_df["track"] == "paper"]
        meta = {
            "sleeve": sid,
            "name": spec["name"],
            "score": spec["score"],
            "universe": f"top{args.top_n}_dv",
            "min_price": MIN_PRICE,
            "as_of": str(nav_df["date"].max().date()),
            "series_start": str(nav_df["date"].min().date()),
            "backtest_through": str(bt_end.date()),
            "paper_start": str(bt_end.date() + pd.Timedelta(days=1)),
            "paper_days": int(len(paper)),
            "n_long": int(nav_df["n_long"].iloc[-1]),
            "n_short": int(nav_df["n_short"].iloc[-1]),
            "n_positions": int(
                nav_df["n_long"].iloc[-1] + nav_df["n_short"].iloc[-1]
            ),
            "nav": float(nav_df["nav"].iloc[-1]),
            "benchmark": "cash (dollar-neutral)",
            "cagr_pct": None if not np.isfinite(cagr) else round(float(cagr), 3),
            "sharpe": None if not np.isfinite(sharpe) else round(float(sharpe), 3),
            "notes": spec["notes"],
            "flags": [
                "gross_of_costs",
                "nav_is_index_not_dollars",
                "simple_return_portfolio_arithmetic",
                "long_short_requires_single_stock_shorting",
                f"price_floor_{int(MIN_PRICE)}usd",
                f"liquid_top{args.top_n}_dv",
                "backtest_leg_only_until_live_geometry_scores",
            ],
            "generated": datetime.now().isoformat(timespec="seconds"),
        }
        (dest / "meta.json").write_text(json.dumps(meta, indent=2))
        navs[sid] = nav_df
        summary[sid] = meta
        log.info(
            "  %s -> %s  NAV=%.2f  CAGR=%+.2f%%  Sharpe=%+.2f  book=%d/%d",
            nav_df["date"].min().date(),
            nav_df["date"].max().date(),
            nav_df["nav"].iloc[-1],
            cagr,
            sharpe,
            meta["n_long"],
            meta["n_short"],
        )

    idx = args.out_root / "multi_signal_sleeves.json"
    # merge into sleeves.json if exists
    sleeves_path = args.out_root / "sleeves.json"
    if sleeves_path.exists():
        try:
            existing = json.loads(sleeves_path.read_text())
        except Exception:
            existing = {}
        existing.update(summary)
        sleeves_path.write_text(json.dumps(existing, indent=2))
    idx.write_text(json.dumps(summary, indent=2))

    report_dir = args.report_dir or (
        ROOT
        / "runs"
        / "residual_geometry"
        / f"multi_signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    text = incremental_ir_report(liq, report_dir, navs)
    print(text)
    print(f"\nTracker artifacts under {args.out_root}")
    print(f"Report: {report_dir / 'incremental_ir_report.md'}")


if __name__ == "__main__":
    main()
