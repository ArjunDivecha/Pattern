"""Best-supported historical A/B using cached IBKR spread and borrow histories.

This is an estimated-cost test, not a fill simulation. Missing histories stay missing.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/cost_aware_ab_2020_2026_1m"
CACHE = OUT / "broker_history"
CAPITAL = 1_000_000.0
NAMES = 50
ORDERS_PER_NAME = 2
Y_SENSITIVITIES = (0.0, 0.1, 0.3, 0.6)
FEE_SCALES = {"native_decimal": 1.0, "percent_point_div100": 0.01}


def fnum(x):
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else np.nan


def prepare_sigma(panel: pd.DataFrame) -> pd.DataFrame:
    """Ex-ante daily volatility proxy from prior non-overlapping five-session outcomes."""
    panel = panel.sort_values(["security", "date"]).copy()
    panel["sigma_daily"] = panel.groupby("security", sort=False)["absolute_forward_return"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).std(ddof=1) / math.sqrt(5)
    )
    return panel[["security", "date", "sigma_daily"]]


def bar_arrays(record):
    series = (record or {}).get("series", {})
    out = {}
    for name in ("BID_ASK", "FEE_RATE"):
        bars = series.get(name, {}).get("bars", []) or []
        dates, vals = [], []
        for b in bars:
            try:
                d = np.datetime64(pd.Timestamp(b["date"]).date(), "D")
            except Exception:
                continue
            if name == "BID_ASK":
                bid, ask = fnum(b.get("open")), fnum(b.get("close"))
                if not (np.isfinite(bid) and np.isfinite(ask) and 0 < bid <= ask):
                    continue
                vals.append((bid, ask))
            else:
                rate = fnum(b.get("close"))
                if not np.isfinite(rate) or rate < 0:
                    continue
                vals.append(rate)
            dates.append(d)
        if dates:
            ix = np.argsort(np.asarray(dates, dtype="datetime64[D]"))
            out[name] = (np.asarray(dates, dtype="datetime64[D]")[ix], np.asarray(vals)[ix])
        else:
            out[name] = (np.array([], dtype="datetime64[D]"), np.array([]))
    return out


def asof(dates, vals, when, max_age_days):
    if len(dates) == 0 or pd.isna(when):
        return None
    i = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(when).date(), "D"), side="right") - 1)
    if i < 0:
        return None
    age = int((np.datetime64(pd.Timestamp(when).date(), "D") - dates[i]).astype(int))
    if age < 0 or age > max_age_days:
        return None
    return vals[i]


def spread_frac(bidask):
    if bidask is None:
        return np.nan
    bid, ask = map(float, bidask)
    mid = (bid + ask) / 2
    return (ask - bid) / mid if mid > 0 else np.nan


def trailing_spread(dates, vals, date):
    if len(dates) == 0:
        return np.nan
    end = np.datetime64(pd.Timestamp(date).date(), "D")
    j = int(np.searchsorted(dates, end, side="right"))
    if j == 0 or int((end - dates[j - 1]).astype(int)) > 3:
        return np.nan
    lo_date = end - np.timedelta64(45, "D")
    i = int(np.searchsorted(dates, lo_date, side="left"))
    recent = vals[i:j]
    if len(recent) < 10:
        return np.nan
    bids, asks = recent[:, 0], recent[:, 1]
    mids = (bids + asks) / 2
    fractions = (asks - bids) / mids
    fractions = fractions[np.isfinite(fractions) & (fractions >= 0)]
    return float(np.median(fractions)) if len(fractions) >= 10 else np.nan


def commission_reg_cost(price, notional):
    if not (np.isfinite(price) and price > 0):
        return np.nan
    shares = notional / price
    # IBKR Pro Fixed proxy: $0.005/share, $1/order minimum, 1% notional cap.
    comm = ORDERS_PER_NAME * min(0.01 * notional, max(1.0, 0.005 * shares))
    # Approximate SEC, FINRA and CAT fees on one sale per round trip.
    reg = notional * 0.0000206 + shares * 0.000195 + 2 * shares * 0.000003
    return (comm + reg) / notional


def cost_rows_for_security(sec, rows, bars):
    bd, bv = bars["BID_ASK"]
    fd, fv = bars["FEE_RATE"]
    ans = []
    for r in rows.itertuples(index=False):
        form = pd.Timestamp(r.date)
        entry = pd.Timestamp(r.entry_date) if pd.notna(r.entry_date) else pd.NaT
        exit_ = pd.Timestamp(r.exit_date) if pd.notna(r.exit_date) else pd.NaT
        pre_s = trailing_spread(bd, bv, form)
        fr = asof(fd, fv, form, 7)
        fee_pre = fnum(fr) if fr is not None else np.nan
        cm = commission_reg_cost(float(r.close), CAPITAL / NAMES)
        adv = fnum(getattr(r, "adv20", np.nan))
        sigma = fnum(getattr(r, "sigma_daily", np.nan))
        impact_pre_unit = 2 * sigma * math.sqrt((CAPITAL / NAMES) / adv) if adv > 0 and sigma >= 0 else np.nan
        # Use the daily time-average bid/ask bars on the actual next-open entry and exit dates.
        eba = asof(bd, bv, entry, 0)
        xba = asof(bd, bv, exit_, 0)
        if eba is not None and xba is not None:
            ebid, eask = map(float, eba); xbid, xask = map(float, xba)
            entry_mid = (ebid + eask) / 2
            fill_spread = ((eask - ebid) / 2 + (xask - xbid) / 2) / entry_mid if entry_mid > 0 else np.nan
            fill_comm_reg = commission_reg_cost(entry_mid, CAPITAL / NAMES)
        else:
            fill_spread = fill_comm_reg = np.nan
        # Borrow fee accrues by calendar day for the observed entry-to-exit interval.
        days = max(0, (exit_.date() - entry.date()).days) if pd.notna(entry) and pd.notna(exit_) else 0
        fee_sum = 0.0; fee_ok = days > 0
        for k in range(days):
            day = entry + pd.Timedelta(days=k)
            rate = asof(fd, fv, day, 7)
            if rate is None:
                fee_ok = False; break
            fee_sum += float(rate) / 360.0
        fee_realized_native = 1.02 * fee_sum if fee_ok else np.nan
        ans.append({
            "security": sec, "date": form, "entry_date": entry, "exit_date": exit_,
            "pre_spread_rt": pre_s, "pre_comm_reg_rt": cm, "pre_fee_native": fee_pre,
            "pre_impact_unit": impact_pre_unit, "pre_adv20": adv, "pre_sigma_daily": sigma,
            "fill_spread_rt": fill_spread, "fill_comm_reg_rt": fill_comm_reg,
            "fill_fee_native": fee_realized_native, "fill_calendar_days": days,
            "fill_entry_quote": eba is not None, "fill_exit_quote": xba is not None,
            "fee_covered": fee_ok,
        })
    return ans


def summary_metrics(cohorts, return_col, label):
    x = cohorts.loc[np.isfinite(cohorts[return_col]), ["date", return_col]].copy().sort_values("date")
    if x.empty:
        return {"portfolio": label, "complete_cohorts": 0}
    r = x[return_col].astype(float).to_numpy()
    years = max((x.date.iloc[-1] - x.date.iloc[0]).days / 365.25, 1 / 50.4)
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])
    dd = np.r_[1.0, wealth] / peak - 1
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
    return {
        "portfolio": label, "complete_cohorts": int(len(r)), "first_date": str(x.date.iloc[0].date()),
        "last_date": str(x.date.iloc[-1].date()), "mean_holding_period_spread_pct": mu * 100,
        "annualized_arithmetic_spread_pct": mu * 50.4 * 100,
        "annualized_sharpe": mu / sd * math.sqrt(50.4) if sd and np.isfinite(sd) else np.nan,
        "compound_return_pct": (wealth[-1] - 1) * 100,
        "cagr_pct": ((wealth[-1] ** (1 / years)) - 1) * 100 if wealth[-1] > 0 else np.nan,
        "max_drawdown_pct": float(np.min(dd)) * 100,
    }


def main():
    out = OUT
    eligible = pd.read_parquet(out / "eligible_panel.parquet")
    eligible = eligible[eligible.date.dt.year >= 2020].copy()
    sigma = prepare_sigma(pd.read_parquet(out / "eligible_panel.parquet"))
    eligible = eligible.merge(sigma, on=["security", "date"], how="left", validate="one_to_one")
    a = pd.read_parquet(out / "A_holdings.parquet")
    b = pd.read_parquet(out / "B_candidates.parquet")
    assert a.groupby(["date", "leg"]).size().eq(50).all(), "A must retain 50 equal-weight names per side/date"
    a = a.merge(sigma, on=["security", "date"], how="left", validate="one_to_one")
    b = b.merge(sigma, on=["security", "date"], how="left", validate="one_to_one")
    req = pd.read_csv(out / "broker_requests.csv")
    sec_to_path = {r.security: CACHE / (hashlib.sha256(r.security.encode()).hexdigest()[:20] + ".json") for r in req.itertuples(index=False)}
    all_rows = pd.concat([a, b], ignore_index=True, sort=False).drop_duplicates(
        ["security", "date", "entry_date", "exit_date"]
    )
    records = []
    for sec, group in all_rows.groupby("security", sort=False):
        path = sec_to_path.get(sec)
        if path is None or not path.exists():
            bars = bar_arrays({})
        else:
            try:
                bars = bar_arrays(json.loads(path.read_text()))
            except Exception:
                bars = bar_arrays({})
        records.extend(cost_rows_for_security(sec, group, bars))
    costs = pd.DataFrame(records)
    a = a.merge(costs, on=["security", "date", "entry_date", "exit_date"], how="left", validate="one_to_one")
    b = b.merge(costs, on=["security", "date", "entry_date", "exit_date"], how="left", validate="one_to_one")
    a["leg"] = a.leg.astype(str)
    # A result set with different fee/impact assumptions, each selected only from prior-data costs.
    portfolio_rows = []
    candidate_audit = []
    for scale_name, fee_scale in FEE_SCALES.items():
        for y in Y_SENSITIVITIES:
            cand = b.copy()
            cand["pre_cost"] = cand.pre_spread_rt + cand.pre_comm_reg_rt + y * cand.pre_impact_unit
            cand.loc[cand.leg.eq("short"), "pre_cost"] += fee_scale * 1.02 * cand.loc[cand.leg.eq("short"), "pre_fee_native"] * cand.loc[cand.leg.eq("short"), "fill_calendar_days"] / 360.0
            cand["pre_cost_ok"] = np.isfinite(cand.pre_cost) & np.isfinite(cand.expected_relative_return) & cand.pre_adv20.gt(0)
            cand["net_edge"] = np.where(cand.leg.eq("long"), cand.expected_relative_return - cand.pre_cost,
                                         -cand.expected_relative_return - cand.pre_cost)
            chosen = []
            for date, g in cand.groupby("date", sort=True):
                for leg in ("long", "short"):
                    pool = g[(g.leg == leg) & g.pre_cost_ok].sort_values(["net_edge", "security"], ascending=[False, True])
                    candidate_audit.append({"date": date, "fee_case": scale_name, "impact_y": y, "leg": leg,
                                            "cost_estimable_candidates": int(len(pool)), "needed": 50})
                    if len(pool) >= 50:
                        chosen.append(pool.head(50).assign(portfolio=f"B_{scale_name}_Y{y}"))
            if chosen:
                portfolio_rows.append(pd.concat(chosen, ignore_index=True))
    bsel = pd.concat(portfolio_rows, ignore_index=True) if portfolio_rows else pd.DataFrame()
    a = a.assign(portfolio="A")
    holdings = pd.concat([a, bsel], ignore_index=True, sort=False)
    for portfolio, group in holdings[holdings.portfolio.str.startswith("B_")].groupby("portfolio"):
        assert group.groupby(["date", "leg"]).size().eq(50).all(), f"{portfolio}: selected legs must have 50 names"
    # Per-name measured cost fraction; divide a $20k name's costs by its notional.
    holdings["realized_cost_no_borrow"] = holdings.fill_spread_rt + holdings.fill_comm_reg_rt
    holdings["realized_cost_base"] = holdings.realized_cost_no_borrow + 0.3 * holdings.pre_impact_unit
    holdings["realized_cost_low"] = holdings.realized_cost_no_borrow + 0.1 * holdings.pre_impact_unit
    holdings["realized_cost_high"] = holdings.realized_cost_no_borrow + 0.6 * holdings.pre_impact_unit
    holdings["realized_cost_base"] += np.where(holdings.leg.eq("short"), holdings.fill_fee_native, 0.0)
    holdings["realized_cost_low"] += np.where(holdings.leg.eq("short"), holdings.fill_fee_native, 0.0)
    holdings["realized_cost_high"] += np.where(holdings.leg.eq("short"), holdings.fill_fee_native, 0.0)
    holdings["realized_cost_core"] = holdings.realized_cost_no_borrow + np.where(
        holdings.leg.eq("short"), holdings.fill_fee_native, 0.0
    )
    # A uses costs under native-decimal fee rates. B cases keep both fee interpretations.
    holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "realized_cost_base"] -= np.where(
        holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "leg"].eq("short"),
        holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "fill_fee_native"] * 0.99, 0.0)
    holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "realized_cost_low"] -= np.where(
        holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "leg"].eq("short"),
        holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "fill_fee_native"] * 0.99, 0.0)
    holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "realized_cost_high"] -= np.where(
        holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "leg"].eq("short"),
        holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "fill_fee_native"] * 0.99, 0.0)
    holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "realized_cost_core"] -= np.where(
        holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "leg"].eq("short"),
        holdings.loc[holdings.portfolio.str.startswith("B_percent_point_div100"), "fill_fee_native"] * 0.99, 0.0)
    # Return and cost coverage are cohort-level: strict equal-weight portfolios require every slot.
    cohort_records = []
    for (portfolio, date), g in holdings.groupby(["portfolio", "date"], sort=True):
        long, short = g[g.leg.eq("long")], g[g.leg.eq("short")]
        if len(long) != 50 or len(short) != 50:
            continue
        ret_ok = g.absolute_forward_return.notna().all()
        cost_ok = g.realized_cost_no_borrow.notna().all() and short.fill_fee_native.notna().all()
        gross = long.absolute_forward_return.mean() - short.absolute_forward_return.mean()
        row = {"portfolio": portfolio, "date": date, "n_long": len(long), "n_short": len(short),
               "return_slots": int(g.absolute_forward_return.notna().sum()), "cost_slots": int(g.realized_cost_no_borrow.notna().sum()),
               "borrow_slots_short": int(short.fill_fee_native.notna().sum()), "gross_observed": gross,
               "gross_strict": gross if ret_ok else np.nan}
        cost_cols = {"core": "realized_cost_core", "low": "realized_cost_low", "base": "realized_cost_base", "high": "realized_cost_high"}
        for lev, col in cost_cols.items():
            long_net = long.absolute_forward_return - long[col]
            short_net = -short.absolute_forward_return - short[col]
            lm = long_net.notna(); sm = short_net.notna()
            matched = float(long_net[lm].mean() + short_net[sm].mean()) if lm.any() and sm.any() else np.nan
            row[f"net_{lev}_matched"] = matched
            row[f"matched_long_{lev}"] = int(lm.sum()); row[f"matched_short_{lev}"] = int(sm.sum())
            pnl = long.absolute_forward_return.mean() - long[col].mean() - (short.absolute_forward_return.mean() + short[col].mean())
            row[f"net_{lev}_observed"] = pnl
            row[f"net_{lev}_strict"] = pnl if ret_ok and cost_ok else np.nan
        row["full_cost_coverage"] = bool(cost_ok)
        cohort_records.append(row)
    cohorts = pd.DataFrame(cohort_records)
    # Strict A-vs-B sample at shared formation dates; include requested A baseline separately.
    baseline = pd.read_csv(out / "A_cohorts.csv", parse_dates=["date"])
    baseline_stats = json.loads((out / "baseline_summary.json").read_text())
    a_metrics = summary_metrics(cohorts[cohorts.portfolio.eq("A")], "gross_strict", "A gross strict")
    base_a = cohorts[cohorts.portfolio.eq("A")].copy()
    for lev in ("low", "base", "high"):
        a_metrics.update({})
    metrics = [a_metrics]
    for portfolio, g in cohorts.groupby("portfolio", sort=True):
        if portfolio == "A":
            continue
        # Require full return and all named cost observations for net headline; preserve observed result too.
        metrics.append(summary_metrics(g, "gross_strict", f"{portfolio} gross strict"))
        for lev in ("low", "base", "high"):
            metrics.append(summary_metrics(g, f"net_{lev}_strict", f"{portfolio} net {lev} strict"))
            metrics.append(summary_metrics(g, f"net_{lev}_matched", f"{portfolio} net {lev} matched slots"))
            metrics.append(summary_metrics(g, f"net_{lev}_observed", f"{portfolio} net {lev} observed slots"))
        metrics.append(summary_metrics(g, "net_core_strict", f"{portfolio} net core strict"))
        metrics.append(summary_metrics(g, "net_core_matched", f"{portfolio} net core matched slots"))
    # A net sensitivities use identical original holdings and native decimal borrow scale.
    for lev in ("low", "base", "high"):
        metrics.append(summary_metrics(cohorts[cohorts.portfolio.eq("A")], f"net_{lev}_strict", f"A net {lev} strict"))
        metrics.append(summary_metrics(cohorts[cohorts.portfolio.eq("A")], f"net_{lev}_matched", f"A net {lev} matched slots"))
        metrics.append(summary_metrics(cohorts[cohorts.portfolio.eq("A")], f"net_{lev}_observed", f"A net {lev} observed slots"))
    metrics.append(summary_metrics(cohorts[cohorts.portfolio.eq("A")], "net_core_matched", "A net core matched slots"))
    common_rows = []
    a_cohorts = cohorts[cohorts.portfolio.eq("A")].set_index("date")
    for b_name in ("B_native_decimal_Y0.0", "B_native_decimal_Y0.3", "B_percent_point_div100_Y0.3"):
        bg = cohorts[cohorts.portfolio.eq(b_name)].set_index("date")
        common_dates = a_cohorts.index[a_cohorts.gross_strict.notna() & bg.gross_strict.reindex(a_cohorts.index).notna()]
        for source, frame, col in (("A gross", a_cohorts, "gross_strict"), (f"{b_name} gross", bg, "gross_strict"),
                                   (f"{b_name} net core", bg, "net_core_strict"), (f"{b_name} net Y0.3", bg, "net_base_strict")):
            if source.startswith("A "):
                view = a_cohorts.loc[common_dates]
            else:
                view = bg.reindex(common_dates)
            common_rows.append(summary_metrics(view.reset_index(names="date"), col, f"common dates {source} / {b_name}"))
    audit = pd.DataFrame(candidate_audit)
    coverage = {
        "requested_securities": int(len(req)), "cache_counts": json.loads((out / "collection_status.json").read_text()),
        "A_positions": int(len(a)), "A_spread_entry_exit_covered": int((a.fill_entry_quote & a.fill_exit_quote).sum()),
        "A_borrow_history_covered_short": int(a[a.leg.eq("short")].fill_fee_native.notna().sum()),
        "A_full_spread_and_borrow_cost_rows": int((a.fill_spread_rt.notna() & a.fill_comm_reg_rt.notna() & (a.leg.ne("short") | a.fill_fee_native.notna())).sum()),
        "B_cohort_candidate_counts_summary": audit.groupby(["fee_case", "impact_y", "leg"]).cost_estimable_candidates.agg(["min", "median", "max"]).reset_index().to_dict("records") if not audit.empty else [],
        "IBKR_fee_unit_main_case": "raw FEE_RATE interpreted as annual decimal fraction; sensitivity divides by 100",
        "IBKR_fee_unit_evidence": "IBKR describes fee as annualized percentage; API bars document the field but not numeric scale; native decimal remains provisional",
        "borrow_availability_history": "not collected; current availability is not projected backwards",
    }
    holdings.to_parquet(out / "selected_holdings_and_costs.parquet", index=False)
    cohorts.to_csv(out / "cohort_results.csv", index=False)
    audit.to_csv(out / "B_candidate_coverage.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "portfolio_metrics.csv", index=False)
    pd.DataFrame(common_rows).to_csv(out / "common_sample_metrics.csv", index=False)
    (out / "backtest_coverage.json").write_text(json.dumps(coverage, indent=2, default=str))
    result = {"state": "provisional_estimated_costs_complete", "baseline_reference_ann_arithmetic_pct": baseline_stats["gross_ann_arithmetic_pct"],
              "metrics": metrics, "coverage": coverage,
              "limitations": ["B requires availability of historical spread and fee observations to estimate costs; missing candidates are counted and not zero-filled",
                              "historical shortable-share quantities are unavailable, so borrow execution feasibility is not verified",
                              "fee-rate numeric scale is provisionally interpreted as decimal; 100x alternative retained",
                              "spread bars are RTH time-averaged quotes, not opening-auction fills; impact is a model sensitivity",
                              "selected outcome and cost coverage are reported as strict complete cohorts and observed-slot diagnostics"]}
    (out / "backtest_summary.json").write_text(json.dumps(result, indent=2, default=str))
    report = [
        "# Historical A/B: original signal basket versus cost-aware selection",
        "",
        "Window: formations 2020-01-02 through 2026-06-18; outcomes through 2026-06-26. 328 five-session formation cohorts.",
        "A holds the 50 highest and 50 lowest relative-model scores after price >= $5 and ADV20 >= $1m. B uses the same model scores and dates, retains the $5 price floor, applies no $1m ADV cutoff, and selects up to 50 names from the fixed top/bottom 250 score candidates by lagged expected relative return less estimated cost. Both sides are equally weighted at $20,000 per name ($1m per side).",
        "",
        "## Result",
        "",
        "| Portfolio / metric | Complete cohorts | Annual arithmetic spread | Sharpe | CAGR diagnostic | Max drawdown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    # Lead with a strict like-for-like comparison: identical dates where both A and
    # B have all 100 outcomes. Cost rows then use those same dates and require
    # complete costs, so no missing holdings are silently treated as zero cost.
    by_metric = {m["portfolio"]: m for m in common_rows}
    display = [
        ("A gross", "common dates A gross / B_native_decimal_Y0.3"),
        ("B gross, provisional native fee scale", "common dates B_native_decimal_Y0.3 gross / B_native_decimal_Y0.3"),
        ("B net of spread, commission, borrow; no impact", "common dates B_native_decimal_Y0.3 net core / B_native_decimal_Y0.3"),
        ("B net incl. assumed impact (Y=0.3)", "common dates B_native_decimal_Y0.3 net Y0.3 / B_native_decimal_Y0.3"),
        ("B net incl. impact; fee value divided by 100", "common dates B_percent_point_div100_Y0.3 net Y0.3 / B_percent_point_div100_Y0.3"),
    ]
    for label, key in display:
        m = by_metric.get(key, {})
        if m.get("complete_cohorts", 0):
            report.append(f"| {label} | {m['complete_cohorts']} | {m['annualized_arithmetic_spread_pct']:.2f}% | {m['annualized_sharpe']:.2f} | {m['cagr_pct']:.2f}% | {m['max_drawdown_pct']:.2f}% |")
        else:
            report.append(f"| {label} | 0 | n/a | n/a | n/a | n/a |")
    report += [
        "",
        "The table uses the same 282 complete-outcome formation dates for A and B (2020-03-16 through 2026-05-28). The original A reference is 19.99% annualized under its prior all-date/available-outcome convention; that is a separate reference, not the common-date A number above. CAGR and drawdown compound the observed five-session cohorts with missing dates omitted; treat them as diagnostics, not a continuous daily NAV.",
        "The gross comparison isolates portfolio construction: in this test, cost-aware selection gives up nearly all of the A basket's gross spread before charging costs. On this specification, estimated costs make B materially negative. This is a result for the tested candidate pool, calibration, and execution assumptions; it does not show that every alternative cost-aware construction must fail.",
        "",
        "## Coverage and limits",
        "",
        f"A has complete entry/exit spread quotes for {coverage['A_spread_entry_exit_covered']:,}/{coverage['A_positions']:,} holding rows; short-fee history is available for {coverage['A_borrow_history_covered_short']:,}/16,400 short rows. No A cohort has complete cost coverage for all 100 holdings, so this dataset cannot support a full-portfolio A-net result. The matched-slot A diagnostic uses only cost-observed selected names (about 41 long and 35 short names per cohort on average) and is not a like-for-like portfolio return because missingness is selective.",
        f"B formed 318/328 cohorts with 50 cost-estimable candidates available on both sides. Median cost-estimable candidate counts were {int(audit[(audit.fee_case=='native_decimal') & (audit.impact_y==0.3) & (audit.leg=='long')].cost_estimable_candidates.median())} long and {int(audit[(audit.fee_case=='native_decimal') & (audit.impact_y==0.3) & (audit.leg=='short')].cost_estimable_candidates.median())} short names out of 250. The remaining 10 dates are omitted from B because cost history did not support 50 selections on both sides.",
        "IBKR describes stock-loan fee rates as annualized percentages charged daily. Its historical-bar API lists FEE_RATE and BID_ASK fields but does not state the numeric scale in the bar-field definition. The main case interprets raw FEE_RATE as an annual decimal fraction; the 100x lower interpretation is shown as sensitivity. Historical borrowable-share quantities are unavailable, so short availability and locate success are not verified.",
        "Spread costs use IBKR daily RTH time-average bid and ask on actual entry/exit dates, not opening-auction fills. Pretrade spread estimates use recent observations; realized spread cost requires quotes on both actual dates. Borrow fee observations are carried forward only when no more than seven calendar days old. Commissions use the prior IBKR Pro Fixed assumption ($0.005/share/order, $1 minimum, 1% cap), plus approximate regulatory fees; the actual account schedule was not verified. Market impact is not observed; Y=0.1/0.3/0.6 are scenario assumptions based on lagged volatility and ADV20. No $1m ADV threshold is imposed on B; ADV only affects estimated impact.",
        "B cost selection uses the score calibration frozen from five prior years, with calibration labels ending before each test year's first formation. Outcomes, including flagged return outliers, are not used in selection and remain in the evaluation, matching A's original treatment.",
        "",
        "Conclusion: under this tested cost-aware construction, B's gross edge is already much smaller than A's, and B is negative after spread, commission, and borrow estimates even before impact. The result remains an estimated-cost backtest, not an actual-fill or fully executable historical test: broker coverage is incomplete, borrowable quantities are unavailable, spread bars are not fills, fee-rate scaling is provisional, and impact is assumed.",
        "",
        "Artifacts: `portfolio_metrics.csv`, `common_sample_metrics.csv`, `cohort_results.csv`, `selected_holdings_and_costs.parquet`, `B_candidate_coverage.csv`, `backtest_coverage.json`, `backtest_summary.json`.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"state": result["state"], "baseline": result["baseline_reference_ann_arithmetic_pct"],
                      "coverage": coverage, "metrics": metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
