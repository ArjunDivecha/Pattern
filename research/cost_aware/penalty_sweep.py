"""Controlled selection-cost sensitivity; full costs charged for every penalty.

Uses saved scores and broker histories only. No retraining or broker requests.
Missing future outcomes/costs never influence candidate eligibility or selection.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.cost_aware.backtest import (  # noqa: E402
    bar_arrays, cost_rows_for_security, summary_metrics,
)

SOURCE = ROOT / "runs/cost_aware_ab_2020_2026_1m"
OUT = SOURCE / "penalty_sweep"
PENALTIES = (0.0, 0.25, 0.5, 1.0)
CAPS = (75, 100, 250)
COST_VERSION = "matured_sigma_calendar_borrow_v1"


def matured_sigma(panel):
    """Last 20 finite, fully realized holding returns available by formation close."""
    pieces = []
    for _, g in panel.groupby("security", sort=False):
        g = g.sort_values("date")
        history = g[g.exit_date.notna() & np.isfinite(g.absolute_forward_return)].sort_values(
            ["exit_date", "date"]
        )
        value = np.full(len(g), np.nan)
        latest = np.full(len(g), np.datetime64("NaT"), dtype="datetime64[ns]")
        if len(history):
            exit_dates = history.exit_date.to_numpy(dtype="datetime64[ns]")
            indices = np.searchsorted(exit_dates, g.date.to_numpy(dtype="datetime64[ns]"), side="right") - 1
            std = history.absolute_forward_return.rolling(20, min_periods=10).std().to_numpy() / np.sqrt(5)
            valid = indices >= 0
            value[valid] = std[indices[valid]]
            latest[valid] = exit_dates[indices[valid]]
        result = g[["security", "date"]].copy()
        result["sigma_daily"] = value
        result["sigma_latest_exit"] = latest
        pieces.append(result)
    answer = pd.concat(pieces, ignore_index=True)
    assert (answer.sigma_latest_exit.isna() | (answer.sigma_latest_exit <= answer.date)).all()
    return answer


def pretrade_fields(candidates):
    c = candidates.copy()
    c["pre_cost"] = c.pre_spread_rt + c.pre_comm_reg_rt + 0.3 * c.pre_impact_unit
    short = c.leg.eq("short")
    c.loc[short, "pre_cost"] += (
        1.02 * c.loc[short, "pre_fee_native"] * c.loc[short, "scheduled_days"] / 360
    )
    c["alpha"] = np.where(short, -c.expected_relative_return, c.expected_relative_return)
    # Saved ranks include the original score/security tie convention.
    c["signal_strength"] = np.where(short, -c["rank"], c["rank"])
    c["pretrade_eligible"] = (
        np.isfinite(c.pre_cost) & (c.pre_cost >= 0) & np.isfinite(c.alpha)
        & np.isfinite(c.signal_strength) & (c.pre_adv20 > 0)
    )
    return c


def select(group, penalty, cap, n=50):
    """The cap is a rank boundary inside the fixed pretrade-eligible universe."""
    pool = group[group.pretrade_eligible].sort_values(
        ["signal_strength", "security"], ascending=[False, True]
    ).copy()
    pool["eligible_signal_rank"] = np.arange(1, len(pool) + 1)
    pool = pool.head(cap)
    if len(pool) < n:
        return pool.iloc[:0]
    pool["objective"] = pool.alpha - penalty * pool.pre_cost
    # Preserve model order within flat calibration plateaus, including lambda=0.
    return pool.sort_values(
        ["objective", "signal_strength", "security"], ascending=[False, False, True]
    ).head(n)


def add_realized_costs(c):
    c = c.copy()
    c["borrow_cost"] = np.where(c.leg.eq("short"), c.fill_fee_native, 0.0)
    c["impact_cost"] = 0.3 * c.pre_impact_unit
    c["core_cost"] = c.fill_spread_rt + c.fill_comm_reg_rt + c.borrow_cost
    c["total_cost"] = c.core_cost + c.impact_cost
    return c


def cohort(group, n=50):
    long = group[group.leg.eq("long")]
    short = group[group.leg.eq("short")]
    if len(long) != n or len(short) != n:
        raise ValueError("Both sides must retain the specified number of names")
    returns_ok = np.isfinite(group.absolute_forward_return).all()
    costs_ok = np.isfinite(group.total_cost).all()
    gross = long.absolute_forward_return.mean() - short.absolute_forward_return.mean() if returns_ok else np.nan
    fields = {
        "gross": gross, "complete": bool(returns_ok and costs_ok),
        "return_slots": int(np.isfinite(group.absolute_forward_return).sum()),
        "cost_slots": int(np.isfinite(group.total_cost).sum()),
    }
    for name, col in (("spread", "fill_spread_rt"), ("commission", "fill_comm_reg_rt"),
                      ("borrow", "borrow_cost"), ("impact", "impact_cost")):
        fields[name] = long[col].mean() + short[col].mean() if np.isfinite(group[col]).all() else np.nan
    # Selection penalty NEVER appears in the cost accounting below.
    fields["net"] = gross - sum(fields[k] for k in ("spread", "commission", "borrow", "impact")) if fields["complete"] else np.nan
    fields["net_no_impact"] = gross - fields["spread"] - fields["commission"] - fields["borrow"] if fields["complete"] else np.nan
    fields["net_fee_div100"] = fields["net"] + 0.99 * fields["borrow"] if fields["complete"] else np.nan
    fields["net_forecast_cost"] = gross - long.pre_cost.mean() - short.pre_cost.mean() if returns_ok else np.nan
    return fields


def fingerprint():
    files = [SOURCE / "B_candidates.parquet", SOURCE / "eligible_panel.parquet",
             Path(__file__).with_name("backtest.py")]
    return {
        "cost_version": COST_VERSION,
        "sources": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "broker_cache_metadata_hash": hashlib.sha256(json.dumps([
            (p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted((SOURCE / "broker_history").glob("*.json"))
        ]).encode()).hexdigest(),
    }


def load_candidates():
    OUT.mkdir(exist_ok=True)
    sig = fingerprint()
    cache = OUT / "cost_candidates.parquet"
    manifest = OUT / "cost_sources.json"
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == sig:
        print("Reusing verified candidate cost cache", flush=True)
        return pd.read_parquet(cache)
    b = pd.read_parquet(SOURCE / "B_candidates.parquet")
    panel = pd.read_parquet(SOURCE / "eligible_panel.parquet")
    sigma = matured_sigma(panel)
    b = b.merge(sigma, on=["security", "date"], validate="one_to_one")
    # Planned holding calendar is common to all names, even names with missing future returns.
    valid = panel[panel.entry_date.notna() & panel.exit_date.notna()]
    assert valid.groupby("date")[["entry_date", "exit_date"]].nunique().le(1).all().all()
    schedule = valid[["date", "entry_date", "exit_date"]].drop_duplicates("date")
    schedule["scheduled_days"] = (schedule.exit_date - schedule.entry_date).dt.days
    b = b.merge(schedule[["date", "scheduled_days"]], on="date", validate="many_to_one")
    assert b.scheduled_days.gt(0).all()
    rows = []
    groups = b.groupby("security", sort=False)
    for i, (sec, g) in enumerate(groups):
        path = SOURCE / "broker_history" / (hashlib.sha256(sec.encode()).hexdigest()[:20] + ".json")
        bars = bar_arrays(json.loads(path.read_text()) if path.exists() else {})
        rows.extend(cost_rows_for_security(sec, g, bars))
        if (i + 1) % 1000 == 0:
            print(f"Calculated historical inputs for {i+1}/{len(groups)} securities", flush=True)
    b = b.merge(pd.DataFrame(rows), on=["security", "date", "entry_date", "exit_date"], validate="one_to_one")
    b = add_realized_costs(pretrade_fields(b))
    b.to_parquet(cache, index=False)
    manifest.write_text(json.dumps(sig, indent=2) + "\n")
    return b


def main():
    b = load_candidates()
    selected, records, coverage = [], [], []
    specs = [(cap, penalty, f"cap{cap}_lambda{penalty:g}") for cap in CAPS for penalty in PENALTIES]
    for date, day in b.groupby("date", sort=True):
        sizes = day.groupby("leg").pretrade_eligible.sum()
        coverage.append({"date": date, "long_eligible": int(sizes.get("long", 0)),
                         "short_eligible": int(sizes.get("short", 0))})
        if sizes.min() < 50 or len(sizes) != 2:
            continue
        baseline_keys = set()
        for leg, g in day.groupby("leg"):
            baseline_keys.update((leg, s) for s in select(g, 0.0, 250).security)
        for cap, penalty, name in specs:
            chosen = pd.concat([select(g, penalty, cap) for _, g in day.groupby("leg")], ignore_index=True)
            assert len(chosen) == 100 and chosen.groupby("leg").size().eq(50).all()
            assert not chosen.security.duplicated().any()
            chosen["portfolio"] = name
            chosen["penalty"] = penalty
            chosen["rank_cap"] = cap
            chosen["retained_signal_baseline"] = [(leg, sec) in baseline_keys for leg, sec in zip(chosen.leg, chosen.security)]
            row = {"date": date, "portfolio": name, "penalty": penalty, "rank_cap": cap, **cohort(chosen)}
            row["retained_per_side"] = chosen.retained_signal_baseline.sum() / 2
            row["mean_signal_rank"] = chosen.eligible_signal_rank.mean()
            records.append(row)
            selected.append(chosen)
    cohorts = pd.DataFrame(records)
    holdings = pd.concat(selected, ignore_index=True)
    # Selection uses only pretrade coverage. Future data availability is used ONLY here,
    # after saving all portfolios, to identify a shared strict evaluation sample.
    complete = cohorts.pivot(index="date", columns="portfolio", values="complete").fillna(False).all(axis=1)
    a = pd.read_csv(SOURCE / "A_cohorts.csv", parse_dates=["date"]).set_index("date")
    dates = complete.index[complete & a.strict_gross_spread.reindex(complete.index).notna()]
    if len(dates) == 0:
        raise RuntimeError("No common complete cohorts; report coverage rather than zero-fill")
    cohorts["common_sample"] = cohorts.date.isin(dates)
    holdings["common_sample"] = holdings.date.isin(dates)
    metrics, per_year = [], []
    for (cap, penalty, name) in specs:
        g = cohorts[(cohorts.portfolio == name) & cohorts.common_sample]
        row = {"portfolio": name, "penalty": penalty, "rank_cap": cap, "cohorts": len(g),
               "retained_per_side": g.retained_per_side.mean(), "mean_signal_rank": g.mean_signal_rank.mean()}
        for field in ("gross", "net", "net_no_impact", "net_fee_div100", "net_forecast_cost"):
            m = summary_metrics(g, field, name)
            row[field + "_ann_pct"] = m["annualized_arithmetic_spread_pct"]
            if field == "net":
                row["net_sharpe"] = m["annualized_sharpe"]
                row["net_max_drawdown_diagnostic_pct"] = m["max_drawdown_pct"]
        for k in ("spread", "commission", "borrow", "impact"):
            row[k + "_ann_pct"] = g[k].mean() * 5040
        metrics.append(row)
        for year, gy in g.groupby(g.date.dt.year):
            per_year.append({"portfolio": name, "year": int(year), "cohorts": len(gy),
                             "gross_ann_pct": gy.gross.mean()*5040, "net_ann_pct": gy.net.mean()*5040})
    metrics = pd.DataFrame(metrics)
    a_common = a.loc[dates].reset_index()
    a_metric = summary_metrics(a_common, "strict_gross_spread", "Original A gross; same dates")
    # Calibration monotonicity + score tiebreak ensure all zero-cost-penalty caps agree.
    for cap in CAPS:
        x = holdings[holdings.portfolio.eq(f"cap{cap}_lambda0")]
        ref = holdings[holdings.portfolio.eq("cap250_lambda0")]
        assert set(zip(x.date, x.leg, x.security)) == set(zip(ref.date, ref.leg, ref.security))
    assert all(cohorts[cohorts.portfolio.eq(n)].date.isin(dates).sum() == len(dates) for _, _, n in specs)
    holdings.to_parquet(OUT / "holdings.parquet", index=False)
    cohorts.to_csv(OUT / "cohorts.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(per_year).to_csv(OUT / "yearly_metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "candidate_coverage.csv", index=False)
    pd.DataFrame({"date": dates}).to_csv(OUT / "common_dates.csv", index=False)
    signal_only = holdings[holdings.portfolio.eq("cap250_lambda0") & holdings.common_sample]
    quote_outliers = signal_only[signal_only.fill_spread_rt.gt(.10)].sort_values("fill_spread_rt", ascending=False)
    quote_outliers[["security", "date", "leg", "close", "adv20", "pre_spread_rt", "fill_spread_rt", "absolute_forward_return"]].to_csv(OUT / "quote_outlier_audit.csv", index=False)
    price_manifest = json.loads((ROOT / "data/research_universe/clean_I20_R5_v1/manifest.json").read_text())
    sample = {
        "requested_dates": int(b.date.nunique()), "formed_dates": int(cohorts.date.nunique()),
        "common_complete_dates": len(dates), "first_common_date": str(dates.min().date()),
        "last_common_date": str(dates.max().date()), "original_A_gross_same_dates": a_metric,
        "formed_but_excluded_dates": sorted(str(d.date()) for d in set(cohorts.date) - set(dates)),
        "cost_version": COST_VERSION, "net_is_full_cost_for_all_penalties": True,
        "saved_price_basis": price_manifest["price_basis"],
        "signal_only_quote_rows_above_10pct": int(len(quote_outliers)),
        "signal_only_fraction_ADV_below_1m": float(signal_only.adv20.lt(1e6).mean()),
        "accounting": "long-minus-short return and summed leg costs, normalized by $1m side notional; divide by two for $2m gross-capital basis",
        "calibration_tiebreak": "original score, then security; lambda0 reproduces signal ranking in the covered universe",
        "candidate_universe": "fixed top/bottom250 price>=5 signal names; common prior-data cost eligibility, no ADV floor",
        "rank_caps": "first75/100/250 of cost-estimable candidates in original signal order; retain at least25/0/0 of baseline50 names per leg",
        "timing_corrections": ["volatility uses only outcomes with exit<=formation close", "expected borrow duration uses common planned holding calendar even when a name lacks outcomes"],
        "limitations": ["shared dates are selected ex post for measurement completeness, not eligibility; missingness may bias evaluation", "fee scale provisional; fixed-holdings fee/100 sensitivity reported", "historical short availability and actual fills unverified", "impact and commission assumptions retained; full round trips charged without turnover netting", "saved $5 screen is on historically adjusted prices, not verified contemporaneous nominal prices; share-based commission estimates are also sensitive to price adjustment", "daily time-average bid/ask bars contain extreme values; forecast-cost sensitivity is reported without deleting or capping quotes", "parameter comparison on previously examined history, not new out-of-sample validation"],
    }
    (OUT / "summary.json").write_text(json.dumps(sample, indent=2, default=str) + "\n")
    make_report(metrics, cohorts, sample)
    print(json.dumps(sample, indent=2, default=str), flush=True)
    print(metrics.round(3).to_string(index=False), flush=True)


def make_report(metrics, cohorts, sample):
    lines = ["# Selection-cost penalty experiment", "",
             f"$1m per side, 50 names per side. Same {sample['common_complete_dates']} complete cohorts for every row, {sample['first_common_date']} through {sample['last_common_date']}.", "",
             "Select by expected directional relative return minus lambda times estimated round-trip cost. Every portfolio is charged 100% of estimated realized spread, commission, borrow and impact costs. Lambda changes selection only. Rank caps are measured among the same pretrade cost-estimable signal candidates. The 75-name cap guarantees at least 25 of the original covered top50 names survive per side.", "",
             "| Rank cap | Selection penalty | Gross annual spread | Net annual spread | Net Sharpe | Retained names / side |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in metrics.itertuples():
        lines.append(f"| {r.rank_cap} | {r.penalty:.0%} | {r.gross_ann_pct:.2f}% | {r.net_ann_pct:.2f}% | {r.net_sharpe:.2f} | {r.retained_per_side:.1f} / 50 |")
    lines += ["", f"Original A gross on these same dates: {sample['original_A_gross_same_dates']['annualized_arithmetic_spread_pct']:.2f}% annualized. A has the original $1m ADV floor. It is not the covered-universe lambda0 baseline, and its net result remains unavailable because of missing cost histories.", "",
              f"Formation coverage: {sample['formed_dates']}/{sample['requested_dates']} dates could form all portfolios; {sample['common_complete_dates']} dates have all selected outcomes/costs and complete A outcomes. Selection never uses future cost coverage. The shared evaluation dates exclude incomplete outcomes/costs after selection; this is a missing-data restriction, not a full-period net result.", "",
              "Timing corrections applied equally to every row: prior volatility now uses only completed holding returns, and pretrade borrow duration uses the common trading calendar. Therefore compare variants within this experiment, rather than attributing differences from the earlier report solely to lambda. Flat return-calibration estimates are tied using the original model score.", "",
              "Returns are arithmetic annualized five-session long-short spreads on $1m side notional (100% long + 100% short). On $2m gross-capital basis, divide returns and cost percentages by two. Any saved drawdown is an observed-cohort diagnostic, not daily NAV. Full round-trip costs are charged each cohort without netting continuing positions.", "",
              f"The source manifest explicitly describes the $5 price screen as '{sample['saved_price_basis']}'. This does not guarantee removal of historical penny stocks. {sample['signal_only_fraction_ADV_below_1m']:.1%} of the covered signal-only positions have ADV below $1m, as permitted by the requested no-ADV-floor experiment. Price adjustment also limits the accuracy of share-based commissions.", "",
              f"There are {sample['signal_only_quote_rows_above_10pct']} signal-only position rows whose daily-quote spread proxy exceeds 10% of notional. One example is EBMT on exit 2020-12-07: average bid 21.79, average ask 227.06, and maximum ask 200000. These are not dependable actual execution-cost measurements. No quotes were deleted or capped in the table. quote_outlier_audit.csv records these cases; metrics.csv also reports net_forecast_cost_ann_pct, using the full pretrade cost forecast, which uses trailing median spreads and is less sensitive to single-day spikes.", "",
              "Fee scaling remains provisional; metrics.csv also reports no-impact net and fee/100 net for the same holdings. Spread inputs are historical daily quote proxies; commissions and impact are assumptions. Historical short availability and actual fills are unverified. These comparisons reuse previously examined history and are exploratory, not a fresh out-of-sample result.", "",
              "![Gross versus net and net-return sensitivity](penalty_comparison.png)", "",
              "Artifacts: metrics.csv, yearly_metrics.csv, cohorts.csv, holdings.parquet, candidate_coverage.csv, common_dates.csv, summary.json, cost_sources.json."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for cap, g in metrics.groupby("rank_cap"):
        axes[0].plot(g.penalty*100, g.gross_ann_pct, marker="o", label=f"Choose 50 from first {cap}")
        positive = g[g.penalty.gt(0)]
        axes[1].plot(positive.penalty*100, positive.net_ann_pct, marker="o", label=f"Choose 50 from first {cap}")
    axes[0].axhline(0, color="#777777", linewidth=.8)
    axes[0].set(title="Gross signal retained", xlabel="Cost weight used for selection (%)", ylabel="Annualized long-short spread (%)")
    axes[0].legend(frameon=False, fontsize=9)
    zero = metrics[metrics.penalty.eq(0)].iloc[0].net_ann_pct
    axes[1].set(title="Net cost proxy: positive selection penalties", xlabel="Cost weight used for selection (%)", ylabel="Annualized long-short spread (%)")
    axes[1].text(.02, .02, f"0% selection penalty: {zero:.1f}% net (outside panel)", transform=axes[1].transAxes, fontsize=8)
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.set_xticks([0,25,50,100])
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xticks([25,50,100])
    fig.suptitle(f"50 long / 50 short · $1m per side · {sample['common_complete_dates']} shared cohorts", fontsize=14)
    fig.savefig(OUT / "penalty_comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
