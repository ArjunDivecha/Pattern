"""Post-hoc top/bottom-20 analysis for the completed model scores."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/full_history_single_2019cutoff_seed101"
SOURCE = BASE / "comparison_to_rolling5"
OUT = SOURCE / "top_bottom_20"
sys.path.insert(0, str(ROOT))
from research.full_history.compare_single_to_rolling import build_characteristics  # noqa: E402


def load_scores() -> dict[str, pd.DataFrame]:
    single = pd.read_parquet(BASE / "scores_rolling_schedule.parquet")
    rolling_parts = []
    for year in range(2020, 2027):
        p = ROOT / f"runs/full_rolling_clean_2003_2026_seed101/{year}_rolling5_seed101/scores.parquet"
        rolling_parts.append(pd.read_parquet(p))
    rolling = pd.concat(rolling_parts, ignore_index=True)
    for x in (single, rolling):
        x["date"] = pd.to_datetime(x.date)
        x.drop_duplicates(["security", "date"], keep="last", inplace=True)
    return {"single_full_history": single, "rolling5": rolling}


def select_tail(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    rows = []
    for date, g in df.groupby("date", sort=True):
        g = g.sort_values(["score", "security"]).copy()
        if len(g) < 2 * n:
            continue
        low = g.head(n).assign(leg="bottom")
        high = g.tail(n).assign(leg="top")
        rows.extend([low, high])
    return pd.concat(rows, ignore_index=True)


def metrics(assign: pd.DataFrame, n: int) -> tuple[dict, pd.DataFrame]:
    rows = []
    for date, g in assign.groupby("date", sort=True):
        top = g.loc[g.leg.eq("top"), "forward_return"]
        bottom = g.loc[g.leg.eq("bottom"), "forward_return"]
        rows.append({
            "date": date,
            "top": top.mean(), "bottom": bottom.mean(), "spread": top.mean() - bottom.mean(),
            "top_n": len(top), "bottom_n": len(bottom),
            "top_known": top.notna().sum(), "bottom_known": bottom.notna().sum(),
            "strict_spread": top.mean() - bottom.mean() if top.notna().all() and bottom.notna().all() else np.nan,
        })
    c = pd.DataFrame(rows).sort_values("date")
    ppy = 252 / 5
    def sharpe(s):
        s = s.dropna()
        return float(s.mean() / s.std(ddof=1) * np.sqrt(ppy)) if len(s) > 1 and s.std(ddof=1) > 0 else np.nan
    wealth = (1 + c.spread.fillna(0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return {
        "cohorts": int(len(c)), "tail_n_per_side": n,
        "top_ann_arithmetic": float(c.top.mean() * ppy),
        "bottom_ann_arithmetic": float(c.bottom.mean() * ppy),
        "spread_ann_arithmetic": float(c.spread.mean() * ppy),
        "spread_vol_ann": float(c.spread.std(ddof=1) * np.sqrt(ppy)),
        "spread_sharpe": sharpe(c.spread), "strict_spread_sharpe": sharpe(c.strict_spread),
        "spread_cagr": float(wealth.iloc[-1] ** (ppy / len(c)) - 1),
        "spread_max_drawdown": float(drawdown.min()),
        "positive_spread_fraction": float(c.spread.gt(0).mean()),
        "unresolved_extreme_cohorts": int(c.strict_spread.isna().sum()),
        "known_top_fraction": float(c.top_known.sum() / c.top_n.sum()),
        "known_bottom_fraction": float(c.bottom_known.sum() / c.bottom_n.sum()),
    }, c


def turnover(assign: pd.DataFrame, leg: str) -> dict:
    sets = {d: set(g.loc[g.leg.eq(leg), "security"]) for d, g in assign.groupby("date")}
    dates = sorted(sets)
    x = [len(sets[a] ^ sets[b]) / (len(sets[a]) + len(sets[b])) for a, b in zip(dates, dates[1:])]
    return {"mean_replaced_fraction": float(np.mean(x)), "median_replaced_fraction": float(np.median(x))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    assignments, summaries = {}, {}
    cohort_tables = {}
    for name, frame in scores.items():
        a = select_tail(frame, 20)
        assignments[name] = a
        summaries[name], cohort_tables[name] = metrics(a, 20)
        cohort_tables[name].to_csv(OUT / f"{name}_cohorts.csv", index=False)
        a.to_parquet(OUT / f"{name}_assignments.parquet", index=False)

    # Year-by-year spread diagnostics.
    annual = []
    for name, c in cohort_tables.items():
        for year, g in c.groupby(c.date.dt.year):
            annual.append({
                "model": name, "year": int(year), "cohorts": len(g),
                "top_ann_arithmetic": g.top.mean() * 50.4,
                "bottom_ann_arithmetic": g.bottom.mean() * 50.4,
                "spread_ann_arithmetic": g.spread.mean() * 50.4,
                "spread_sharpe": g.spread.mean() / g.spread.std(ddof=1) * np.sqrt(50.4),
                "positive_fraction": g.spread.gt(0).mean(),
            })
    pd.DataFrame(annual).to_csv(OUT / "annual_metrics.csv", index=False)

    # Exact top/bottom overlap across model families.
    l = assignments["single_full_history"]
    r = assignments["rolling5"]
    overlap_rows = []
    for date in sorted(set(l.date) & set(r.date)):
        for leg in ["top", "bottom"]:
            a = set(l.loc[(l.date == date) & l.leg.eq(leg), "security"])
            b = set(r.loc[(r.date == date) & r.leg.eq(leg), "security"])
            overlap_rows.append({"date": date, "leg": leg, "overlap": len(a & b), "jaccard": len(a & b) / len(a | b), "fraction_of_single": len(a & b) / len(a)})
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(OUT / "overlap_by_date.csv", index=False)

    # Trailing characteristics are measured at formation and use only history through that date.
    selected = pd.concat([a.assign(model=name) for name, a in assignments.items()], ignore_index=True)
    selected = build_characteristics(selected)
    feature_cols = ["close", "log_dollar_volume_20", "ret_20", "ret_60", "mom_12_1", "vol_20_ann", "ma200_gap"]
    char_rows = []
    for (model, leg), g in selected.groupby(["model", "leg"], observed=True):
        row = {"model": model, "leg": leg, "rows": len(g)}
        for col in feature_cols:
            x = g[col].dropna()
            row[f"{col}_median"] = x.median() if len(x) else np.nan
            row[f"{col}_p25"] = x.quantile(0.25) if len(x) else np.nan
            row[f"{col}_p75"] = x.quantile(0.75) if len(x) else np.nan
        char_rows.append(row)
    pd.DataFrame(char_rows).to_csv(OUT / "characteristics_summary.csv", index=False)
    selected.to_parquet(OUT / "selected_characteristics.parquet", index=False)

    turnover_rows = []
    for name, a in assignments.items():
        for leg in ["top", "bottom"]:
            turnover_rows.append({"model": name, "leg": leg, **turnover(a, leg)})
    pd.DataFrame(turnover_rows).to_csv(OUT / "turnover_summary.csv", index=False)

    summary = {"models": summaries, "overlap": {
        "mean_top_overlap": float(overlap.loc[overlap.leg.eq("top"), "fraction_of_single"].mean()),
        "mean_bottom_overlap": float(overlap.loc[overlap.leg.eq("bottom"), "fraction_of_single"].mean()),
        "mean_top_jaccard": float(overlap.loc[overlap.leg.eq("top"), "jaccard"].mean()),
        "mean_bottom_jaccard": float(overlap.loc[overlap.leg.eq("bottom"), "jaccard"].mean()),
    }}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
