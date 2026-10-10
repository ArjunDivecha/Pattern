"""Quality-filtered top/bottom-20 portfolios with a rank buffer.

This is a post-hoc portfolio construction experiment over already-generated
scores.  The model and score dates are unchanged.  At each formation date we
require only the requested $5 price floor, then rank by the model score.  The
buffered portfolio keeps an existing holding while it remains inside the
current top/bottom buffer and fills any vacancies from the score-ranked
candidates.

The return-outlier flag is never used for selection because it is based on a
future outcome.  It is joined afterwards only for a disclosed sensitivity.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SINGLE = ROOT / "runs/full_history_single_2019cutoff_seed101"
ROLLING = ROOT / "runs/full_rolling_clean_2003_2026_seed101"
OUT = SINGLE / "comparison_to_rolling5/quality_tail20_buffer"
PRICE_PATH = ROOT / "data/frozen/pattern_daily_v1_20260918/prices.parquet"
ELIGIBILITY_PATH = ROOT / "data/research_universe/clean_I20_R5_v1/eligibility.parquet"

N = 20
BUFFER = 40
PRICE_MIN = 5.0
PPY = 252 / 5

sys.path.insert(0, str(ROOT))
from research.full_history.compare_single_to_rolling import build_characteristics  # noqa: E402


def load_scores() -> dict[str, pd.DataFrame]:
    single = pd.read_parquet(SINGLE / "scores_rolling_schedule.parquet")
    rolling = pd.concat(
        [
            pd.read_parquet(ROLLING / f"{year}_rolling5_seed101/scores.parquet")
            for year in range(2020, 2027)
        ],
        ignore_index=True,
    )
    result = {}
    for name, frame in {
        "single_full_history": single,
        "rolling5": rolling,
    }.items():
        x = frame.copy()
        x["date"] = pd.to_datetime(x["date"])
        x = x[(x["date"] >= "2020-01-01") & (x["date"] <= "2026-06-30")]
        x = x.drop_duplicates(["security", "date"], keep="last")
        result[name] = x
    return result


def add_quality_features(scores: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    keys = pd.concat(
        [x[["security", "date"]] for x in scores.values()], ignore_index=True
    ).drop_duplicates()
    key_path = OUT / "_quality_keys.parquet"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    keys.to_parquet(key_path, index=False)
    con = duckdb.connect()
    query = f"""
    WITH px AS (
      SELECT security, date, close
      FROM read_parquet('{PRICE_PATH}')
    )
    SELECT k.security, k.date, px.close
    FROM read_parquet('{key_path}') AS k
    LEFT JOIN px USING (security, date)
    """
    quality = con.execute(query).fetchdf()
    con.close()
    key_path.unlink(missing_ok=True)
    enriched: dict[str, pd.DataFrame] = {}
    for name, frame in scores.items():
        x = frame.merge(quality, on=["security", "date"], how="left", validate="many_to_one")
        x["quality_eligible"] = x["close"].ge(PRICE_MIN)
        enriched[name] = x
    return enriched, quality


def select_ranked(frame: pd.DataFrame, n: int = N) -> pd.DataFrame:
    rows = []
    for date, g in frame[frame.quality_eligible].groupby("date", sort=True):
        asc = g.sort_values(["score", "security"], ascending=[True, True])
        if len(asc) < 2 * n:
            continue
        rows.extend(
            [
                asc.head(n).assign(leg="bottom", retained=False, rank_buffer=n),
                asc.tail(n).assign(leg="top", retained=False, rank_buffer=n),
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def select_buffered(frame: pd.DataFrame, n: int = N, buffer: int = BUFFER) -> pd.DataFrame:
    rows = []
    previous_top: set[str] = set()
    previous_bottom: set[str] = set()
    for date, raw in frame[frame.quality_eligible].groupby("date", sort=True):
        asc = raw.sort_values(["score", "security"], ascending=[True, True]).copy()
        if len(asc) < 2 * n:
            continue
        bottom_pool = asc.head(buffer)
        top_pool = asc.tail(buffer)

        top_keep = top_pool[top_pool.security.isin(previous_top)].sort_values(
            ["score", "security"], ascending=[False, True]
        )
        top_fill = top_pool[~top_pool.security.isin(set(top_keep.security))].sort_values(
            ["score", "security"], ascending=[False, True]
        )
        top = pd.concat([top_keep, top_fill], ignore_index=True).head(n).copy()
        top["leg"] = "top"
        top["retained"] = top["security"].isin(previous_top)
        top["rank_buffer"] = buffer

        bottom_pool = bottom_pool[~bottom_pool.security.isin(set(top.security))]
        bottom_keep = bottom_pool[bottom_pool.security.isin(previous_bottom)].sort_values(
            ["score", "security"], ascending=[True, True]
        )
        bottom_fill = bottom_pool[~bottom_pool.security.isin(set(bottom_keep.security))].sort_values(
            ["score", "security"], ascending=[True, True]
        )
        bottom = pd.concat([bottom_keep, bottom_fill], ignore_index=True).head(n).copy()
        bottom["leg"] = "bottom"
        bottom["retained"] = bottom["security"].isin(previous_bottom)
        bottom["rank_buffer"] = buffer

        if len(top) == n and len(bottom) == n:
            rows.extend([top, bottom])
            previous_top = set(top.security)
            previous_bottom = set(bottom.security)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def portfolio_metrics(assign: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    for date, g in assign.groupby("date", sort=True):
        top = g.loc[g.leg.eq("top"), "forward_return"]
        bottom = g.loc[g.leg.eq("bottom"), "forward_return"]
        rows.append(
            {
                "date": date,
                "top": top.mean(),
                "bottom": bottom.mean(),
                "spread": top.mean() - bottom.mean(),
                "top_n": len(top),
                "bottom_n": len(bottom),
                "top_known": top.notna().sum(),
                "bottom_known": bottom.notna().sum(),
                "strict_spread": (top.mean() - bottom.mean())
                if top.notna().all() and bottom.notna().all()
                else np.nan,
            }
        )
    cohorts = pd.DataFrame(rows).sort_values("date")

    def sharpe(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.mean() / x.std(ddof=1) * np.sqrt(PPY)) if len(x) > 1 and x.std(ddof=1) > 0 else np.nan

    wealth = (1 + cohorts.spread.fillna(0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return (
        {
            "cohorts": int(len(cohorts)),
            "tail_n_per_side": N,
            "top_ann_arithmetic": float(cohorts.top.mean() * PPY),
            "bottom_ann_arithmetic": float(cohorts.bottom.mean() * PPY),
            "spread_ann_arithmetic": float(cohorts.spread.mean() * PPY),
            "spread_vol_ann": float(cohorts.spread.std(ddof=1) * np.sqrt(PPY)),
            "spread_sharpe": sharpe(cohorts.spread),
            "strict_spread_sharpe": sharpe(cohorts.strict_spread),
            "spread_cagr": float(wealth.iloc[-1] ** (PPY / len(cohorts)) - 1),
            "spread_max_drawdown": float(drawdown.min()),
            "positive_spread_fraction": float(cohorts.spread.gt(0).mean()),
            "unresolved_extreme_cohorts": int(cohorts.strict_spread.isna().sum()),
            "known_top_fraction": float(cohorts.top_known.sum() / cohorts.top_n.sum()),
            "known_bottom_fraction": float(cohorts.bottom_known.sum() / cohorts.bottom_n.sum()),
            "selected_flagged_rows": int(assign.get("return_outlier", pd.Series(dtype=bool)).fillna(False).sum()),
        },
        cohorts,
    )


def turnover(assign: pd.DataFrame, leg: str) -> dict:
    sets = {d: set(g.loc[g.leg.eq(leg), "security"]) for d, g in assign.groupby("date")}
    dates = sorted(sets)
    replaced = [len(sets[a] ^ sets[b]) / (2 * N) for a, b in zip(dates, dates[1:])]
    retained = [1 - x for x in replaced]
    return {
        "mean_replaced_fraction": float(np.mean(replaced)) if replaced else np.nan,
        "median_replaced_fraction": float(np.median(replaced)) if replaced else np.nan,
        "mean_retained_fraction": float(np.mean(retained)) if retained else np.nan,
        "median_retained_fraction": float(np.median(retained)) if retained else np.nan,
    }


def attach_outlier_flags(assign: pd.DataFrame) -> pd.DataFrame:
    if assign.empty:
        return assign.assign(return_outlier=False)
    path = OUT / "_selected_keys.parquet"
    assign[["security", "date"]].drop_duplicates().to_parquet(path, index=False)
    con = duckdb.connect()
    query = f"""
    SELECT k.security, k.date,
           COALESCE(e.return_outlier, FALSE) AS return_outlier
    FROM read_parquet('{path}') AS k
    LEFT JOIN read_parquet('{ELIGIBILITY_PATH}') AS e USING (security, date)
    """
    flags = con.execute(query).fetchdf()
    con.close()
    path.unlink(missing_ok=True)
    return assign.merge(flags, on=["security", "date"], how="left", validate="many_to_one").assign(
        return_outlier=lambda x: x.return_outlier.fillna(False).astype(bool)
    )


def diagnostic_flag_excluded(assign: pd.DataFrame) -> dict:
    x = assign.copy()
    x.loc[x.return_outlier, "forward_return"] = np.nan
    metrics, _ = portfolio_metrics(x)
    metrics["interpretation"] = "Observed-outcome sensitivity only; not a causal selection rule."
    return metrics


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    enriched, quality = add_quality_features(scores)
    quality.to_parquet(OUT / "quality_features.parquet", index=False)

    summary: dict = {
        "configuration": {
            "formation_start": "2020-01-01",
            "formation_end": "2026-06-30",
            "tail_n_per_side": N,
            "rank_buffer_per_side": BUFFER,
            "price_min": PRICE_MIN,
            "liquidity_filter": None,
            "formation_stride_sessions": 5,
            "return_outlier_used_for_selection": False,
            "returns": "raw forward outcomes; no costs",
        },
        "models": {},
        "quality_counts": {},
        "turnover": {},
        "diagnostic_flag_excluded": {},
    }

    all_selected = []
    for name, frame in enriched.items():
        q = frame.groupby("date", sort=True).agg(
            candidates=("security", "size"),
            quality_candidates=("quality_eligible", "sum"),
        ).reset_index()
        q["model"] = name
        summary["quality_counts"][name] = {
            "dates": int(len(q)),
            "min_quality_candidates": int(q.quality_candidates.min()),
            "p05_quality_candidates": float(q.quality_candidates.quantile(0.05)),
            "median_quality_candidates": float(q.quality_candidates.median()),
            "max_quality_candidates": int(q.quality_candidates.max()),
        }
        q.to_csv(OUT / f"{name}_quality_counts.csv", index=False)

        for variant, selector in {
            "quality_top_bottom20": lambda x: select_ranked(x, N),
            "quality_top_bottom20_buffer40": lambda x: select_buffered(x, N, BUFFER),
        }.items():
            a = selector(frame)
            a["model"] = name
            a["variant"] = variant
            a = attach_outlier_flags(a)
            a.to_parquet(OUT / f"{name}_{variant}_assignments.parquet", index=False)
            metrics, cohorts = portfolio_metrics(a)
            cohorts.to_csv(OUT / f"{name}_{variant}_cohorts.csv", index=False)
            summary["models"][f"{name}:{variant}"] = metrics
            summary["diagnostic_flag_excluded"][f"{name}:{variant}"] = diagnostic_flag_excluded(a)
            summary["turnover"][f"{name}:{variant}"] = {
                leg: turnover(a, leg) for leg in ["top", "bottom"]
            }
            all_selected.append(a)

    selected = pd.concat(all_selected, ignore_index=True)
    # build_characteristics computes these fields from the frozen price panel;
    # remove the quality-join copies first so pandas does not create close_x /
    # close_y columns in the characteristics table.
    selected_for_chars = selected.drop(columns=["close"], errors="ignore")
    selected_chars = build_characteristics(selected_for_chars)
    selected_chars.to_parquet(OUT / "selected_characteristics.parquet", index=False)
    feature_cols = ["close", "log_dollar_volume_20", "ret_20", "ret_60", "mom_12_1", "vol_20_ann", "ma200_gap"]
    char_rows = []
    for (model, variant, leg), g in selected_chars.groupby(["model", "variant", "leg"], observed=True):
        row = {"model": model, "variant": variant, "leg": leg, "rows": len(g)}
        for col in feature_cols:
            x = g[col].dropna()
            row[f"{col}_median"] = x.median() if len(x) else np.nan
            row[f"{col}_p25"] = x.quantile(0.25) if len(x) else np.nan
            row[f"{col}_p75"] = x.quantile(0.75) if len(x) else np.nan
        char_rows.append(row)
    pd.DataFrame(char_rows).to_csv(OUT / "characteristics_summary.csv", index=False)
    (OUT / "config.json").write_text(json.dumps(summary["configuration"], indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
