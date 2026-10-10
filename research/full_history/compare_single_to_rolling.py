"""Compare the single full-history fit with the existing rolling5 fits.

The comparison is restricted to identical 2020--2026 formation dates and
uses the same top/bottom decile construction for both models.  It also saves
score/decile agreement and trailing, formation-date-only characteristics for
each selected leg.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SINGLE = ROOT / "runs/full_history_single_2019cutoff_seed101"
ROLLING = ROOT / "runs/full_rolling_clean_2003_2026_seed101"
OUT = SINGLE / "comparison_to_rolling5"


def deciles(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for date, g in df.groupby("date", sort=True):
        g = g.sort_values(["score", "security"]).copy()
        g["decile"] = (np.arange(len(g)) * 10 // len(g) + 1).astype(np.int8)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def cohort_metrics(assign: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    for date, g in assign.groupby("date", sort=True):
        top = g.loc[g.decile == 10, "forward_return"]
        bot = g.loc[g.decile == 1, "forward_return"]
        rows.append({
            "date": date,
            "top": top.mean(), "bottom": bot.mean(),
            "spread": top.mean() - bot.mean(),
            "top_n": len(top), "bottom_n": len(bot),
            "top_known": top.notna().sum(), "bottom_known": bot.notna().sum(),
            "strict_spread": (top.mean() - bot.mean()) if top.notna().all() and bot.notna().all() else np.nan,
        })
    c = pd.DataFrame(rows).sort_values("date")
    ppy = 252 / 5
    def sharpe(x):
        x = x.dropna()
        return float(x.mean() / x.std(ddof=1) * np.sqrt(ppy)) if len(x) > 1 and x.std(ddof=1) > 0 else np.nan
    return {
        "cohorts": int(len(c)),
        "top": float(c.top.mean()), "bottom": float(c.bottom.mean()), "spread": float(c.spread.mean()),
        "top_ann_arithmetic": float(c.top.mean() * ppy), "bottom_ann_arithmetic": float(c.bottom.mean() * ppy),
        "spread_ann_arithmetic": float(c.spread.mean() * ppy), "spread_sharpe": sharpe(c.spread),
        "strict_spread_sharpe": sharpe(c.strict_spread),
        "positive_spread_fraction": float(c.spread.gt(0).mean()),
        "unresolved_extreme_cohorts": int(c.strict_spread.isna().sum()),
        "known_top_fraction": float(c.top_known.sum() / c.top_n.sum()),
        "known_bottom_fraction": float(c.bottom_known.sum() / c.bottom_n.sum()),
    }, c


def turnover(assign: pd.DataFrame, leg: int) -> pd.DataFrame:
    sets = {d: set(g.loc[g.decile == leg, "security"]) for d, g in assign.groupby("date")}
    dates = sorted(sets)
    rows = []
    for a, b in zip(dates, dates[1:]):
        old, new = sets[a], sets[b]
        rows.append({"date": b, "replaced_fraction": len(old ^ new) / (len(old) + len(new))})
    return pd.DataFrame(rows)


def load_scores() -> dict[str, pd.DataFrame]:
    single_path = SINGLE / "scores_rolling_schedule.parquet"
    if not single_path.exists():
        single_path = SINGLE / "scores.parquet"
    single = pd.read_parquet(single_path)
    single["date"] = pd.to_datetime(single.date)
    frames = []
    for year in range(2020, 2027):
        p = ROLLING / f"{year}_rolling5_seed101/scores.parquet"
        if p.exists():
            x = pd.read_parquet(p)
            x["date"] = pd.to_datetime(x.date)
            frames.append(x)
    if not frames:
        raise FileNotFoundError("rolling scores are missing")
    rolling = pd.concat(frames, ignore_index=True)
    return {"single_full_history": single, "rolling5": rolling}


def build_characteristics(selected: pd.DataFrame) -> pd.DataFrame:
    """Return formation-date trailing characteristics for selected names."""
    dates = pd.to_datetime(selected.date)
    securities = set(selected.security)
    prices_path = ROOT / "data/frozen/pattern_daily_v1_20260918/prices.parquet"
    px = pd.read_parquet(prices_path, columns=["security", "date", "close", "volume", "total_return_index_gross_dividends"])
    px = px[px.security.isin(securities)].sort_values(["security", "date"]).copy()
    px["tri_ret_1"] = px.groupby("security").total_return_index_gross_dividends.pct_change()
    g = px.groupby("security", group_keys=False)
    px["ret_20"] = px.groupby("security").total_return_index_gross_dividends.pct_change(20)
    px["ret_60"] = px.groupby("security").total_return_index_gross_dividends.pct_change(60)
    px["mom_12_1"] = px.groupby("security").total_return_index_gross_dividends.shift(21) / px.groupby("security").total_return_index_gross_dividends.shift(252) - 1
    px["vol_20_ann"] = px.groupby("security").tri_ret_1.transform(lambda s: s.rolling(20, min_periods=15).std() * np.sqrt(252))
    px["dollar_volume_20"] = (px.close * px.volume).groupby(px.security).transform(lambda s: s.rolling(20, min_periods=5).mean())
    px["ma200_gap"] = px.close / px.groupby("security").close.transform(lambda s: s.rolling(200, min_periods=100).mean()) - 1
    cols = ["security", "date", "close", "volume", "ret_20", "ret_60", "mom_12_1", "vol_20_ann", "dollar_volume_20", "ma200_gap"]
    selected = selected.merge(px[cols], on=["security", "date"], how="left", validate="many_to_one")
    selected["log_dollar_volume_20"] = np.log(selected.dollar_volume_20.where(selected.dollar_volume_20 > 0))
    return selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    assignments = {}
    metrics = {}
    cohorts = {}
    for name, frame in scores.items():
        frame = frame[(frame.date >= "2020-01-01") & (frame.date <= "2026-06-30")].copy()
        frame = frame.drop_duplicates(["security", "date"], keep="last")
        a = deciles(frame)
        assignments[name] = a
        metrics[name], cohorts[name] = cohort_metrics(a)
        cohorts[name].to_csv(OUT / f"{name}_cohorts.csv", index=False)
        a.to_parquet(OUT / f"{name}_assignments.parquet", index=False)

    # Persist the exact assignments and cohort returns used below.
    for name, a in assignments.items():
        a.to_parquet(OUT / f"{name}_assignments.parquet", index=False)
        cohorts[name].to_csv(OUT / f"{name}_cohorts.csv", index=False)

    left = assignments["single_full_history"].rename(columns={"score": "score_single", "decile": "decile_single"})
    right = assignments["rolling5"].rename(columns={"score": "score_rolling", "decile": "decile_rolling", "forward_return": "forward_return_rolling"})
    merged = left.merge(right[["security", "date", "score_rolling", "decile_rolling", "forward_return_rolling"]], on=["security", "date"], how="inner")
    merged.to_parquet(OUT / "score_and_decile_comparison.parquet", index=False)
    rows = []
    for date, g in merged.groupby("date", sort=True):
        rows.append({
            "date": date,
            "n": len(g),
            "score_pearson": g.score_single.corr(g.score_rolling),
            "score_spearman": g.score_single.corr(g.score_rolling, method="spearman"),
            "decile_disagreement": (g.decile_single != g.decile_rolling).mean(),
            "top_overlap": len(set(g.loc[g.decile_single == 10, "security"]) & set(g.loc[g.decile_rolling == 10, "security"])) / len(set(g.loc[g.decile_single == 10, "security"])),
            "bottom_overlap": len(set(g.loc[g.decile_single == 1, "security"]) & set(g.loc[g.decile_rolling == 1, "security"])) / len(set(g.loc[g.decile_single == 1, "security"])),
        })
    agreement = pd.DataFrame(rows)
    agreement.to_csv(OUT / "agreement_by_date.csv", index=False)

    turnover_rows = []
    for name, a in assignments.items():
        for leg, label in [(10, "top"), (1, "bottom")]:
            t = turnover(a, leg)
            turnover_rows.append({"model": name, "leg": label, "mean_replaced_fraction": t.replaced_fraction.mean(), "median_replaced_fraction": t.replaced_fraction.median()})
    pd.DataFrame(turnover_rows).to_csv(OUT / "turnover_summary.csv", index=False)

    selected = []
    for name, a in assignments.items():
        selected.append(a[a.decile.isin([1, 10])].assign(model=name, leg=np.where(a.loc[a.decile.isin([1, 10]), "decile"].eq(10), "top", "bottom")))
    selected = pd.concat(selected, ignore_index=True)
    selected = build_characteristics(selected)
    feature_cols = ["close", "log_dollar_volume_20", "ret_20", "ret_60", "mom_12_1", "vol_20_ann", "ma200_gap"]
    # Use robust summaries: raw close contains a small number of known stale or
    # split-related extremes, so arithmetic means are not a useful portfolio
    # characteristic.  Preserve median and interquartile ranges instead.
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

    summary = {
        "models": metrics,
        "agreement": {
            "dates": int(len(agreement)),
            "mean_score_pearson": float(agreement.score_pearson.mean()),
            "mean_score_spearman": float(agreement.score_spearman.mean()),
            "mean_decile_disagreement": float(agreement.decile_disagreement.mean()),
            "mean_top_overlap": float(agreement.top_overlap.mean()),
            "mean_bottom_overlap": float(agreement.bottom_overlap.mean()),
        },
        "characteristic_features": feature_cols,
    }
    (OUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
