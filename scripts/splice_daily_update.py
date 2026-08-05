"""
=============================================================================
SCRIPT NAME: splice_daily_update.py
=============================================================================

INPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_ohlcv_database.parquet
      Bloomberg history leg (immutable; NEVER written by this script).
      Columns: Ticker, Date, Open, High, Low, Close, Volume, AdjClose
      where AdjClose = TOT_RETURN_INDEX_GROSS_DVDS.
- yfinance (network)   daily OHLCV + Adj Close for the forward leg.

OUTPUT FILES:
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_ohlcv_spliced.parquet
      Bloomberg history + yfinance forward extension, one panel, same schema.
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_ohlcv_spliced.csv
      Same, CSV (what pattern.data.loader.load_data reads).  Written only with --csv.
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_tradeable_universe.csv
      Ticker, yf_ticker, last_date, stale_days, tradeable.  THE STALENESS GUARD --
      daily scoring must consume this and drop tradeable=False names.
- /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/runs/splice/<TIMESTAMP>/
      splice_report.xlsx   per-ticker validation: overlap corr, rows added, verdict
      splice_summary.json  headline counts for the run
      splice.log           full log

DESCRIPTION:
Extends the Bloomberg OHLCV panel forward with yfinance so the CNN pipeline can
be refreshed daily without a Bloomberg pull.  Bloomberg remains the history of
record; yfinance supplies only dates AFTER the Bloomberg panel ends.

THE ONE THING THAT MATTERS -- the two AdjClose columns are NOT on the same
scale.  Bloomberg's total-return index accumulates dividends FORWARD from 1996;
yfinance's Adj Close is back-adjusted from today.  Measured level ratios on
2026-04-17: AAPL 1.19, MSFT 1.64, JNJ 2.10.  Concatenating them would inject a
fake 20-110% one-day return at the seam.  So AdjClose is CHAINED, not appended:

    AdjClose[t] = AdjClose_bbg[splice] * PROD(1 + yf_adj_return)   for t > splice

Raw Open/High/Low/Close/Volume ARE appended directly -- those were verified to
match Bloomberg exactly (return corr 1.000000 on 198/198 sampled tickers).

Safety rules enforced here:
  * The Bloomberg input is opened read-only and never rewritten.
  * Only tickers whose Bloomberg history reaches the panel's last date (within
    --max-gap-days) are extended.  A name that went dark in 2005 is NOT
    resurrected -- its ticker may since have been reassigned to another company.
  * Every extended ticker is validated on an overlap window before being
    accepted; failures are quarantined and reported, never silently spliced.
  * Output is written atomically (temp file then rename).
  * A ticker that fails to extend keeps its Bloomberg history and goes STALE.
    Stale prices silently poison dv_60d / vol_60d / mom_12_1 and can put a name
    in the wrong tertile, so every run writes a tradeable-universe manifest and
    the scoring step must drop names flagged tradeable=False.

Known free-source holes (measured 2026-08-04, 32 of 1,371 = 2.3%):
  * Hard 404 on Yahoo despite being live, liquid companies: BK, CTRA, VSCO.
  * Fragmentary Yahoo history (1-29 rows over 7 months): MASI, SATS, IAC, NSA,
    JHG, BLD, LC, GOCO and similar.
  * Genuine corporate actions correctly rejected by the overlap test: APTV
    (corr 0.67) and BDX (corr 0.99) both spun off businesses, so the two legs
    legitimately diverge.  These need a Bloomberg refresh, not a threshold tweak.

USAGE:
  # dry run -- validate and report, write nothing
  python scripts/splice_daily_update.py --dry-run

  # normal daily refresh
  python scripts/splice_daily_update.py

  # also emit the CSV the loader reads
  python scripts/splice_daily_update.py --csv

NOTES:
- Index membership is NOT refreshed here; this extends the names already in the
  panel.  Russell reconstitutes annually (effective end of June), so refresh the
  membership from Bloomberg once a year and re-run this daily in between.
- Requires: pandas, pyarrow, yfinance, openpyxl.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BBG = ROOT / "data" / "r1000_ohlcv_database.parquet"
DEFAULT_OUT = ROOT / "data" / "r1000_ohlcv_spliced.parquet"

SCHEMA = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume", "AdjClose"]

log = logging.getLogger("splice")


# ── ticker mapping ───────────────────────────────────────────────────────────

def to_yf(bbg_ticker: str) -> str:
    """'BRK/B US Equity' -> 'BRK-B';  'AAPL US Equity' -> 'AAPL'."""
    t = re.sub(r"\s+US\s+Equity$", "", bbg_ticker).strip()
    return t.replace("/", "-").replace(" ", "-")


# ── yfinance fetch ───────────────────────────────────────────────────────────

def fetch_yf(tickers: list[str], start: str, end: str,
             batch_size: int = 200) -> dict[str, pd.DataFrame]:
    """Return {yf_ticker: DataFrame[Date, Open, High, Low, Close, Volume, AdjClose]}."""
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i:i + batch_size]
        log.info(f"  yfinance batch {i // batch_size + 1}"
                 f"/{(len(tickers) - 1) // batch_size + 1} ({len(chunk)} tickers)")
        raw = yf.download(chunk, start=start, end=end, interval="1d",
                          progress=False, auto_adjust=False, threads=False,
                          group_by="ticker")
        if raw is None or len(raw) == 0:
            continue
        for t in chunk:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = sub.reset_index()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                need = {"Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"}
                if not need.issubset(set(df.columns)):
                    continue
                df = df[list(need)].rename(columns={"Adj Close": "AdjClose"})
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                df = df.dropna(subset=["Close", "AdjClose"]).sort_values("Date")
                if len(df):
                    out[t] = df.reset_index(drop=True)
            except Exception as exc:            # noqa: BLE001 - per-ticker isolation
                log.debug(f"    {t}: {exc}")
    return out


# ── the splice ───────────────────────────────────────────────────────────────

def splice_ticker(bbg: pd.DataFrame, yf_df: pd.DataFrame, splice_date: pd.Timestamp,
                  min_corr: float, min_overlap: int) -> tuple[pd.DataFrame | None, dict]:
    """
    Validate on the overlap, then chain AdjClose forward.

    Returns (new_rows_or_None, diagnostics).  new_rows covers dates > splice_date
    only; the Bloomberg leg is returned untouched by the caller.
    """
    diag: dict = {"overlap_n": 0, "corr": np.nan, "med_abs_diff_bp": np.nan,
                  "rows_added": 0, "verdict": "", "last_new_date": ""}

    # --- overlap validation (dates present in BOTH legs) ---
    ov = bbg.merge(yf_df, on="Date", suffixes=("_bb", "_yf"))
    diag["overlap_n"] = len(ov)
    if len(ov) < min_overlap:
        diag["verdict"] = f"REJECT: overlap {len(ov)} < {min_overlap}"
        return None, diag

    r_bb = ov["AdjClose_bb"].pct_change(fill_method=None)
    r_yf = ov["AdjClose_yf"].pct_change(fill_method=None)
    both = pd.concat([r_bb, r_yf], axis=1).dropna()
    if len(both) < min_overlap - 1:
        diag["verdict"] = f"REJECT: usable overlap returns {len(both)}"
        return None, diag

    corr = both.iloc[:, 0].corr(both.iloc[:, 1])
    diff = (both.iloc[:, 0] - both.iloc[:, 1]).abs()
    diag["corr"] = float(corr) if pd.notna(corr) else np.nan
    diag["med_abs_diff_bp"] = float(diff.median() * 1e4)

    if pd.isna(corr) or corr < min_corr:
        diag["verdict"] = f"REJECT: overlap corr {corr:.4f} < {min_corr}"
        return None, diag

    # --- forward leg ---
    fwd = yf_df[yf_df["Date"] > splice_date].copy()
    if len(fwd) == 0:
        diag["verdict"] = "OK: already current, nothing to add"
        return None, diag

    # Anchor: the last Bloomberg AdjClose, and the yf AdjClose on the same date.
    anchor_bb = bbg.loc[bbg["Date"] == splice_date, "AdjClose"]
    anchor_yf = yf_df.loc[yf_df["Date"] == splice_date, "AdjClose"]
    if anchor_bb.empty or anchor_yf.empty:
        diag["verdict"] = "REJECT: no common anchor date"
        return None, diag
    a_bb, a_yf = float(anchor_bb.iloc[0]), float(anchor_yf.iloc[0])
    if not np.isfinite(a_bb) or not np.isfinite(a_yf) or a_yf <= 0 or a_bb <= 0:
        diag["verdict"] = "REJECT: non-finite anchor"
        return None, diag

    # CHAIN, do not concatenate: scale the yf adjusted series onto the Bloomberg
    # total-return level using the ratio at the anchor date.  Equivalent to
    # compounding yf returns off the Bloomberg level, and robust to gaps.
    fwd["AdjClose"] = fwd["AdjClose"] * (a_bb / a_yf)

    fwd["Ticker"] = bbg["Ticker"].iloc[0]
    fwd = fwd[SCHEMA]
    diag["rows_added"] = len(fwd)
    diag["last_new_date"] = str(fwd["Date"].max().date())
    diag["verdict"] = "OK"
    return fwd, diag


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Splice Bloomberg history + yfinance forward.")
    ap.add_argument("--bbg", type=Path, default=DEFAULT_BBG,
                    help="Bloomberg history parquet (read-only)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="spliced output parquet")
    ap.add_argument("--csv", action="store_true",
                    help="also write the .csv the loader reads")
    ap.add_argument("--end", default=None,
                    help="end date YYYY-MM-DD (default: today)")
    ap.add_argument("--overlap-days", type=int, default=90,
                    help="calendar days of overlap pulled for validation")
    ap.add_argument("--max-gap-days", type=int, default=7,
                    help="a ticker is 'current' if its last Bloomberg date is "
                         "within this many days of the panel max; others are "
                         "NOT extended (ticker-reuse guard)")
    ap.add_argument("--min-corr", type=float, default=0.99,
                    help="minimum overlap return correlation to accept a ticker")
    ap.add_argument("--min-overlap", type=int, default=15,
                    help="minimum overlapping observations to validate")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--max-staleness-days", type=int, default=5,
                    help="a ticker is tradeable only if its last row in the "
                         "SPLICED panel is within this many calendar days of "
                         "the panel max; stale names poison the tertile buckets")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only process the first N current tickers")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and report; write no panel")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / "splice" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(run_dir / "splice.log")],
    )

    log.info(f"run dir: {run_dir}")
    log.info(f"Bloomberg history (read-only): {args.bbg}")

    bbg_all = pd.read_parquet(args.bbg)
    bbg_all["Date"] = pd.to_datetime(bbg_all["Date"])
    missing = set(SCHEMA) - set(bbg_all.columns)
    if missing:
        raise SystemExit(f"Bloomberg panel missing columns: {missing}")
    bbg_all = bbg_all[SCHEMA]

    panel_max = bbg_all["Date"].max()
    last_by_ticker = bbg_all.groupby("Ticker")["Date"].max()
    cutoff = panel_max - pd.Timedelta(days=args.max_gap_days)
    current = sorted(last_by_ticker[last_by_ticker >= cutoff].index)
    dormant = len(last_by_ticker) - len(current)

    log.info(f"panel: {len(bbg_all):,} rows  {bbg_all['Ticker'].nunique():,} tickers  "
             f"-> {panel_max.date()}")
    log.info(f"current tickers (extendable): {len(current):,}")
    log.info(f"dormant/delisted (NOT extended, ticker-reuse guard): {dormant:,}")

    if args.limit:
        current = current[:args.limit]
        log.info(f"--limit: processing first {len(current)} tickers")

    end = args.end or (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (panel_max - pd.Timedelta(days=args.overlap_days)).strftime("%Y-%m-%d")
    if pd.Timestamp(end) <= panel_max:
        log.warning(f"end {end} <= panel max {panel_max.date()}; nothing to add.")

    ymap = {t: to_yf(t) for t in current}
    log.info(f"fetching yfinance {start} -> {end} for {len(current):,} tickers ...")
    fetched = fetch_yf(sorted(set(ymap.values())), start, end, args.batch_size)
    log.info(f"  yfinance returned data for {len(fetched):,} tickers")

    # --- per-ticker splice ---
    bbg_recent = bbg_all[bbg_all["Date"] >= pd.Timestamp(start)]
    groups = {t: g for t, g in bbg_recent.groupby("Ticker", sort=False)}

    new_parts, diags = [], []
    for t in current:
        y = fetched.get(ymap[t])
        d = {"Ticker": t, "yf_ticker": ymap[t],
             "bbg_last": str(last_by_ticker[t].date())}
        if y is None or len(y) == 0:
            d.update({"overlap_n": 0, "corr": np.nan, "med_abs_diff_bp": np.nan,
                      "rows_added": 0, "last_new_date": "",
                      "verdict": "NO_YF_DATA (delisted/renamed?)"})
            diags.append(d)
            continue
        g = groups.get(t)
        if g is None or len(g) == 0:
            d.update({"overlap_n": 0, "corr": np.nan, "med_abs_diff_bp": np.nan,
                      "rows_added": 0, "last_new_date": "",
                      "verdict": "REJECT: no recent Bloomberg rows"})
            diags.append(d)
            continue
        rows, diag = splice_ticker(g.sort_values("Date"), y, last_by_ticker[t],
                                   args.min_corr, args.min_overlap)
        d.update(diag)
        diags.append(d)
        if rows is not None and len(rows):
            new_parts.append(rows)

    rep = pd.DataFrame(diags)
    accepted = rep[rep["verdict"] == "OK"]
    rejected = rep[rep["verdict"].str.startswith("REJECT")]
    nodata = rep[rep["verdict"].str.startswith("NO_YF_DATA")]
    current_ok = rep[rep["verdict"].str.startswith("OK: already current")]

    log.info("")
    log.info("=== VALIDATION ===")
    log.info(f"  extended        : {len(accepted):,}  ({accepted['rows_added'].sum():,} new rows)")
    log.info(f"  already current : {len(current_ok):,}")
    log.info(f"  no yfinance data: {len(nodata):,}")
    log.info(f"  rejected        : {len(rejected):,}")
    if len(accepted):
        log.info(f"  overlap corr    : median {accepted['corr'].median():.6f}  "
                 f"min {accepted['corr'].min():.6f}")
        log.info(f"  med |diff|      : {accepted['med_abs_diff_bp'].median():.4f} bp/day")
        log.info(f"  new data through: {accepted['last_new_date'].max()}")
    if len(rejected):
        log.warning("  rejected tickers (NOT spliced):")
        for _, r in rejected.head(25).iterrows():
            log.warning(f"    {r['Ticker']:<24s} {r['verdict']}")
    if len(nodata):
        sample = ", ".join(nodata["yf_ticker"].head(25).tolist())
        log.warning(f"  no-yf-data sample: {sample}")

    # --- write report always ---
    with pd.ExcelWriter(run_dir / "splice_report.xlsx", engine="openpyxl") as xw:
        rep.to_excel(xw, sheet_name="per_ticker", index=False)
        if len(rejected):
            rejected.to_excel(xw, sheet_name="rejected", index=False)
        if len(nodata):
            nodata.to_excel(xw, sheet_name="no_yf_data", index=False)
    log.info(f"wrote {run_dir / 'splice_report.xlsx'}")

    summary = {
        "run": stamp,
        "bbg_source": str(args.bbg),
        "bbg_panel_max": str(panel_max.date()),
        "bbg_rows": int(len(bbg_all)),
        "bbg_tickers": int(bbg_all["Ticker"].nunique()),
        "current_tickers": int(len(current)),
        "dormant_not_extended": int(dormant),
        "extended": int(len(accepted)),
        "already_current": int(len(current_ok)),
        "no_yf_data": int(len(nodata)),
        "rejected": int(len(rejected)),
        "rows_added": int(accepted["rows_added"].sum()) if len(accepted) else 0,
        "new_data_through": (accepted["last_new_date"].max() if len(accepted) else ""),
        "min_corr_threshold": args.min_corr,
        "overlap_corr_median": (float(accepted["corr"].median()) if len(accepted) else None),
        "overlap_corr_min": (float(accepted["corr"].min()) if len(accepted) else None),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        summary["output"] = "DRY RUN - no panel written"
        (run_dir / "splice_summary.json").write_text(json.dumps(summary, indent=2))
        log.info("DRY RUN - no panel written")
        log.info(f"wrote {run_dir / 'splice_summary.json'}")
        return

    if not new_parts:
        summary["output"] = "no new rows - panel unchanged"
        (run_dir / "splice_summary.json").write_text(json.dumps(summary, indent=2))
        log.info("No new rows; panel not rewritten.")
        return

    # --- combine + atomic write (Bloomberg input untouched) ---
    log.info("")
    log.info("combining ...")
    combined = pd.concat([bbg_all] + new_parts, ignore_index=True)
    for c in ["Open", "High", "Low", "Close", "Volume", "AdjClose"]:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")
    combined = (combined
                .drop_duplicates(subset=["Ticker", "Date"], keep="first")
                .sort_values(["Ticker", "Date"])
                .reset_index(drop=True))
    log.info(f"  spliced panel: {len(combined):,} rows  "
             f"{combined['Ticker'].nunique():,} tickers  "
             f"{combined['Date'].min().date()} -> {combined['Date'].max().date()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + f".tmp{os.getpid()}")
    combined.to_parquet(tmp, index=False)
    os.replace(tmp, args.out)
    log.info(f"wrote {args.out}")
    summary["output"] = str(args.out)
    summary["spliced_rows"] = int(len(combined))
    summary["spliced_max_date"] = str(combined["Date"].max().date())

    if args.csv:
        csv_path = args.out.with_suffix(".csv")
        tmpc = csv_path.with_suffix(f".csv.tmp{os.getpid()}")
        combined.to_csv(tmpc, index=False)
        os.replace(tmpc, csv_path)
        log.info(f"wrote {csv_path}")
        summary["output_csv"] = str(csv_path)

    # --- staleness guard: which names may actually be traded today ---
    panel_end = combined["Date"].max()
    last_now = combined.groupby("Ticker")["Date"].max()
    uni = pd.DataFrame({
        "Ticker": last_now.index,
        "last_date": last_now.dt.date.astype(str).values,
        "stale_days": (panel_end - last_now).dt.days.values,
    })
    uni = uni[uni["Ticker"].isin(current)].copy()
    uni["yf_ticker"] = uni["Ticker"].map(ymap)
    uni["tradeable"] = uni["stale_days"] <= args.max_staleness_days
    uni = uni[["Ticker", "yf_ticker", "last_date", "stale_days", "tradeable"]]
    uni = uni.sort_values(["tradeable", "Ticker"], ascending=[False, True])

    uni_path = args.out.parent / "r1000_tradeable_universe.csv"
    tmpu = uni_path.with_suffix(f".csv.tmp{os.getpid()}")
    uni.to_csv(tmpu, index=False)
    os.replace(tmpu, uni_path)

    n_ok = int(uni["tradeable"].sum())
    n_stale = int((~uni["tradeable"]).sum())
    log.info("")
    log.info("=== STALENESS GUARD ===")
    log.info(f"  panel end     : {panel_end.date()}")
    log.info(f"  tradeable     : {n_ok:,}")
    log.info(f"  STALE-EXCLUDE : {n_stale:,}  (>{args.max_staleness_days}d behind)")
    if n_stale:
        s = uni[~uni["tradeable"]].nlargest(min(n_stale, 15), "stale_days")
        for _, r in s.iterrows():
            log.warning(f"    {r['Ticker']:<24s} last {r['last_date']}  "
                        f"{r['stale_days']}d stale")
    log.info(f"wrote {uni_path}")
    summary["tradeable_universe"] = str(uni_path)
    summary["tradeable"] = n_ok
    summary["stale_excluded"] = n_stale

    (run_dir / "splice_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"wrote {run_dir / 'splice_summary.json'}")


if __name__ == "__main__":
    main()
