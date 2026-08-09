"""
=============================================================================
SCRIPT NAME: evaluate_signal.py
=============================================================================

INPUT FILES:
- --signal PATH    parquet with columns (ticker, end_date, <score column>).
                   Any ranking signal: the CNN's p_up_mean, a reversal signal,
                   a GBDT score -- the harness does not care which.
- --ohlcv PATH     /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/data/r1000_ohlcv_spliced.parquet
                   (falls back to data/r1000_ohlcv_database.parquet)

OUTPUT FILES:
- {out_dir}/signal_card.xlsx    deciles, LS variants, sub-periods, one sheet each
- {out_dir}/signal_card.json    machine-readable summary for cross-run comparison

DESCRIPTION:
THE ONE MEASURING STICK. Every experiment in the kill-or-confirm plan is scored
here so numbers from different stages are actually comparable. Prior to this,
each backtest script computed returns its own way, which is how a 7x reporting
error survived (README addenda J and K).

Two rules this harness enforces, both learned the hard way:

1. PORTFOLIO ARITHMETIC IS THE HEADLINE. An equal-weighted book earns the mean
   of SIMPLE returns. The repo's `forward_return` column is a 20-day LOG return,
   and compounding a difference of mean-LOG returns overstates a long/short
   spread by roughly the short leg's sigma^2/2 -- measured at +7.14pp/yr on the
   trash tier. The log number is still reported, in a column explicitly labelled
   as a cross-reference to the old published figures, never as the headline.

2. A PRICE FLOOR IS MANDATORY FOR THE HEADLINE. Sub-penny bankruptcy shells sort
   into the small/high-vol/loser cell by construction (tiny dollar volume, huge
   vol, terrible momentum) and a one-tick move there is a +4,800% return. Without
   a floor the simple-return decile book is not even computable -- some months
   fall below -100% because shorting a name that 10x's loses 900%.

Everything is GROSS of costs, per house convention.

USAGE:
  # the CNN reference card
  python3 scripts/evaluate_signal.py \
      --signal runs/expanding/20260419_174908_cdef6809/predictions_monthly.parquet \
      --score-col p_up_mean --label "CNN p_up" \
      --out-dir runs/experiments/stage0_reference_p_up

  # any challenger signal, same stick
  python3 scripts/evaluate_signal.py --signal <file> --score-col rev_1m --label STR ...

NOTES:
- `--higher-is-better/--lower-is-better` flips the sort, so a reversal signal
  (low past return -> expected winner) is scored on equal footing without
  hand-editing the input.
- Deciles need a reasonably wide cross-section; months with fewer than
  --min-names usable rows are skipped and counted in the JSON.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OHLCV = ROOT / "data" / "r1000_ohlcv_spliced.parquet"
FALLBACK_OHLCV = ROOT / "data" / "r1000_ohlcv_database.parquet"

MIN_PRICE = 5.0
PERIODS_PER_YEAR = 12          # monthly-cadence formation dates

# Cross-sectional cells, as tertiles of (dollar volume, realized vol, momentum).
# The price floor is applied BEFORE bucketing so the tertile boundaries match
# backtest_liquid_grid.py --min-price; filtering after would leave the cut
# points set by names that are then discarded.
CELLS = {
    "trash":   {"dv_b": "Low",  "vol_b": "High"},          # + mom below
    "megacap": {"dv_b": "High", "vol_b": "High"},
}
CELL_MOM = {"trash": "Low"}


# ── stats ────────────────────────────────────────────────────────────────────

def newey_west_t(x: np.ndarray, lag: int = 0) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    xc = x - mu
    s = (xc ** 2).mean()
    for k in range(1, lag + 1):
        if k >= n:
            break
        s += 2 * (1 - k / (lag + 1)) * (xc[k:] * xc[:-k]).mean()
    se = np.sqrt(max(s, 0) / n)
    return float(mu / se) if se > 0 else float("nan")


def series_stats(r: pd.Series) -> dict:
    """Annualized stats for a monthly simple-return series."""
    r = r.dropna()
    n = len(r)
    if n < 6:
        return {"months": n, "cagr_pct": None, "mean_pct": None, "vol_pct": None,
                "sharpe": None, "nw_t": None, "worst_pct": None, "computable": False}
    growth = (1 + r)
    computable = bool((growth > 0).all())
    cagr = ((growth.prod() ** (PERIODS_PER_YEAR / n) - 1) * 100) if computable else None
    sd = r.std()
    return {
        "months": n,
        "cagr_pct": None if cagr is None else round(float(cagr), 3),
        "mean_pct": round(float(r.mean() * 100), 4),
        "vol_pct": round(float(sd * np.sqrt(PERIODS_PER_YEAR) * 100), 3),
        "sharpe": (round(float((r.mean() * PERIODS_PER_YEAR) /
                               (sd * np.sqrt(PERIODS_PER_YEAR))), 3) if sd > 0 else None),
        "nw_t": round(newey_west_t(r.to_numpy()), 3),
        "worst_pct": round(float(r.min() * 100), 2),
        # False means a month <= -100% -- the book blew up, CAGR is undefined
        "computable": computable,
    }


# ── cross-sectional cells ────────────────────────────────────────────────────

def compute_cell_features(ohlcv_path: Path) -> pd.DataFrame:
    """dv_60d / vol_60d / mom_12_1 per (ticker, date) -- same definitions as
    scripts/backtest_liquid_grid.py, so cells are comparable across scripts."""
    o = pd.read_parquet(ohlcv_path,
                        columns=["Ticker", "Date", "Close", "Volume", "AdjClose"])
    o["Date"] = pd.to_datetime(o["Date"])
    o = o.sort_values(["Ticker", "Date"])
    g = o.groupby("Ticker")
    o["dv"] = o["Close"] * o["Volume"]
    o["dv_60d"] = g["dv"].transform(lambda s: s.rolling(60, min_periods=15).mean())
    o["logp"] = np.log(o["AdjClose"].replace(0, np.nan))
    o["mom_12_1"] = g["logp"].shift(21) - g["logp"].shift(252 + 21)
    o["ret"] = g["AdjClose"].pct_change(fill_method=None)
    o["vol_60d"] = g["ret"].transform(
        lambda s: s.rolling(60, min_periods=20).std() * np.sqrt(252))
    return (o[["Ticker", "Date", "dv_60d", "mom_12_1", "vol_60d"]]
            .rename(columns={"Ticker": "ticker", "Date": "end_date"}))


def _tertile(df: pd.DataFrame, col: str) -> pd.Series:
    def assign(g):
        if g[col].notna().sum() < 3:
            return pd.Series([pd.NA] * len(g), index=g.index)
        return pd.qcut(g[col].rank(method="first"), 3,
                       labels=["Low", "Mid", "High"], duplicates="drop")
    return df.groupby("end_date", group_keys=False).apply(assign)


def apply_cell(df: pd.DataFrame, cell: str, ohlcv_path: Path,
               min_price: float) -> pd.DataFrame:
    feats = compute_cell_features(ohlcv_path)
    df = df.merge(feats, on=["ticker", "end_date"], how="left")
    df = df.dropna(subset=["dv_60d", "vol_60d", "mom_12_1"])
    df = df[df["Close"] >= min_price]                   # floor BEFORE bucketing
    df["dv_b"] = _tertile(df, "dv_60d")
    df["vol_b"] = _tertile(df, "vol_60d")
    df["mom_b"] = _tertile(df, "mom_12_1")
    for k, v in CELLS[cell].items():
        df = df[df[k] == v]
    if cell in CELL_MOM:
        df = df[df["mom_b"] == CELL_MOM[cell]]
    return df


# ── portfolio construction ───────────────────────────────────────────────────

def build_books(df: pd.DataFrame, score: str, n_deciles: int,
                min_names: int) -> tuple[pd.DataFrame, dict]:
    """Per-formation-date decile and 50/50 books, on BOTH return conventions."""
    rows, skipped = [], 0
    for dt, g in df.groupby("end_date"):
        g = g.dropna(subset=[score, "r_simple", "forward_return"])
        if len(g) < min_names:
            skipped += 1
            continue
        n = len(g)
        q = g[score].rank(method="first", pct=True)
        top_d = g[q > 1 - 1 / n_deciles]
        bot_d = g[q <= 1 / n_deciles]
        half = g[score].rank(method="first") / n
        top_h, bot_h = g[half > 0.5], g[half <= 0.5]
        if min(len(top_d), len(bot_d), len(top_h), len(bot_h)) < 2:
            skipped += 1
            continue
        rows.append({
            "end_date": dt, "n": n,
            "ls_decile": top_d.r_simple.mean() - bot_d.r_simple.mean(),
            "ls_5050": top_h.r_simple.mean() - bot_h.r_simple.mean(),
            "top_decile": top_d.r_simple.mean(),
            "bot_decile": bot_d.r_simple.mean(),
            "ew": g.r_simple.mean(),
            # cross-reference only: the mean-LOG convention used pre-addendum-K
            "ls_decile_LOGCONV": top_d.forward_return.mean() - bot_d.forward_return.mean(),
            "ls_5050_LOGCONV": top_h.forward_return.mean() - bot_h.forward_return.mean(),
            "top_set": frozenset(top_d.ticker), "bot_set": frozenset(bot_d.ticker),
        })
    book = pd.DataFrame(rows).sort_values("end_date").reset_index(drop=True)
    return book, {"skipped_months": skipped}


def turnover(book: pd.DataFrame) -> float | None:
    if len(book) < 2:
        return None
    vals = []
    for i in range(1, len(book)):
        for c in ("top_set", "bot_set"):
            cur, prev = book.loc[i, c], book.loc[i - 1, c]
            if cur and prev:
                vals.append(len(cur ^ prev) / (len(cur) + len(prev)))
    return round(float(np.mean(vals)), 4) if vals else None


def decile_table(df: pd.DataFrame, score: str, n_deciles: int,
                 min_names: int) -> pd.DataFrame:
    """Mean simple return per decile -- the monotonicity check."""
    acc = {d: [] for d in range(1, n_deciles + 1)}
    for _, g in df.groupby("end_date"):
        g = g.dropna(subset=[score, "r_simple"])
        if len(g) < min_names:
            continue
        d = pd.qcut(g[score].rank(method="first"), n_deciles,
                    labels=range(1, n_deciles + 1))
        for dec, sub in g.groupby(d, observed=True):
            acc[int(dec)].append(sub.r_simple.mean())
    out = []
    for d, v in acc.items():
        s = pd.Series(v)
        st = series_stats(s)
        out.append({"decile": d, "months": st["months"],
                    "mean_pct": st["mean_pct"], "cagr_pct": st["cagr_pct"]})
    return pd.DataFrame(out)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Score any ranking signal on one stick.")
    ap.add_argument("--signal", type=Path, required=True)
    ap.add_argument("--score-col", default="p_up_mean")
    ap.add_argument("--label", default=None, help="display name (default: score col)")
    ap.add_argument("--ohlcv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-deciles", type=int, default=10)
    ap.add_argument("--min-names", type=int, default=50)
    ap.add_argument("--min-price", type=float, default=MIN_PRICE)
    ap.add_argument("--lower-is-better", action="store_true",
                    help="negate the score (e.g. reversal: low past return = buy)")
    ap.add_argument("--cell", choices=sorted(CELLS), default=None,
                    help="restrict to a cross-sectional cell before ranking "
                         "(trash = Low dv x High vol x Low mom; "
                         "megacap = High dv x High vol). Implies the price floor.")
    args = ap.parse_args()

    label = args.label or args.score_col
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ohlcv_path = args.ohlcv or (DEFAULT_OHLCV if DEFAULT_OHLCV.exists() else FALLBACK_OHLCV)

    print(f"Signal : {label}  ({args.signal.name}::{args.score_col})")
    print(f"OHLCV  : {ohlcv_path.name}")

    sig = pd.read_parquet(args.signal)
    if args.score_col not in sig.columns:
        raise SystemExit(f"--score-col '{args.score_col}' not in {list(sig.columns)}")
    sig["end_date"] = pd.to_datetime(sig["end_date"])

    if "forward_return" not in sig.columns:
        raise SystemExit(
            "signal file has no 'forward_return' column (20d LOG return). "
            "Join it from predictions_monthly.parquet before evaluating.")

    px = pd.read_parquet(ohlcv_path, columns=["Ticker", "Date", "Close"])
    px["Date"] = pd.to_datetime(px["Date"])
    df = sig.merge(px.rename(columns={"Ticker": "ticker", "Date": "end_date"}),
                   on=["ticker", "end_date"], how="left")

    df["score"] = -df[args.score_col] if args.lower_is_better else df[args.score_col]
    # forward_return is a 20-day LOG return -> simple return for portfolio math
    df["r_simple"] = np.expm1(df["forward_return"])

    if args.cell:
        before = len(df)
        df = apply_cell(df, args.cell, ohlcv_path, args.min_price)
        print(f"Cell   : {args.cell}  ({before:,} -> {len(df):,} rows)")
        # The cell already carries the floor, and a cell is far too thin for
        # deciles, so score the 50/50 book on one variant only.
        variants = {f"cell_{args.cell}": df}
        args.min_names = min(args.min_names, 20)
    else:
        variants = {
            "floored": df[df["Close"] >= args.min_price],
            "unfloored": df,
        }

    sheets, summary = {}, {"signal": label, "score_col": args.score_col,
                           "lower_is_better": bool(args.lower_is_better),
                           "min_price": args.min_price, "variants": {}}

    for vname, vdf in variants.items():
        book, meta = build_books(vdf, "score", args.n_deciles, args.min_names)
        if book.empty:
            print(f"  [{vname}] no usable months — skipped")
            continue

        periods = {"full": None, "since_2015": "2015-01-01", "since_2020": "2020-01-01"}
        recs = []
        for pname, since in periods.items():
            b = book if since is None else book[book.end_date >= pd.Timestamp(since)]
            if len(b) < 6:
                continue
            for col, disp in (("ls_decile", "LS decile (D10-D1)"),
                              ("ls_5050", "LS 50/50"),
                              ("top_decile", "TOP decile"),
                              ("bot_decile", "BOT decile"),
                              ("ew", "EW universe"),
                              ("ls_decile_LOGCONV", "LS decile [LOG conv - x-ref only]"),
                              ("ls_5050_LOGCONV", "LS 50/50 [LOG conv - x-ref only]")):
                st = series_stats(b[col])
                recs.append({"period": pname, "book": disp, **st})
        res = pd.DataFrame(recs)
        sheets[f"{vname}_results"] = res
        sheets[f"{vname}_deciles"] = decile_table(vdf, "score", args.n_deciles,
                                                  args.min_names)
        sheets[f"{vname}_monthly"] = book.drop(columns=["top_set", "bot_set"])

        head = res[(res.period == "full") & (res.book == "LS decile (D10-D1)")]
        h5050 = res[(res.period == "full") & (res.book == "LS 50/50")]
        summary["variants"][vname] = {
            "months": int(len(book)),
            "mean_names": round(float(book.n.mean()), 1),
            "skipped_months": meta["skipped_months"],
            "turnover_one_sided": turnover(book),
            "ls_decile_full": (head.iloc[0].to_dict() if len(head) else None),
            "ls_5050_full": (h5050.iloc[0].to_dict() if len(h5050) else None),
        }

        print(f"\n  [{vname}]  {len(book)} months, mean {book.n.mean():.0f} names")
        show = res[res.period == "full"][["book", "months", "cagr_pct", "sharpe",
                                          "nw_t", "computable"]]
        print(show.to_string(index=False))

    with pd.ExcelWriter(args.out_dir / "signal_card.xlsx", engine="openpyxl") as xw:
        for name, sh in sheets.items():
            sh.to_excel(xw, sheet_name=name[:31], index=False)
    (args.out_dir / "signal_card.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {args.out_dir / 'signal_card.xlsx'}")
    print(f"wrote {args.out_dir / 'signal_card.json'}")


if __name__ == "__main__":
    main()
