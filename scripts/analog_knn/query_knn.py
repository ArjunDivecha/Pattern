"""
=============================================================================
SCRIPT NAME: query_knn.py
=============================================================================

INPUT FILES:
- {embeddings_dir}/embeddings.npz
- data/r1000_ohlcv_spliced.parquet  (trading calendar for outcome completion)

OUTPUT FILES:
- {out_dir}/analog_scores.parquet
- {out_dir}/knn_meta.json

DESCRIPTION:
Leakage-safe cross-ticker visual kNN (PRD §6–7).
  L1: j != i
  L2: neighbor outcome fully known at query date (s + H trading days <= t)
  L3: embeddings already walk-forward (from extract_embeddings)
  L4: only rows in the embeddings panel

Primary score: analog_r_mean (mean of neighbor log forward returns).
=============================================================================
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
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("analog_knn.query")

DEFAULT_OHLCV = ROOT / "data" / "r1000_ohlcv_spliced.parquet"
HORIZON = 20


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def trading_calendar(ohlcv_path: Path) -> np.ndarray:
    o = pd.read_parquet(ohlcv_path, columns=["Date"])
    dates = np.sort(pd.to_datetime(o["Date"]).dropna().unique())
    return dates.astype("datetime64[ns]")


def outcome_available_dates(
    end_dates: np.ndarray, calendar: np.ndarray, horizon: int = HORIZON
) -> np.ndarray:
    """For each formation end_date, first calendar day on/after end_date + horizon trading days."""
    cal = calendar.astype("datetime64[ns]")
    ends = end_dates.astype("datetime64[ns]")
    # searchsorted: index of first cal >= end
    pos = np.searchsorted(cal, ends, side="left")
    out_pos = pos + horizon
    avail = np.empty(len(ends), dtype="datetime64[ns]")
    ok = out_pos < len(cal)
    avail[ok] = cal[out_pos[ok]]
    # If not enough future calendar, mark as far future so they never enter bank early
    avail[~ok] = np.datetime64("2100-01-01", "ns")
    # If end_date itself not found (pos==len), also far future
    missing = pos >= len(cal)
    avail[missing] = np.datetime64("2100-01-01", "ns")
    return avail


def load_embeddings(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "ticker": z["ticker"].astype(str),
        "end_date": z["end_date"].astype("datetime64[ns]"),
        "window": z["window"].astype(np.int16),
        "forward_return": z["forward_return"].astype(np.float64),
        "label": z["label"].astype(np.int8),
        "emb": z["emb"].astype(np.float32),
    }


def knn_for_date_multi_k(
    q_emb: torch.Tensor,
    q_ticker_codes: np.ndarray,
    bank_emb: torch.Tensor,
    bank_ticker_codes: np.ndarray,
    bank_fwd: np.ndarray,
    k_list: list[int],
    device: torch.device,
    bank_chunk: int = 50_000,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    One retrieval pass; aggregate for each k in k_list.
    Returns {k: (mean_r, med_r, hit, dist_mean, k_eff)}.
    """
    q = q_emb.shape[0]
    b = bank_emb.shape[0]
    max_k = max(k_list)
    empty = {
        k: (
            np.full(q, np.nan),
            np.full(q, np.nan),
            np.full(q, np.nan),
            np.full(q, np.nan),
            np.zeros(q),
        )
        for k in k_list
    }
    if b == 0 or q == 0:
        return empty

    # Over-fetch then drop same-ticker history of the query name
    buf = min(b, max(max_k * 8, max_k + 100))
    best_sims = torch.full((q, buf), -1e9, device=device)
    best_idx = torch.zeros((q, buf), dtype=torch.long, device=device)

    offset = 0
    while offset < b:
        end = min(offset + bank_chunk, b)
        chunk = bank_emb[offset:end]
        sims = q_emb @ chunk.T
        cat_sims = torch.cat([best_sims, sims], dim=1)
        cat_idx = torch.cat(
            [
                best_idx,
                torch.arange(offset, end, device=device).unsqueeze(0).expand(q, -1),
            ],
            dim=1,
        )
        top_sims, top_pos = torch.topk(
            cat_sims, k=min(buf, cat_sims.shape[1]), dim=1
        )
        best_sims = top_sims
        best_idx = torch.gather(cat_idx, 1, top_pos)
        offset = end

    best_sims_np = best_sims.cpu().numpy()
    best_idx_np = best_idx.cpu().numpy()

    out: dict[int, tuple] = {}
    for k in k_list:
        mean_r = np.empty(q, dtype=np.float64)
        med_r = np.empty(q, dtype=np.float64)
        hit = np.empty(q, dtype=np.float64)
        dist_mean = np.empty(q, dtype=np.float64)
        k_eff = np.empty(q, dtype=np.float64)
        for i in range(q):
            idxs = best_idx_np[i]
            sims = best_sims_np[i]
            mask = (bank_ticker_codes[idxs] != q_ticker_codes[i]) & (sims > -1e8)
            idxs = idxs[mask]
            sims = sims[mask]
            if len(idxs) == 0:
                mean_r[i] = med_r[i] = hit[i] = dist_mean[i] = np.nan
                k_eff[i] = 0
                continue
            take = min(k, len(idxs))
            idxs_k = idxs[:take]
            sims_k = sims[:take]
            rets = bank_fwd[idxs_k]
            mean_r[i] = float(np.nanmean(rets))
            med_r[i] = float(np.nanmedian(rets))
            hit[i] = float(np.mean(rets > 0))
            dist_mean[i] = float(np.mean(1.0 - sims_k))
            k_eff[i] = float(take)
        out[k] = (mean_r, med_r, hit, dist_mean, k_eff)
    return out


def run_knn(
    data: dict,
    calendar: np.ndarray,
    k_list: list[int],
    device: torch.device,
    bank_chunk: int = 50_000,
) -> pd.DataFrame:
    n = len(data["ticker"])
    tickers = data["ticker"]
    end_dates = data["end_date"]
    fwd = data["forward_return"]
    emb = data["emb"]
    # unit-norm safety
    norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    emb = emb / norms

    # factorize tickers for fast compare
    codes, uniques = pd.factorize(tickers, sort=False)
    codes = codes.astype(np.int32)

    avail = outcome_available_dates(end_dates, calendar, HORIZON)

    # Sort by end_date for chronological processing
    order = np.argsort(end_dates, kind="mergesort")
    end_dates_s = end_dates[order]
    codes_s = codes[order]
    fwd_s = fwd[order]
    emb_s = emb[order]
    avail_s = avail[order]
    tickers_s = tickers[order]
    window_s = data["window"][order]
    label_s = data["label"][order]
    # map original positions after sort for stable output
    # We'll emit in sorted order then that's fine.

    unique_dates = np.unique(end_dates_s)

    results: dict[str, list] = {
        "ticker": [],
        "end_date": [],
        "window": [],
        "forward_return": [],
        "label": [],
    }
    for k in k_list:
        results[f"analog_r_mean_k{k}"] = []
        results[f"analog_r_med_k{k}"] = []
        results[f"analog_hit_k{k}"] = []
        results[f"analog_dist_mean_k{k}"] = []
        results[f"analog_k_eff_k{k}"] = []

    bank_indices: list[int] = []
    avail_order = np.argsort(avail_s, kind="mergesort")
    avail_sorted = avail_s[avail_order]
    avail_ptr = 0

    emb_t = torch.from_numpy(emb_s).to(device)

    for di, t in enumerate(unique_dates):
        t = np.datetime64(t, "ns")
        while avail_ptr < n and avail_sorted[avail_ptr] <= t:
            bank_indices.append(int(avail_order[avail_ptr]))
            avail_ptr += 1

        left = np.searchsorted(end_dates_s, t, side="left")
        right = np.searchsorted(end_dates_s, t, side="right")
        if left == right:
            continue
        q_idx = np.arange(left, right)

        if not bank_indices:
            for i in q_idx:
                results["ticker"].append(tickers_s[i])
                results["end_date"].append(end_dates_s[i])
                results["window"].append(int(window_s[i]))
                results["forward_return"].append(fwd_s[i])
                results["label"].append(int(label_s[i]))
                for k in k_list:
                    results[f"analog_r_mean_k{k}"].append(np.nan)
                    results[f"analog_r_med_k{k}"].append(np.nan)
                    results[f"analog_hit_k{k}"].append(np.nan)
                    results[f"analog_dist_mean_k{k}"].append(np.nan)
                    results[f"analog_k_eff_k{k}"].append(0.0)
            continue

        bank_idx_arr = np.asarray(bank_indices, dtype=np.int64)
        q_emb = emb_t[q_idx]
        b_emb = emb_t[bank_idx_arr]
        b_codes = codes_s[bank_idx_arr]
        b_fwd = fwd_s[bank_idx_arr]
        q_codes = codes_s[q_idx]

        per_k = knn_for_date_multi_k(
            q_emb,
            q_codes,
            b_emb,
            b_codes,
            b_fwd,
            k_list=k_list,
            device=device,
            bank_chunk=bank_chunk,
        )

        for local_i, i in enumerate(q_idx):
            results["ticker"].append(tickers_s[i])
            results["end_date"].append(end_dates_s[i])
            results["window"].append(int(window_s[i]))
            results["forward_return"].append(fwd_s[i])
            results["label"].append(int(label_s[i]))
            for k in k_list:
                mr, md, ht, dm, ke = per_k[k]
                results[f"analog_r_mean_k{k}"].append(mr[local_i])
                results[f"analog_r_med_k{k}"].append(md[local_i])
                results[f"analog_hit_k{k}"].append(ht[local_i])
                results[f"analog_dist_mean_k{k}"].append(dm[local_i])
                results[f"analog_k_eff_k{k}"].append(ke[local_i])

        if (di + 1) % 50 == 0 or di == len(unique_dates) - 1:
            log.info(
                "date %s  queries=%d  bank=%s  (%d/%d dates)",
                str(t)[:10],
                len(q_idx),
                f"{len(bank_indices):,}",
                di + 1,
                len(unique_dates),
            )

    df = pd.DataFrame(results)
    df["end_date"] = pd.to_datetime(df["end_date"])
    # Convenience aliases for primary k
    primary_k = 50 if 50 in k_list else k_list[0]
    df["analog_r_mean"] = df[f"analog_r_mean_k{primary_k}"]
    df["analog_r_med"] = df[f"analog_r_med_k{primary_k}"]
    df["analog_hit"] = df[f"analog_hit_k{primary_k}"]
    df["analog_dist_mean"] = df[f"analog_dist_mean_k{primary_k}"]
    df["analog_k_eff"] = df[f"analog_k_eff_k{primary_k}"]
    return df


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, required=True, help="embeddings.npz")
    ap.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bank-chunk", type=int, default=50_000)
    ap.add_argument(
        "--predictions-monthly",
        type=Path,
        default=None,
        help="Optional: merge p_up_mean for comparison columns",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
    )
    device = resolve_device(args.device)
    log.info("Device: %s  k=%s", device, args.k)

    data = load_embeddings(args.embeddings)
    log.info("Loaded embeddings N=%s dim=%s", f"{len(data['ticker']):,}", data["emb"].shape[1])

    cal = trading_calendar(args.ohlcv)
    log.info("Trading calendar %s → %s  (%s days)", cal[0], cal[-1], f"{len(cal):,}")

    scores = run_knn(data, cal, list(args.k), device, bank_chunk=args.bank_chunk)

    if args.predictions_monthly and args.predictions_monthly.exists():
        pred = pd.read_parquet(args.predictions_monthly)
        pred["end_date"] = pd.to_datetime(pred["end_date"])
        pred = pred[["ticker", "end_date", "p_up_mean"]].drop_duplicates(
            ["ticker", "end_date"]
        )
        scores = scores.drop(columns=["p_up_mean"], errors="ignore")
        scores = scores.merge(pred, on=["ticker", "end_date"], how="left")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "analog_scores.parquet"
    scores.to_parquet(out_path, index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(scores)),
        "k_list": list(args.k),
        "horizon_trading_days": HORIZON,
        "embeddings": str(args.embeddings),
        "date_min": str(scores["end_date"].min().date()),
        "date_max": str(scores["end_date"].max().date()),
        "mean_k_eff_k50": float(
            scores.get("analog_k_eff_k50", scores["analog_k_eff"]).mean()
        ),
        "frac_nan_analog_r_mean": float(scores["analog_r_mean"].isna().mean()),
    }
    (args.out_dir / "knn_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Wrote %s  rows=%s  nan_frac=%.3f", out_path, f"{len(scores):,}", meta["frac_nan_analog_r_mean"])
    return out_path


if __name__ == "__main__":
    main()
