# PRD — Residual Pattern + Dixon Geometry + Attention Gate

**Codename:** `residual-geometry`  
**Parent:** Pattern  
**Status:** Phase 0 implementation  
**Date:** 2026-08-09  

## Goal

Harden the only visual-adjacent equity signal that already cleared a gate on this desk: **`p_up` residualized against free baselines**. Add Dixon geometric features as interpretability / residualization controls, and an **FOMC attention gate** as a cheap timing overlay.

Not a new CNN/ViT. No new market data beyond OHLCV + FOMC calendar.

## Success / kill (Full sample, $5 floor, simple returns via `evaluate_signal.py`)

| ID | Criterion |
|----|-----------|
| **S1** | `p_up_resid` IC NW t > 2 **or** LS 50/50 NW t > 2 |
| **S2** | Residual after geometry still t > 1.5 (not pure Dixon/WSTR) |
| **S3** | FOMC-gated residual improves IC t or LS t vs ungated (else gate is null) |
| **Kill residual path** | S1 fails after contemporaneous CS residual (matches gauntlet) |
| **Expression** | If S1 holds → feature export only, not standalone small-cap book |

## Signals produced

| Column | Definition |
|--------|------------|
| `p_up_mean` | CNN ensemble (input) |
| `p_up_resid` | CS residual of z(p_up) on z(oriented baselines) within month |
| `geom_score` | CS z-score blend of Dixon-style geometry (higher = buy if oriented) |
| `p_up_resid_geom` | CS residual of z(p_up) on geometry features only |
| `p_up_resid_full` | CS residual of z(p_up) on baselines + geometry |
| `p_up_resid_nofomc` | `p_up_resid` with NaN in FOMC attention windows |
| `geom_rlcc` | Last-bar relative close-change (Dixon WRLCC analogue, I20) |

## Attention window

Formation `end_date` is in FOMC window if within **[-2, +5] trading days** of a Fed FOMC decision date (`data/fomc_dates.csv`).

## Out of scope

ViT, analog kNN, futures transfer, live trading, cost gates.
