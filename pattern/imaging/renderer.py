"""
=============================================================================
SCRIPT NAME: renderer.py
=============================================================================
DESCRIPTION:
Core image renderer. Converts a OHLCV + TotalReturnIndex window into the
64×60 (or configurable) grayscale chart image described in PRD §3–4.

Entry points:
  render_window(ohlcv, tri, ...)  — single image, returns (1, H, W) uint8
  render_batch(samples, ...)      — list of (ohlcv, tri) → stacked array
=============================================================================
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Bresenham line (avoids skimage dependency for a single primitive)
# ---------------------------------------------------------------------------

def _bresenham(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    """Return list of (row, col) pixels on the line from (r0,c0) to (r1,c1)."""
    pixels = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        pixels.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r   += sr
        if e2 < dr:
            err += dr
            c   += sc
    return pixels


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

def render_window(
    ohlcv: np.ndarray,
    tri: np.ndarray,
    window: int = 20,
    height: int = 64,
    width: int = 60,
    ohlc_height_ratio: float = 0.797,
    include_ma: bool = True,
    include_volume: bool = True,
) -> Optional[np.ndarray]:
    """
    Render one chart image.

    Args:
        ohlcv:  (N, 5) float64 — [Open, High, Low, Close, Volume] for N days.
                The LAST `window` rows are the image; earlier rows provide MA
                lookback.  N must be >= window.  Use np.nan for missing values.
        tri:    (N,) float64 — TotalReturnIndex (Bloomberg TOT_RETURN_INDEX_GROSS_DVDS)
                for the same N rows.
        window: Trading days per image (default 20 for I20).
        height: Pixel height (default 64).
        width:  Pixel width (default 60 = 3 * 20).
        ohlc_height_ratio: Fraction of height devoted to OHLC region (≈0.797 for 51/64).
        include_ma: Draw the moving-average line (PRD §4.2).
        include_volume: Draw volume bars in the bottom region.

    Returns:
        (1, height, width) uint8 array with pixels in {0, 255}, or None if
        insufficient data.
    """
    N = len(ohlcv)
    if N < window:
        return None

    # ── Geometry ─────────────────────────────────────────────────────────────
    ohlc_rows = int(round(height * ohlc_height_ratio))   # 51
    gap_row   = ohlc_rows                                # row 51 (blank)
    vol_start = ohlc_rows + 1                            # row 52
    vol_rows  = height - vol_start                       # 12

    # ── Extract image window (last `window` rows) ─────────────────────────────
    O_raw = ohlcv[-window:, 0]
    H_raw = ohlcv[-window:, 1]
    L_raw = ohlcv[-window:, 2]
    C_raw = ohlcv[-window:, 3]
    V_raw = ohlcv[-window:, 4]
    tri_w = tri[-window:]

    # ── Price normalisation (PRD §3) ─────────────────────────────────────────
    # Step 1–2: normalise first image close to 1.0, reconstruct via TRI
    tri0 = tri_w[0]
    if not np.isfinite(tri0) or tri0 <= 0:
        return None
    p = tri_w / tri0   # normalised close, p[0] = 1.0

    # Step 3: scale O/H/L by p_t / Close_raw_t
    # Only use Close_raw for the scale factor; if Close is missing the normalised
    # close p is still valid (from TRI), so we mark only the close TICK as missing
    # rather than blacking out the whole day (PRD §4.3).
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(C_raw > 0, p / C_raw, np.nan)
    O = O_raw * scale
    H = H_raw * scale
    L = L_raw * scale
    # C_adj = p regardless of whether raw Close is NaN (TRI-driven close is always valid)
    C = p.copy()

    # Step 4: rescale so global OHLC min/max fills the OHLC region
    all_px = np.concatenate([O, H, L, C])
    valid  = all_px[np.isfinite(all_px)]
    if len(valid) == 0:
        return None
    p_min, p_max = valid.min(), valid.max()
    if p_max == p_min:
        p_max = p_min + 1e-8

    def _row(price: float) -> int:
        """Map normalised price → pixel row (0 = top = highest price)."""
        frac = (p_max - price) / (p_max - p_min)
        return int(round(np.clip(frac, 0.0, 1.0) * (ohlc_rows - 1)))

    # ── Moving average (uses full N rows for lookback) ─────────────────────
    # MA_t = W-period trailing mean of normalised close, undefined if < W points available
    ma_rows: List[Optional[int]] = [None] * window
    if include_ma:
        full_p = tri / tri0        # normalised close for all N days
        for t in range(window):
            full_idx = N - window + t
            start    = full_idx - window + 1
            if start < 0:
                continue           # not enough history — leave undefined
            segment = full_p[start : full_idx + 1]
            valid_s = segment[np.isfinite(segment)]
            if len(valid_s) >= window:
                ma_rows[t] = _row(float(valid_s.mean()))

    # ── Volume normalisation (Step 5) ────────────────────────────────────────
    V_valid = V_raw[np.isfinite(V_raw) & (V_raw >= 0)]
    vol_max = float(V_valid.max()) if len(V_valid) > 0 else 1.0
    if vol_max <= 0:
        vol_max = 1.0

    # ── Draw ──────────────────────────────────────────────────────────────────
    img = np.zeros((height, width), dtype=np.uint8)

    prev_ma: Optional[Tuple[int, int]] = None   # (row, col) of last drawn MA pixel

    for t in range(window):
        c_left   = 3 * t
        c_center = 3 * t + 1
        c_right  = 3 * t + 2

        # PRD §4.3: if H or L missing → whole day black (skip all columns)
        h_missing = not np.isfinite(H[t])
        l_missing = not np.isfinite(L[t])
        if h_missing or l_missing:
            prev_ma = None
            continue

        # H–L bar (center column)
        r_h = _row(float(H[t]))
        r_l = _row(float(L[t]))
        img[r_h : r_l + 1, c_center] = 255

        # Open tick (left column) — omit if missing (PRD §4.3)
        # scale is NaN when C_raw missing, so O will also be NaN → omit tick
        if np.isfinite(O[t]):
            img[_row(float(O[t])), c_left] = 255

        # Close tick (right column) — C = p (TRI-based), always finite;
        # but raw Close may be missing → in that case omit the right tick only
        # C_raw missing ↔ scale[t] is NaN ↔ we can detect via raw array
        if np.isfinite(C_raw[t]):
            img[_row(float(C[t])), c_right] = 255

        # MA pixel + Bresenham line to previous defined MA point
        if include_ma and ma_rows[t] is not None:
            r_ma = ma_rows[t]
            img[r_ma, c_center] = 255
            if prev_ma is not None:
                for r, c in _bresenham(prev_ma[0], prev_ma[1], r_ma, c_center):
                    if 0 <= r < ohlc_rows and 0 <= c < width:
                        img[r, c] = 255
            prev_ma = (r_ma, c_center)
        else:
            prev_ma = None    # gap → don't connect across this day

        # Volume bar (center column, volume region, bottom-up)
        if include_volume and np.isfinite(V_raw[t]) and V_raw[t] >= 0:
            v_norm   = float(V_raw[t]) / vol_max
            bar_h    = int(round(v_norm * vol_rows))
            if bar_h > 0:
                top = max(height - bar_h, vol_start)
                img[top:height, c_center] = 255

    return img[np.newaxis]   # (1, H, W)


def render_batch(
    samples: List[Tuple[np.ndarray, np.ndarray]],
    window: int = 20,
    height: int = 64,
    width: int = 60,
    ohlc_height_ratio: float = 0.797,
    include_ma: bool = True,
    include_volume: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Render a list of (ohlcv, tri) samples.

    Returns:
        images:      (M, 1, H, W) uint8 array for the M valid samples.
        valid_idx:   indices into `samples` that produced valid images.
    """
    images    = []
    valid_idx = []
    for i, (ohlcv, tri) in enumerate(samples):
        img = render_window(
            ohlcv, tri, window, height, width,
            ohlc_height_ratio, include_ma, include_volume,
        )
        if img is not None:
            images.append(img)
            valid_idx.append(i)

    if images:
        return np.stack(images, axis=0), valid_idx
    return np.empty((0, 1, height, width), dtype=np.uint8), []
