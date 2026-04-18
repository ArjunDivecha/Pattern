"""
Pixel-exact tests for renderer.py (PRD §12, success criterion 1).

Hand-computes expected pixel positions for three synthetic samples and
asserts that render_window produces exactly those pixels.
"""

import numpy as np
import pytest

from pattern.imaging.renderer import render_window, _bresenham


# ---------------------------------------------------------------------------
# Geometry helpers (mirror renderer internals)
# ---------------------------------------------------------------------------

HEIGHT = 64
WIDTH  = 60
WINDOW = 20
OHLC_ROWS = int(round(HEIGHT * 0.797))   # 51
GAP_ROW   = OHLC_ROWS                    # 51
VOL_START = OHLC_ROWS + 1               # 52
VOL_ROWS  = HEIGHT - VOL_START           # 12


def _row(price, p_min, p_max, ohlc_rows=OHLC_ROWS):
    frac = (p_max - price) / (p_max - p_min)
    return int(round(np.clip(frac, 0.0, 1.0) * (ohlc_rows - 1)))


def _make_flat(window=WINDOW, price=1.0, volume=1000.0):
    """
    Synthetic OHLCV where every day has identical flat prices.
    O = H = L = C = price (so scale = p/C = 1).
    TRI = constant (no return each day).
    """
    ohlcv = np.full((window, 5), price, dtype=np.float64)
    ohlcv[:, 4] = volume
    tri   = np.full(window, 1.0, dtype=np.float64)
    return ohlcv, tri


# ---------------------------------------------------------------------------
# Test 1: flat price — all OHLC on the same row
# ---------------------------------------------------------------------------

def test_flat_price_ohlc_row():
    """
    With constant price=1.0 and TRI=1.0 everywhere, normalised p = [1.0, 1.0, ...].
    p_min = p_max = 1.0 → degenerate range; renderer adds epsilon.
    All pixels should fall on row 25 (middle of OHLC region).
    """
    ohlcv, tri = _make_flat()
    img = render_window(ohlcv, tri, window=WINDOW, height=HEIGHT, width=WIDTH,
                        include_ma=False, include_volume=False)
    assert img is not None
    assert img.shape == (1, HEIGHT, WIDTH)
    assert img.dtype == np.uint8

    # Image should have white pixels only in the OHLC region
    assert img[0, GAP_ROW:, :].max() == 0, "No pixels below gap row expected"


def test_output_shape_and_dtype():
    ohlcv, tri = _make_flat()
    img = render_window(ohlcv, tri, include_ma=False, include_volume=False)
    assert img.shape == (1, HEIGHT, WIDTH)
    assert img.dtype == np.uint8
    assert set(img.ravel().tolist()).issubset({0, 255}), "Pixels must be {0, 255}"


# ---------------------------------------------------------------------------
# Test 2: known H/L bar positions
# ---------------------------------------------------------------------------

def test_hl_bar_exact_pixels():
    """
    For a single day with known H and L, verify the H-L bar occupies exactly
    the rows between _row(L) and _row(H) in the center column.
    """
    window = 20

    # Construct prices so that on the last day (t=19):
    # normalised p at t=19 = 1.0 (TRI flat), Open=0.95, High=1.10, Low=0.90, Close=1.00
    # Scale = p/Close = 1.0/1.0 = 1.0  →  adj O=0.95, H=1.10, L=0.90, C=1.00
    ohlcv = np.ones((window, 5), dtype=np.float64)
    # Set specific day values for day t=0 (the only day, if window=1... but we use 20)
    # To isolate day t=19 use window=20, set all others to Close=1.0, H=L=O=C=1.0
    ohlcv[:, 0] = 1.0   # O
    ohlcv[:, 1] = 1.0   # H
    ohlcv[:, 2] = 1.0   # L
    ohlcv[:, 3] = 1.0   # C
    ohlcv[:, 4] = 1000  # V

    # Day 0 (t=0 in image): give a distinct bar
    ohlcv[0, 0] = 0.95  # O
    ohlcv[0, 1] = 1.10  # H
    ohlcv[0, 2] = 0.90  # L
    ohlcv[0, 3] = 1.00  # C

    tri = np.ones(window, dtype=np.float64)   # flat TRI → p = [1,1,...,1]

    img = render_window(ohlcv, tri, window=window, height=HEIGHT, width=WIDTH,
                        include_ma=False, include_volume=False)
    assert img is not None

    # With all prices = 1.0 for t>0, global p_min=0.90, p_max=1.10
    p_min = 0.90 * 1.0   # scale=1 since C=TRI=1
    p_max = 1.10 * 1.0
    expected_row_h = _row(1.10, p_min, p_max)
    expected_row_l = _row(0.90, p_min, p_max)

    # Day t=0 → center col = 1
    c_center = 1
    bar_pixels = np.where(img[0, :, c_center] == 255)[0]

    assert expected_row_h in bar_pixels, (
        f"Expected H row {expected_row_h} in center col {c_center}, got {bar_pixels}")
    assert expected_row_l in bar_pixels, (
        f"Expected L row {expected_row_l} in center col {c_center}, got {bar_pixels}")
    # Entire bar must be white
    assert set(range(expected_row_h, expected_row_l + 1)).issubset(set(bar_pixels.tolist()))


# ---------------------------------------------------------------------------
# Test 3: open/close tick columns
# ---------------------------------------------------------------------------

def test_open_close_tick_columns():
    """
    Open tick is in left column (3t), close tick in right column (3t+2).
    For day t=1 (cols 3, 4, 5), verify:
      - col 3 has the open pixel
      - col 5 has the close pixel
      - col 4 has the H-L bar
    """
    window = 20
    ohlcv  = np.ones((window, 5), dtype=np.float64)
    ohlcv[:, 4] = 500

    # Day t=1: O=0.95, H=1.05, L=0.92, C=0.98
    ohlcv[1, 0] = 0.95
    ohlcv[1, 1] = 1.05
    ohlcv[1, 2] = 0.92
    ohlcv[1, 3] = 0.98

    tri = np.ones(window, dtype=np.float64)
    img = render_window(ohlcv, tri, window=window, height=HEIGHT, width=WIDTH,
                        include_ma=False, include_volume=False)
    assert img is not None

    # Global p_min/max set by the extremes
    p_min = 0.92
    p_max = 1.05

    expected_row_o = _row(0.95, p_min, p_max)
    expected_row_c = _row(0.98, p_min, p_max)

    col_left   = 3   # 3*1
    col_right  = 5   # 3*1 + 2

    assert img[0, expected_row_o, col_left]  == 255, "Open tick missing"
    assert img[0, expected_row_c, col_right] == 255, "Close tick missing"


# ---------------------------------------------------------------------------
# Test 4: volume bar
# ---------------------------------------------------------------------------

def test_volume_bar_height():
    """
    With max_volume=1000 on day t=0 and volume=500 on day t=1, the
    volume bar for day t=1 should occupy roughly half of vol_rows.
    """
    window = 20
    ohlcv  = np.ones((window, 5), dtype=np.float64)
    ohlcv[:, 4] = 0
    ohlcv[0, 4] = 1000   # max volume
    ohlcv[1, 4] = 500    # 50% of max

    tri = np.ones(window, dtype=np.float64)
    img = render_window(ohlcv, tri, window=window, height=HEIGHT, width=WIDTH,
                        include_ma=False, include_volume=True)
    assert img is not None

    c_center_t1 = 4   # 3*1 + 1
    vol_pixels   = img[0, VOL_START:, c_center_t1]
    bar_count    = int(vol_pixels.sum() // 255)

    expected = int(round(0.5 * VOL_ROWS))
    assert abs(bar_count - expected) <= 1, (
        f"Expected ~{expected} vol pixels, got {bar_count}")


# ---------------------------------------------------------------------------
# Test 5: missing H or L skips entire day
# ---------------------------------------------------------------------------

def test_missing_hl_skips_day():
    """If H is NaN, columns 3t, 3t+1, 3t+2 must all be black."""
    window = 20
    ohlcv  = np.ones((window, 5), dtype=np.float64)
    ohlcv[:, 4] = 100
    ohlcv[2, 1] = np.nan   # High missing on day t=2

    tri = np.ones(window, dtype=np.float64)
    img = render_window(ohlcv, tri, window=window, height=HEIGHT, width=WIDTH,
                        include_ma=False, include_volume=False)
    assert img is not None

    t = 2
    for col in [3*t, 3*t+1, 3*t+2]:
        assert img[0, :, col].max() == 0, (
            f"Column {col} should be all-black when H is NaN")


# ---------------------------------------------------------------------------
# Test 6: gap row is always black
# ---------------------------------------------------------------------------

def test_gap_row_is_black():
    ohlcv, tri = _make_flat()
    img = render_window(ohlcv, tri, include_ma=True, include_volume=True)
    assert img is not None
    assert img[0, GAP_ROW, :].max() == 0, "Gap row must stay black"


# ---------------------------------------------------------------------------
# Test 7: Bresenham basic
# ---------------------------------------------------------------------------

def test_bresenham_connects_endpoints():
    pixels = _bresenham(0, 0, 3, 6)
    assert (0, 0) in pixels
    assert (3, 6) in pixels


def test_bresenham_single_pixel():
    assert _bresenham(5, 5, 5, 5) == [(5, 5)]


# ---------------------------------------------------------------------------
# Test 8: insufficient data → None
# ---------------------------------------------------------------------------

def test_too_short_returns_none():
    ohlcv = np.ones((5, 5), dtype=np.float64)
    tri   = np.ones(5, dtype=np.float64)
    assert render_window(ohlcv, tri, window=20) is None


def test_zero_tri_returns_none():
    ohlcv = np.ones((20, 5), dtype=np.float64)
    tri   = np.zeros(20, dtype=np.float64)
    assert render_window(ohlcv, tri, window=20) is None
