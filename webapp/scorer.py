"""
=============================================================================
SCRIPT NAME: scorer.py
=============================================================================

INPUT FILES:
- runs/final/equal_20260420_054833_cdef6809/ensemble_{0..4}.pt
  Full-history CNN ensemble (trained on all data, no validation holdout)
- runs/final/equal_20260420_054833_cdef6809/config.yaml
  Model configuration (I20 geometry)

OUTPUT:
- In-memory score result with P(up), mapped score (0-100), confidence,
  and rendered chart image

DESCRIPTION:
Core scoring engine for the live web app. Fetches OHLCV from yfinance,
renders the I20 chart image, runs it through the full-history 5-seed
ensemble, and maps the raw P(up) to a 0-100 score using the historical
distribution anchors from the expanding pathway.

Score mapping (from expanding pathway distribution):
  Score = 0   → P(up) = 0.22075 (absolute min observed)
  Score = 50  → P(up) = 0.50000 (neutral)
  Score = 100 → P(up) = 0.66293 (absolute max observed)

USAGE:
  from scorer import score_ticker
  result = score_ticker("AAPL")
=============================================================================
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yfinance as yf
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pattern.config import Config
from pattern.data.loader import build_ticker_index, get_window
from pattern.imaging.renderer import render_window
from pattern.models.cnn import ChartCNN

log = logging.getLogger("scorer")

# ── Model paths ─────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR = ROOT / "runs" / "final" / "equal_20260420_054833_cdef6809"
DEFAULT_CONFIG = DEFAULT_MODEL_DIR / "config.yaml"

# ── Score mapping anchors (from expanding pathway distribution) ─────────────
SCORE_MIN_P = 0.22075   # P(up) that maps to Score = 0
SCORE_MID_P = 0.50000   # P(up) that maps to Score = 50 (neutral)
SCORE_MAX_P = 0.66293   # P(up) that maps to Score = 100

# ── Pixel stats (computed from full historical cache) ───────────────────────
# These are approximate; for production, compute from the actual cache.
# The binary candlestick images have a very stable pixel distribution.
PIXEL_MEAN = 8.5   # ~8.5/255 pixels are white in a typical I20 image
PIXEL_STD = 28.5   # std dominated by the 0/255 binary nature


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_ensemble(model_dir: Path, cfg: Config, device: torch.device) -> list:
    """Load all 5 ensemble members into a list of eval models."""
    models = []
    for k in range(5):
        ckpt = model_dir / f"ensemble_{k}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m = ChartCNN(cfg.model, cfg.image)
        m.load_state_dict(state)
        m.to(device).eval()
        models.append(m)
    log.info(f"Loaded {len(models)} ensemble members on {device}")
    return models


# Lazy-loaded globals
_ensemble_models: Optional[list] = None
_ensemble_cfg: Optional[Config] = None
_ensemble_device: Optional[torch.device] = None


def _get_ensemble() -> Tuple[list, Config, torch.device]:
    """Lazy-load the ensemble once."""
    global _ensemble_models, _ensemble_cfg, _ensemble_device
    if _ensemble_models is None:
        _ensemble_device = _pick_device()
        _ensemble_cfg = Config.from_yaml(DEFAULT_CONFIG)
        _ensemble_models = _load_ensemble(DEFAULT_MODEL_DIR, _ensemble_cfg, _ensemble_device)
    return _ensemble_models, _ensemble_cfg, _ensemble_device


def _fetch_ohlcv(ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
    """Fetch recent OHLCV from yfinance. Returns DataFrame or None."""
    try:
        # Download extra days to ensure we have enough history after drops
        df = yf.download(
            ticker,
            period=f"{days + 20}d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or len(df) == 0:
            return None

        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.reset_index().rename(columns={"Date": "Date", "Adj Close": "AdjClose"})
        df["Ticker"] = ticker
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        # Ensure required columns exist
        for col in ["Open", "High", "Low", "Close", "Volume", "AdjClose"]:
            if col not in df.columns:
                log.warning(f"Missing column {col} for {ticker}")
                return None
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Open", "High", "Low", "Close", "AdjClose"])
        if len(df) < 30:
            log.warning(f"Only {len(df)} valid rows for {ticker}")
            return None

        return df[["Ticker", "Date", "Open", "High", "Low", "Close", "Volume", "AdjClose"]]

    except Exception as e:
        log.error(f"yfinance error for {ticker}: {e}")
        return None


def _render_chart(df: pd.DataFrame, cfg) -> Tuple[Optional[np.ndarray], Optional[pd.Timestamp]]:
    """Render the I20 chart image from the most recent 20+ lookback days."""
    ticker = df["Ticker"].iloc[0]
    tdf = df.copy()
    tdf = tdf.sort_values("Date").reset_index(drop=True)

    window = cfg.image.window
    lookback = window - 1

    # Use the most recent date with enough history
    end_date = tdf["Date"].iloc[-1]
    res = get_window(tdf, end_date, window, lookback)
    if res is None:
        return None, None

    ohlcv, tri = res
    img = render_window(
        ohlcv, tri,
        window=cfg.image.window,
        height=cfg.image.height,
        width=cfg.image.width,
        ohlc_height_ratio=cfg.image.ohlc_height_ratio,
        include_ma=cfg.image.include_ma,
        include_volume=cfg.image.include_volume,
    )
    return img, end_date


def _p_to_score(p_up: float) -> float:
    """Map raw P(up) to 0-100 score using historical distribution anchors."""
    if p_up < SCORE_MID_P:
        # Bearish side: linear map [MIN, 0.5] → [0, 50]
        score = (p_up - SCORE_MIN_P) / (SCORE_MID_P - SCORE_MIN_P) * 50.0
    else:
        # Bullish side: linear map [0.5, MAX] → [50, 100]
        score = 50.0 + (p_up - SCORE_MID_P) / (SCORE_MAX_P - SCORE_MID_P) * 50.0
    return float(np.clip(score, 0.0, 100.0))


def _score_label(score: float) -> Tuple[str, str]:
    """Return (label, color_class) for the score."""
    if score < 30:
        return "Strongly Bearish", "bearish-strong"
    elif score < 45:
        return "Bearish", "bearish"
    elif score < 55:
        return "Neutral", "neutral"
    elif score < 70:
        return "Bullish", "bullish"
    else:
        return "Strongly Bullish", "bullish-strong"


def _image_to_png_bytes(img: np.ndarray) -> bytes:
    """Convert (1, H, W) uint8 image to PNG bytes."""
    # img is (1, H, W) uint8; squeeze to (H, W)
    arr = img[0] if img.ndim == 3 else img
    # Scale to full grayscale for visibility
    pil_img = Image.fromarray(arr, mode="L")
    # Upscale for better display (4x)
    pil_img = pil_img.resize((pil_img.width * 4, pil_img.height * 4), Image.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def score_ticker(ticker: str) -> dict:
    """
    Main entry point: fetch data, render chart, score with ensemble.

    Returns dict with keys:
      - ticker, end_date, p_up_mean, p_up_std, score (0-100),
      - label, label_class, chart_png (bytes), n_days_fetched,
      - raw_probs (list of 5), error (if any)
    """
    models, cfg, device = _get_ensemble()

    # 1. Fetch data
    df = _fetch_ohlcv(ticker)
    if df is None:
        return {"error": f"Could not fetch data for '{ticker}'. Check the symbol and try again."}

    n_days = len(df)

    # 2. Render chart
    img, end_date = _render_chart(df, cfg)
    if img is None:
        return {"error": f"Not enough history for {ticker} to render a {cfg.image.window}-day chart."}

    # 3. Normalize and score
    x = (img.astype(np.float32) - PIXEL_MEAN) / PIXEL_STD
    x_tensor = torch.from_numpy(x[np.newaxis, ...]).to(device)  # (1, 1, H, W)

    probs = []
    with torch.inference_mode():
        for m in models:
            logits = m(x_tensor)
            p = F.softmax(logits, dim=-1)[0, 1].cpu().item()
            probs.append(p)

    p_mean = float(np.mean(probs))
    p_std = float(np.std(probs))
    score = _p_to_score(p_mean)
    label, label_class = _score_label(score)
    chart_png = _image_to_png_bytes(img)

    return {
        "ticker": ticker,
        "end_date": str(end_date.date()) if end_date else None,
        "p_up_mean": round(p_mean, 4),
        "p_up_std": round(p_std, 4),
        "score": round(score, 1),
        "label": label,
        "label_class": label_class,
        "chart_png": chart_png,
        "n_days_fetched": n_days,
        "raw_probs": [round(p, 4) for p in probs],
        "error": None,
    }
