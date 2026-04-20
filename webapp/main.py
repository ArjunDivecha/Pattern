"""
=============================================================================
SCRIPT NAME: main.py
=============================================================================

INPUT:
- User-provided stock ticker via web form
- runs/final/equal_20260420_054833_cdef6809/ensemble_{0..4}.pt

OUTPUT:
- Web page showing rendered chart, score (0-100), confidence, and label

DESCRIPTION:
FastAPI web application for live stock scoring. Serves a beautiful
single-page UI where users enter a ticker and get an instant CNN-based
bullish/bearish score with a rendered candlestick chart.

USAGE:
  cd webapp && uvicorn main:app --reload --host 0.0.0.0 --port 8000
  Then open http://localhost:8000 in your browser.
=============================================================================
"""
from __future__ import annotations

import base64
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scorer import score_ticker

app = FastAPI(title="Pattern Live Scorer", version="1.0")

# Templates
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main scoring page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/score")
async def score(ticker: str = Form(...)):
    """Score a ticker and return JSON result."""
    ticker = ticker.strip().upper()
    if not ticker:
        return JSONResponse({"error": "Please enter a ticker symbol."}, status_code=400)

    result = score_ticker(ticker)

    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=400)

    # Encode chart PNG as base64 for inline display
    chart_b64 = base64.b64encode(result["chart_png"]).decode("utf-8")

    return JSONResponse({
        "ticker": result["ticker"],
        "end_date": result["end_date"],
        "p_up_mean": result["p_up_mean"],
        "p_up_std": result["p_up_std"],
        "score": result["score"],
        "label": result["label"],
        "label_class": result["label_class"],
        "chart_b64": f"data:image/png;base64,{chart_b64}",
        "n_days_fetched": result["n_days_fetched"],
        "raw_probs": result["raw_probs"],
    })


@app.get("/health")
async def health():
    return {"status": "ok"}
