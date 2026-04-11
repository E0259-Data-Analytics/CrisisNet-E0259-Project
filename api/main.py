"""
CrisisNet — FastAPI Endpoint
==============================
Serves per-ticker health scores from the trained LightGBM fusion model.

Endpoints:
    GET /score/{ticker}            Latest quarter score
    GET /score/{ticker}/{quarter}  Specific quarter score
    GET /scores                    All scores (optional ?ticker=CHK&min_year=2020)
    GET /tickers                   List of available tickers
    GET /health                    Service health check

Usage:
    pip install fastapi uvicorn lightgbm pandas pyarrow
    uvicorn api.main:app --reload --port 8000

    # Example
    curl http://localhost:8000/score/CHK
    curl http://localhost:8000/score/CHK/2020Q1
"""

from pathlib import Path
from typing import Optional
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
MODULE_D     = REPO_ROOT / "Module_D"
MODEL_PATH   = MODULE_D / "lgbm_fusion.txt"
FUSED_PATH   = MODULE_D / "X_fused.parquet"
HEALTH_PATH  = MODULE_D / "health_scores.parquet"
FEAT_PATH    = MODULE_D / "shap_feat_cols.json"
METRICS_PATH = MODULE_D / "metrics.json"

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CrisisNet API",
    description=(
        "Corporate Default Early Warning System — "
        "serves LightGBM fusion model health scores. "
        "E0 259 Data Analytics | IISc Bangalore"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Startup: load model + data ─────────────────────────────────────────────────
_model: Optional[lgb.Booster] = None
_fused: Optional[pd.DataFrame] = None
_scores: Optional[pd.DataFrame] = None
_feat_cols: Optional[list] = None
_metrics: Optional[dict] = None


def _load_all():
    global _model, _fused, _scores, _feat_cols, _metrics

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run: python Module_D/train_fusion.py"
        )
    _model = lgb.Booster(model_file=str(MODEL_PATH))

    if not FUSED_PATH.exists():
        raise RuntimeError(
            f"X_fused not found at {FUSED_PATH}. "
            "Run: python Module_D/build_x_fused.py"
        )
    _fused = pd.read_parquet(FUSED_PATH)
    _fused['year'] = _fused['quarter'].str[:4].astype(int)

    if HEALTH_PATH.exists():
        _scores = pd.read_parquet(HEALTH_PATH)
    else:
        # Compute on-the-fly if health_scores.parquet not pre-built
        with open(FEAT_PATH) as f:
            _feat_cols = json.load(f)
        META = {'ticker', 'quarter', 'Date', 'distress_label', 'year'}
        cols  = [c for c in _feat_cols if c in _fused.columns]
        probs = _model.predict(_fused[cols].values)
        _scores = _fused[['ticker', 'quarter', 'year', 'distress_label']].copy()
        _scores['distress_prob'] = probs
        _scores['health_score']  = 1.0 - probs

    if FEAT_PATH.exists():
        with open(FEAT_PATH) as f:
            _feat_cols = json.load(f)

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)


@app.on_event("startup")
def startup_event():
    try:
        _load_all()
    except RuntimeError as e:
        # Don't crash the server — return errors on individual requests
        print(f"WARNING: {e}")


def _check_ready():
    if _scores is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model outputs not loaded. "
                "Run build_x_fused.py and train_fusion.py first."
            )
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Service health check."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "data_loaded": _scores is not None,
        "n_rows": len(_scores) if _scores is not None else 0,
    }


@app.get("/tickers")
def list_tickers():
    """Return all available tickers."""
    _check_ready()
    return {"tickers": sorted(_scores['ticker'].unique().tolist())}


@app.get("/score/{ticker}")
def get_score(ticker: str):
    """
    Latest quarter health score for a ticker.

    Returns:
        ticker, quarter, health_score, distress_prob, risk_tier
    """
    _check_ready()
    t = ticker.upper()
    sub = _scores[_scores['ticker'] == t]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{t}' not found.")

    row = sub.sort_values('quarter').iloc[-1]
    return _format_row(row)


@app.get("/score/{ticker}/{quarter}")
def get_score_quarter(ticker: str, quarter: str):
    """
    Health score for a specific ticker + quarter (e.g. CHK/2020Q1).
    """
    _check_ready()
    t = ticker.upper()
    sub = _scores[(_scores['ticker'] == t) & (_scores['quarter'] == quarter)]
    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data for ticker='{t}' quarter='{quarter}'."
        )
    return _format_row(sub.iloc[0])


@app.get("/scores")
def get_all_scores(
    ticker:   Optional[str] = None,
    quarter:  Optional[str] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    min_health: Optional[float] = None,
    max_health: Optional[float] = None,
):
    """
    Return scores with optional filters.

    Query params:
        ticker      — filter by ticker (e.g. ?ticker=CHK)
        quarter     — filter by quarter (e.g. ?quarter=2020Q1)
        min_year    — filter from year (e.g. ?min_year=2019)
        max_year    — filter up to year
        min_health  — filter by health score floor
        max_health  — filter by health score ceiling
    """
    _check_ready()
    df = _scores.copy()

    if ticker:
        df = df[df['ticker'] == ticker.upper()]
    if quarter:
        df = df[df['quarter'] == quarter]
    if min_year:
        df = df[df['year'] >= min_year]
    if max_year:
        df = df[df['year'] <= max_year]
    if min_health is not None:
        df = df[df['health_score'] >= min_health]
    if max_health is not None:
        df = df[df['health_score'] <= max_health]

    if df.empty:
        raise HTTPException(status_code=404, detail="No matching records found.")

    return {
        "count": len(df),
        "records": [_format_row(row) for _, row in df.iterrows()]
    }


@app.get("/metrics")
def get_metrics():
    """Return model performance metrics (AUC, Brier scores)."""
    _check_ready()
    if _metrics is None:
        raise HTTPException(status_code=404, detail="metrics.json not found.")
    return _metrics


@app.get("/top_risk")
def top_risk(quarter: Optional[str] = None, n: int = 10):
    """Return the top-N highest distress-probability companies for a given quarter."""
    _check_ready()
    df = _scores.copy()
    if quarter:
        df = df[df['quarter'] == quarter]
    else:
        latest_q = df['quarter'].max()
        df       = df[df['quarter'] == latest_q]

    top = df.nlargest(n, 'distress_prob')
    return {
        "quarter":  df['quarter'].iloc[0] if len(df) > 0 else None,
        "top_risk": [_format_row(row) for _, row in top.iterrows()],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _format_row(row) -> dict:
    dp = float(row['distress_prob'])
    hs = float(row['health_score'])
    if dp < 0.3:
        tier = 'Low'
    elif dp < 0.6:
        tier = 'Medium'
    elif dp < 0.8:
        tier = 'High'
    else:
        tier = 'Critical'

    return {
        "ticker":        row['ticker'],
        "quarter":       row['quarter'],
        "health_score":  round(hs, 4),
        "distress_prob": round(dp, 4),
        "risk_tier":     tier,
        "actual_distress": int(row.get('distress_label', -1)),
    }
