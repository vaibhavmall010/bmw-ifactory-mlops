"""
serving/serve.py

Stage 5 — FastAPI Model Serving

Loads the Production model from MLflow Registry.
Exposes REST endpoints for real-time anomaly detection.

Endpoints:
  GET  /health             — liveness probe
  GET  /model/info         — production model metadata
  POST /predict            — single assembly cycle anomaly check
  POST /predict/batch      — batch cycle anomaly check
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVE] %(message)s")
log = logging.getLogger(__name__)

ACTION_MAP = {
    0: "Cycle within normal parameters. No action required.",
    1: "Anomaly detected. Flag cycle for inspection and notify station supervisor.",
}

app = FastAPI(
    title="BMW iFACTORY Anomaly Detection API",
    description=(
        "Real-time assembly line anomaly detection for BMW iFACTORY production stations. "
        "Loads champion model from MLflow Model Registry."
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_model      = None
_model_info = None
_feature_cols = None


def _load():
    global _model, _model_info, _feature_cols
    try:
        from pipeline.evaluate import load_production_model
        _model, _model_info = load_production_model()

        fs_dir    = os.path.join(PROJECT_ROOT, "data", "feature_store")
        manifests = sorted([f for f in os.listdir(fs_dir) if f.startswith("manifest_")])
        if manifests:
            with open(os.path.join(fs_dir, manifests[-1])) as f:
                _feature_cols = json.load(f)["feature_cols"]
        log.info(f"Production model v{_model_info.get('version')} loaded.")
    except Exception as e:
        log.warning(f"Model not loaded at startup: {e}")


@app.on_event("startup")
async def startup():
    _load()


# ── Schemas ───────────────────────────────────────────────────────────────────

class CycleFeatures(BaseModel):
    features: List[float] = Field(..., description="Feature vector in order from /model/features")
    cycle_id: Optional[str] = None
    station:  Optional[str] = None


class BatchCycles(BaseModel):
    cycles: List[CycleFeatures]


class PredictionResponse(BaseModel):
    cycle_id:       Optional[str]
    station:        Optional[str]
    anomaly:        int
    anomaly_label:  str
    confidence:     float
    recommended_action: str
    model_version:  str
    timestamp:      str


# ── Prediction ────────────────────────────────────────────────────────────────

def _predict_one(cycle: CycleFeatures) -> PredictionResponse:
    if _model is None:
        raise HTTPException(503, "Model not loaded. Run pipeline first.")
    X     = np.array(cycle.features).reshape(1, -1)
    pred  = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0]
    conf  = float(np.max(proba))
    return PredictionResponse(
        cycle_id           = cycle.cycle_id,
        station            = cycle.station,
        anomaly            = pred,
        anomaly_label      = "Anomaly" if pred == 1 else "Normal",
        confidence         = round(conf, 4),
        recommended_action = ACTION_MAP[pred],
        model_version      = str(_model_info.get("version", "unknown")) if _model_info else "unknown",
        timestamp          = datetime.now(timezone.utc).isoformat(),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/model/info")
def model_info():
    if not _model_info:
        raise HTTPException(503, "No model loaded.")
    return _model_info


@app.get("/model/features")
def model_features():
    if not _feature_cols:
        raise HTTPException(503, "Feature manifest not available.")
    return {"feature_cols": _feature_cols, "n_features": len(_feature_cols)}


@app.post("/predict", response_model=PredictionResponse)
def predict(cycle: CycleFeatures):
    return _predict_one(cycle)


@app.post("/predict/batch")
def predict_batch(request: BatchCycles):
    preds = [_predict_one(c) for c in request.cycles]
    n_anomalies = sum(p.anomaly for p in preds)
    return {
        "predictions":  preds,
        "n_cycles":     len(preds),
        "n_anomalies":  n_anomalies,
        "anomaly_rate": round(n_anomalies / len(preds), 4),
        "model_version": str(_model_info.get("version")) if _model_info else "unknown",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
def root():
    return {"service": "BMW iFACTORY Anomaly Detection API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.serve:app", host="0.0.0.0", port=8000, reload=False)
