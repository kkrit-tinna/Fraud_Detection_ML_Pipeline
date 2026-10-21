"""FastAPI service for real-time fraud scoring.

    uvicorn fraud_detection.api.app:app --port 8000

The client sends the raw transaction plus the card's recent-history values
(normally read from a feature store). The service computes the same features
as training via ``fraud_detection.features.row_features``, so there is no
training/serving skew. Every prediction is appended to a JSONL log that
``fraud_detection.monitor`` uses for drift monitoring.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import config
from ..data import KNOWN_CATEGORIES
from ..features import FEATURE_COLUMNS, row_features
from ..registry import load_model

Category = Literal[tuple(KNOWN_CATEGORIES)]  # type: ignore[valid-type]


class Transaction(BaseModel):
    # transaction
    trans_date_trans_time: datetime = Field(examples=["2020-06-21T22:14:25"])
    amt: float = Field(gt=0, examples=[912.50])
    category: Category = Field(examples=["shopping_net"])
    merch_lat: float = Field(ge=-90, le=90, examples=[33.98])
    merch_long: float = Field(ge=-180, le=180, examples=[-81.20])
    # cardholder
    gender: Literal["M", "F"] = Field(examples=["F"])
    dob: date = Field(examples=["1968-03-19"])
    lat: float = Field(ge=-90, le=90, examples=[33.97])
    long: float = Field(ge=-180, le=180, examples=[-80.94])
    city_pop: int = Field(ge=0, examples=[333497])
    # card history, precomputed from the card's earlier transactions
    secs_since_last_txn: float = Field(ge=0, description="seconds since this card's previous transaction", examples=[300])
    card_txn_count_1h: int = Field(ge=0, examples=[2])
    card_txn_count_24h: int = Field(ge=0, examples=[9])
    card_txn_count_7d: int = Field(ge=0, examples=[30])
    card_amt_sum_24h: float = Field(ge=0, description="total spend on this card in the previous 24h", examples=[1850.0])
    card_avg_amt: float | None = Field(default=None, gt=0, description="card's average amount before this transaction", examples=[64.2])


class Prediction(BaseModel):
    prediction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    threshold: float
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


STATE: dict = {}
_log_lock = threading.Lock()


def _load() -> None:
    pipeline, metadata = load_model()
    if metadata.get("features") != FEATURE_COLUMNS:
        raise RuntimeError("Model was trained with a different feature set than this code version.")
    # FRAUD_OPERATING_POINT=high_recall trades precision for catching more fraud.
    point = os.getenv("FRAUD_OPERATING_POINT", "f1_optimal")
    points = metadata.get("operating_points") or {}
    if point not in points and point != "f1_optimal":
        raise RuntimeError(f"Unknown operating point {point!r}; available: {sorted(points) or ['f1_optimal']}")
    threshold = points[point]["threshold"] if point in points else metadata["decision_threshold"]
    STATE.update(pipeline=pipeline, metadata=metadata, threshold=float(threshold), operating_point=point)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _load()
    except Exception as exc:  # keep serving /health so the failure is visible
        STATE["load_error"] = str(exc)
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud scoring with a versioned XGBoost/scikit-learn pipeline.",
    version="2.0.0",
    lifespan=lifespan,
)


def risk_level(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "HIGH"
    if prob >= threshold / 2:
        return "MEDIUM"
    return "LOW"


def _log(records: list[dict]) -> None:
    if os.getenv("FRAUD_DISABLE_PREDICTION_LOG"):
        return
    log_path = Path(config.LOG_DIR) / "predictions.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, open(log_path, "a") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def _score(transactions: list[Transaction]) -> list[Prediction]:
    if "pipeline" not in STATE:
        raise HTTPException(503, f"Model not loaded: {STATE.get('load_error', 'unknown error')}")
    t0 = time.perf_counter()
    raw = pd.DataFrame([t.model_dump() for t in transactions])
    feats = row_features(raw)
    probs = STATE["pipeline"].predict_proba(feats[FEATURE_COLUMNS])[:, 1]
    latency = (time.perf_counter() - t0) * 1000 / len(transactions)

    thr, version = STATE["threshold"], STATE["metadata"]["version"]
    preds, logs = [], []
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    for i, p in enumerate(probs):
        pred = Prediction(
            prediction_id=str(uuid.uuid4()), is_fraud=bool(p >= thr), fraud_probability=round(float(p), 6),
            risk_level=risk_level(float(p), thr), threshold=thr, model_version=version, latency_ms=round(latency, 3),
        )
        preds.append(pred)
        logs.append({"logged_at": now, **pred.model_dump(),
                     "features": feats.iloc[i][FEATURE_COLUMNS].to_dict()})
    _log(logs)
    return preds


@app.get("/health")
def health():
    return {"status": "ok" if "pipeline" in STATE else "model_not_loaded",
            "model_loaded": "pipeline" in STATE, "error": STATE.get("load_error")}


@app.get("/model/info")
def model_info():
    if "metadata" not in STATE:
        raise HTTPException(503, "Model not loaded")
    m = STATE["metadata"]
    info = {k: m.get(k) for k in ("model_name", "version", "trained_at", "git_commit",
                                  "features", "metrics", "operating_points")}
    info.update(operating_point=STATE["operating_point"], threshold=STATE["threshold"])
    return info


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    return _score([transaction])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: BatchRequest):
    return _score(batch.transactions)
