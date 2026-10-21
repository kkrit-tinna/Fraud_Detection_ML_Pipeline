"""End-to-end smoke test on synthetic data: ETL -> fit -> save -> serve -> monitor."""
import json

import pytest
from fastapi.testclient import TestClient

from fraud_detection.features import FEATURE_COLUMNS
from fraud_detection.monitor import psi
from fraud_detection.registry import load_model

REQUEST = {
    "trans_date_trans_time": "2020-06-21T23:14:25", "amt": 950.0, "category": "shopping_net",
    "merch_lat": 34.1, "merch_long": -81.0, "gender": "F", "dob": "1968-03-19",
    "lat": 33.97, "long": -80.94, "city_pop": 333497,
    "secs_since_last_txn": 420, "card_txn_count_1h": 3, "card_txn_count_24h": 6,
    "card_txn_count_7d": 20, "card_amt_sum_24h": 2900.0, "card_avg_amt": 55.0,
}


def test_saved_model_roundtrip(trained_model):
    pipe, meta = load_model(trained_model)
    assert meta["features"] == FEATURE_COLUMNS
    assert hasattr(pipe, "predict_proba")


def test_api_predict_and_logging(trained_model, tmp_path):
    from fraud_detection.api.app import app

    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert client.get("/model/info").json()["version"] == "v0.0.0-test"

        r = client.post("/predict", json=REQUEST)
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

        r = client.post("/predict/batch", json={"transactions": [REQUEST, {**REQUEST, "amt": 12.0}]})
        assert r.status_code == 200 and len(r.json()) == 2

        bad = client.post("/predict", json={**REQUEST, "category": "not_a_category"})
        assert bad.status_code == 422

    lines = (tmp_path / "logs" / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == 3 and set(json.loads(lines[0])["features"]) == set(FEATURE_COLUMNS)


def test_psi_detects_shift():
    import numpy as np

    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert psi(ref, rng.normal(1.5, 1, 5000)) > 0.2
