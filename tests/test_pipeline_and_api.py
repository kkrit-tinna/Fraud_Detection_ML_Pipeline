"""End-to-end smoke test on synthetic data: ETL -> fit -> save -> serve -> monitor."""
import json

import pytest
from fastapi.testclient import TestClient

from fraud_detection import config, etl
from fraud_detection.data import load_table
from fraud_detection.evaluate import best_f1_threshold
from fraud_detection.features import FEATURE_COLUMNS, TARGET
from fraud_detection.model import make_pipeline
from fraud_detection.monitor import psi
from fraud_detection.registry import load_model, save_model

REQUEST = {
    "trans_date_trans_time": "2020-06-21T23:14:25", "amt": 950.0, "category": "shopping_net",
    "merch_lat": 34.1, "merch_long": -81.0, "gender": "F", "dob": "1968-03-19",
    "lat": 33.97, "long": -80.94, "city_pop": 333497,
    "secs_since_last_txn": 420, "card_txn_count_1h": 3, "card_txn_count_24h": 6,
    "card_txn_count_7d": 20, "card_amt_sum_24h": 2900.0, "card_avg_amt": 55.0,
}


@pytest.fixture
def trained_model(tmp_path, raw_csv_dir, monkeypatch):
    processed = tmp_path / "processed"
    manifest = etl.run(raw_csv_dir, processed)
    assert manifest["rows"]["train"] > 0
    train = load_table(processed / "train_features")
    pipe = make_pipeline("xgboost", {"n_estimators": 30, "max_depth": 3, "sampler": "smote", "smote_ratio": 0.5})
    pipe.fit(train[FEATURE_COLUMNS], train[TARGET])
    thr = best_f1_threshold(train[TARGET], pipe.predict_proba(train[FEATURE_COLUMNS])[:, 1])
    model_dir = save_model(pipe, {"model_name": "xgboost", "version": "v0.0.0-test", "decision_threshold": thr,
                                  "features": FEATURE_COLUMNS, "metrics": {}}, tmp_path / "model")
    monkeypatch.setattr(config, "MODEL_DIR", model_dir)
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    return model_dir


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
