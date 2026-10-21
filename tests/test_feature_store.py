"""The online feature store must reproduce the batch features used in training."""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from fraud_detection.feature_store import CardHistoryStore
from fraud_detection.features import HISTORY_INPUTS, card_history
from fraud_detection.replay import replay


def _stream_history(store: CardHistoryStore, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        rows.append(store.get(r.cc_num, r.trans_date_trans_time))
        store.update(r.cc_num, r.trans_date_trans_time, r.amt)
    out = pd.DataFrame(rows, index=df.index)
    out["card_avg_amt"] = out["card_avg_amt"].astype(float)  # None -> NaN, as in the batch version
    return out[HISTORY_INPUTS]


def _ordered(raw_train, raw_test) -> pd.DataFrame:
    df = pd.concat([raw_train, raw_test], ignore_index=True)
    return df.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)


def test_streaming_matches_batch_from_cold_start(raw_train, raw_test):
    df = _ordered(raw_train, raw_test)
    expected = card_history(df)
    got = _stream_history(CardHistoryStore(), df)
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_streaming_matches_batch_after_warm_up(raw_train, raw_test):
    df = _ordered(raw_train, raw_test)
    expected = card_history(df)
    cut = df["trans_date_trans_time"] >= raw_test["trans_date_trans_time"].min()
    store = CardHistoryStore()
    store.warm_up(df[~cut])
    got = _stream_history(store, df[cut])
    pd.testing.assert_frame_equal(got, expected[cut], check_dtype=False)


def test_new_card_has_no_history():
    h = CardHistoryStore().get(123, "2020-01-01 00:00:00")
    assert h["card_txn_count_7d"] == 0 and h["card_avg_amt"] is None


def test_replay_through_api(trained_model, raw_test):
    from fraud_detection.api.app import app

    rows = raw_test.head(120).to_dict("records")
    with TestClient(app) as client:
        def score(payloads):
            if len(payloads) == 1:
                r = client.post("/predict", json=payloads[0])
                assert r.status_code == 200, r.text
                return [r.json()]
            r = client.post("/predict/batch", json={"transactions": payloads})
            assert r.status_code == 200, r.text
            return r.json()

        single = replay(rows, CardHistoryStore(), score, batch_size=1, show_alerts=False, report_every=0)
        batched = replay(rows, CardHistoryStore(), score, batch_size=50, show_alerts=False, report_every=0)

    assert single["scored"] == batched["scored"] == 120
    p1 = [r["fraud_probability"] for r in single["results"]]
    p2 = [r["fraud_probability"] for r in batched["results"]]
    assert np.allclose(p1, p2)  # batching must not change any score
    assert single["tp"] + single["fn"] == int(raw_test.head(120)["is_fraud"].sum())
