import numpy as np
import pandas as pd

from fraud_detection.features import FEATURE_COLUMNS, build_features, card_history, row_features


def _txns(times, amts, card=1):
    return pd.DataFrame({
        "trans_date_trans_time": pd.to_datetime(times),
        "cc_num": card,
        "amt": amts,
    })


def test_card_history_uses_only_earlier_transactions():
    df = _txns(["2020-01-01 10:00", "2020-01-01 10:30", "2020-01-01 12:00", "2020-01-03 12:00"],
               [10.0, 20.0, 30.0, 40.0])
    h = card_history(df)
    assert h["card_txn_count_1h"].tolist() == [0, 1, 0, 0]
    assert h["card_txn_count_24h"].tolist() == [0, 1, 2, 0]
    assert h["card_txn_count_7d"].tolist() == [0, 1, 2, 3]
    assert h["card_amt_sum_24h"].tolist() == [0.0, 10.0, 30.0, 0.0]
    assert h["secs_since_last_txn"].tolist()[1:] == [1800.0, 5400.0, 172800.0]
    assert np.isnan(h["card_avg_amt"].iloc[0])
    assert h["card_avg_amt"].tolist()[1:] == [10.0, 15.0, 20.0]


def test_card_history_keeps_cards_separate_and_input_order():
    a = _txns(["2020-01-01 10:00", "2020-01-01 10:10"], [5.0, 7.0], card=1)
    b = _txns(["2020-01-01 10:05"], [100.0], card=2)
    df = pd.concat([a, b], ignore_index=True).sample(frac=1, random_state=0)  # shuffled input
    h = card_history(df).loc[df.index]
    by_amt = dict(zip(df["amt"], h["card_txn_count_1h"]))
    assert by_amt == {5.0: 0, 7.0: 1, 100.0: 0}


def test_build_features_shapes_and_no_nans(raw_train, raw_test):
    train, test = build_features(raw_train, raw_test)
    assert len(train) == len(raw_train) and len(test) == len(raw_test)
    assert list(train.columns[: len(FEATURE_COLUMNS)]) == FEATURE_COLUMNS
    assert not train[FEATURE_COLUMNS].isna().any().any()
    # test-period history continues from the training period
    assert test["card_txn_count_7d"].iloc[:50].max() > 0


def test_row_features_single_request_matches_batch(raw_train):
    one = raw_train.iloc[[5]].assign(secs_since_last_txn=60, card_txn_count_1h=1, card_txn_count_24h=3,
                                     card_txn_count_7d=10, card_amt_sum_24h=120.0, card_avg_amt=None)
    feats = row_features(one)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert feats["amt_to_card_avg"].iloc[0] == 1.0  # no prior average -> neutral ratio
