"""Transform step: feature engineering.

Two groups of features:

* Row features come from the transaction itself: amount, merchant category,
  time of day (cyclically encoded), cardholder age and gender, city
  population, and cardholder-to-merchant distance.
* Card-history features summarise the same card's *earlier* transactions:
  velocity (counts in the last 1h / 24h / 7d), recent spend, time since the
  previous transaction, and how unusual this amount is for the card. They use
  only transactions strictly before the current one, so there is no look-ahead
  leakage. In production these come from a feature store; the API accepts
  them precomputed (see ``HISTORY_INPUTS``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NIGHT_HOURS = (22, 23, 0, 1, 2, 3)
NO_HISTORY_SECONDS = 30 * 24 * 3600  # used when a card has no previous transaction

# Raw per-card history values a client sends to the API.
HISTORY_INPUTS = [
    "secs_since_last_txn",
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_txn_count_7d",
    "card_amt_sum_24h",
    "card_avg_amt",
]

NUMERIC_FEATURES = [
    "amt",
    "log_amt",
    "log_city_pop",
    "age",
    "gender_male",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_night",
    "distance_km",
    "log_secs_since_last_txn",
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_txn_count_7d",
    "log_card_amt_sum_24h",
    "amt_to_card_avg",
]
CATEGORICAL_FEATURES = ["category"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_fraud"


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def card_history(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw history values (``HISTORY_INPUTS``) for every transaction.

    Vectorised: transactions are sorted by (card, time) and window counts/sums
    come from binary search over a single combined (card, time) key.
    """
    # Unit-safe conversion to epoch seconds (pandas 2 uses ns, pandas 3 may use s/us).
    ts = pd.to_datetime(df["trans_date_trans_time"]).to_numpy().astype("datetime64[s]").astype(np.int64)
    card = pd.factorize(df["cc_num"])[0].astype(np.int64)
    order = np.lexsort((ts, card))
    ts_s, card_s = ts[order], card[order]
    amt_s = df["amt"].to_numpy(dtype=float)[order]

    key = card_s * 10**10 + ts_s  # cards never overlap, time stays ordered within a card
    idx = np.arange(len(key))
    cum_amt = np.concatenate([[0.0], np.cumsum(amt_s)])

    out = {}
    for name, window in (("1h", 3600), ("24h", 86400), ("7d", 7 * 86400)):
        left = np.searchsorted(key, key - window, side="left")
        out[f"card_txn_count_{name}"] = idx - left
        if name == "24h":
            out["card_amt_sum_24h"] = cum_amt[idx] - cum_amt[left]

    first_of_card = np.r_[True, card_s[1:] != card_s[:-1]]
    gap = np.r_[0, np.diff(ts_s)].astype(float)
    gap[first_of_card] = NO_HISTORY_SECONDS
    out["secs_since_last_txn"] = gap

    card_start = np.maximum.accumulate(np.where(first_of_card, idx, 0))
    n_prior = idx - card_start
    prior_sum = cum_amt[idx] - cum_amt[card_start]
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(n_prior > 0, prior_sum / np.maximum(n_prior, 1), np.nan)
    out["card_avg_amt"] = avg

    hist = pd.DataFrame(out)
    result = pd.DataFrame(index=df.index, columns=HISTORY_INPUTS, dtype=float)
    result.iloc[order] = hist[HISTORY_INPUTS].to_numpy(dtype=float)
    return result


def row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derived from a single transaction plus its precomputed history."""
    t = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])
    out = pd.DataFrame(index=df.index)
    out["amt"] = df["amt"].astype(float)
    out["log_amt"] = np.log1p(out["amt"])
    out["log_city_pop"] = np.log1p(df["city_pop"].astype(float))
    out["age"] = (t - dob).dt.days / 365.25
    out["gender_male"] = (df["gender"] == "M").astype(int)
    hour, dow = t.dt.hour, t.dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(int)
    out["is_night"] = hour.isin(NIGHT_HOURS).astype(int)
    out["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    out["log_secs_since_last_txn"] = np.log1p(df["secs_since_last_txn"].astype(float).clip(lower=0))
    for c in ("card_txn_count_1h", "card_txn_count_24h", "card_txn_count_7d"):
        out[c] = df[c].astype(float)
    out["log_card_amt_sum_24h"] = np.log1p(df["card_amt_sum_24h"].astype(float).clip(lower=0))
    avg = df["card_avg_amt"].astype(float)
    out["amt_to_card_avg"] = (out["amt"] / avg).where(avg > 0, 1.0).fillna(1.0)

    out["category"] = df["category"].astype(str)
    return out[FEATURE_COLUMNS]


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Engineer features for the train and test files.

    History is computed over the two files combined in time order (test follows
    train chronologically), so a test transaction can see the same card's
    earlier training-period transactions, exactly as it would in production.
    Labels are never used to build features.
    """
    combined = pd.concat([train_raw.assign(_split="train"), test_raw.assign(_split="test")], ignore_index=True)
    combined = pd.concat([combined, card_history(combined)], axis=1)
    feats = row_features(combined)
    feats[TARGET] = combined[TARGET].astype(int).to_numpy()
    feats["trans_date_trans_time"] = combined["trans_date_trans_time"].to_numpy()
    feats["_split"] = combined["_split"].to_numpy()
    train = feats[feats["_split"] == "train"].drop(columns="_split").reset_index(drop=True)
    test = feats[feats["_split"] == "test"].drop(columns="_split").reset_index(drop=True)
    return train, test
