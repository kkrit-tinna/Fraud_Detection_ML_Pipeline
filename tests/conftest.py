"""Synthetic data fixtures so tests run in seconds without the Kaggle CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import KNOWN_CATEGORIES


def make_raw(n_cards: int = 40, start: str = "2019-01-01", days: int = 60, seed: int = 0) -> pd.DataFrame:
    """Raw transactions in the Kaggle schema. Fraud comes in late-night bursts of large amounts."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp(start)
    for card in range(n_cards):
        cc = 4000_0000_0000_0000 + card
        lat, lon = rng.uniform(30, 45), rng.uniform(-120, -75)
        dob = t0 - pd.Timedelta(days=int(rng.uniform(20, 80) * 365))
        gender = "M" if card % 2 else "F"
        n = rng.integers(40, 80)
        times = np.sort(rng.uniform(0, days * 86400, n))
        for s in times:
            rows.append((t0 + pd.Timedelta(seconds=float(s)), cc, rng.choice(KNOWN_CATEGORIES),
                         float(np.round(rng.gamma(2, 30), 2)) + 1, gender, lat, lon, 5000, dob, 0))
        if card % 4 == 0:  # fraud burst
            burst_start = t0 + pd.Timedelta(days=float(rng.uniform(5, days - 2))).floor("D") + pd.Timedelta(hours=23)
            for k in range(6):
                rows.append((burst_start + pd.Timedelta(minutes=7 * k), cc, "shopping_net",
                             float(rng.uniform(700, 1200)), gender, lat, lon, 5000, dob, 1))
    df = pd.DataFrame(rows, columns=["trans_date_trans_time", "cc_num", "category", "amt", "gender",
                                     "lat", "long", "city_pop", "dob", "is_fraud"])
    df["merch_lat"] = df["lat"] + rng.normal(0, 0.5, len(df))
    df["merch_long"] = df["long"] + rng.normal(0, 0.5, len(df))
    df["trans_num"] = [f"t{i}" for i in range(len(df))]
    return df.sort_values("trans_date_trans_time").reset_index(drop=True)


@pytest.fixture(scope="session")
def raw_train() -> pd.DataFrame:
    return make_raw(seed=1)


@pytest.fixture(scope="session")
def raw_test() -> pd.DataFrame:
    return make_raw(start="2019-03-02", days=30, seed=2)


@pytest.fixture
def raw_csv_dir(tmp_path, raw_train, raw_test):
    fmt = "%Y-%m-%d %H:%M:%S"
    for name, df in (("fraudTrain.csv", raw_train), ("fraudTest.csv", raw_test)):
        out = df.copy()
        out["trans_date_trans_time"] = out["trans_date_trans_time"].dt.strftime(fmt)
        out["dob"] = out["dob"].dt.strftime("%Y-%m-%d")
        out.to_csv(tmp_path / name)
    return tmp_path
