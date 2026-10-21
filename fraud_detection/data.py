"""Extract and load steps of the ETL pipeline.

Extract: read the raw CSVs and validate them against the expected schema.
Load:    write the engineered feature tables to Parquet (or CSV if pyarrow is
         unavailable) so training and monitoring reuse one processed copy.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from . import config

REQUIRED_COLUMNS = {
    "trans_date_trans_time": "datetime",
    "cc_num": "int",
    "category": "str",
    "amt": "float",
    "gender": "str",
    "lat": "float",
    "long": "float",
    "city_pop": "int",
    "dob": "datetime",
    "merch_lat": "float",
    "merch_long": "float",
    "is_fraud": "int",
}

KNOWN_CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
    "shopping_net", "shopping_pos", "travel",
]


class SchemaError(ValueError):
    """Raised when an input file does not match the expected schema."""


def validate_schema(df: pd.DataFrame, name: str = "data") -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}")
    if df["amt"].lt(0).any():
        raise SchemaError(f"{name}: negative transaction amounts")
    if not set(df["is_fraud"].unique()) <= {0, 1}:
        raise SchemaError(f"{name}: is_fraud must be 0/1")
    unknown = set(df["category"].unique()) - set(KNOWN_CATEGORIES)
    if unknown:
        raise SchemaError(f"{name}: unknown categories {sorted(unknown)}")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise SchemaError(f"{name}: null values in required columns")


def read_raw(path: Path) -> pd.DataFrame:
    """Read one raw CSV, keeping only the columns the pipeline needs."""
    usecols = list(REQUIRED_COLUMNS) + ["trans_num"]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    validate_schema(df, path.name)
    return df


def extract(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir or config.DATA_DIR)
    train_path, test_path = data_dir / config.TRAIN_FILE, data_dir / config.TEST_FILE
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Download fraudTrain.csv and fraudTest.csv (see README) "
                f"into {data_dir} or set FRAUD_DATA_DIR."
            )
    return read_raw(train_path), read_raw(test_path)


def file_fingerprint(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, recorded in model metadata for data versioning."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def save_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = path.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except ImportError:
        out = path.with_suffix(".csv.gz")
        df.to_csv(out, index=False)
    return out


def load_table(path_stem: Path) -> pd.DataFrame:
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv)):
        p = path_stem.with_suffix(suffix)
        if p.exists():
            return reader(p)
    raise FileNotFoundError(f"No processed table at {path_stem}.*  Run `make etl` first.")
