"""Project paths and constants.

Every path can be overridden with an environment variable so the same code runs
from a fresh clone, from CI (on a small sample) and inside the API container.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("FRAUD_DATA_DIR", REPO_ROOT / "data"))
PROCESSED_DIR = Path(os.getenv("FRAUD_PROCESSED_DIR", REPO_ROOT / "artifacts" / "processed"))
MODEL_DIR = Path(os.getenv("FRAUD_MODEL_DIR", REPO_ROOT / "artifacts" / "model"))
REPORTS_DIR = Path(os.getenv("FRAUD_REPORTS_DIR", REPO_ROOT / "reports"))
LOG_DIR = Path(os.getenv("FRAUD_LOG_DIR", REPO_ROOT / "artifacts" / "logs"))

TRAIN_FILE = "fraudTrain.csv"
TEST_FILE = "fraudTest.csv"

RANDOM_STATE = 42

# Share of the (chronologically last) training data held out for early stopping
# and decision-threshold tuning. The official test file is never used for tuning.
VALIDATION_FRACTION = 0.15

# Hugging Face Hub repo the model is published to / pulled from.
HF_REPO_ID = os.getenv("FRAUD_HF_REPO_ID", "")
HF_REVISION = os.getenv("FRAUD_HF_REVISION", "main")
