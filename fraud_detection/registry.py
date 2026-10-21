"""Model versioning: save/load the model artifact locally or from the Hugging Face Hub.

An artifact is a directory with:
  model.joblib   - the fitted imblearn Pipeline (preprocessing + classifier)
  metadata.json  - version, decision threshold, features, metrics, data hashes,
                   git commit and library versions
  README.md      - model card (written by ``publish``)

Each published version is a git tag on the Hub repo (e.g. ``v1.0.0``), so the API
can pin an exact model with ``FRAUD_HF_REVISION``.
"""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from . import config

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def library_versions() -> dict:
    import imblearn
    import numpy
    import pandas
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "imbalanced-learn": imblearn.__version__,
        "xgboost": xgboost.__version__,
    }


def save_model(pipeline, metadata: dict, model_dir: Path | None = None) -> Path:
    import joblib

    model_dir = Path(model_dir or config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / MODEL_FILE, compress=3)
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))
    return model_dir


def download_model(repo_id: str, revision: str = "main", model_dir: Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    model_dir = Path(model_dir or config.MODEL_DIR)
    snapshot_download(
        repo_id=repo_id, revision=revision, local_dir=model_dir,
        allow_patterns=[MODEL_FILE, METADATA_FILE, "README.md"],
    )
    return model_dir


def load_model(model_dir: Path | None = None, repo_id: str | None = None, revision: str | None = None):
    """Load (pipeline, metadata). Falls back to the Hub when no local copy exists."""
    model_dir = Path(model_dir or config.MODEL_DIR)
    repo_id = repo_id if repo_id is not None else config.HF_REPO_ID
    if not (model_dir / MODEL_FILE).exists():
        if not repo_id:
            raise FileNotFoundError(
                f"No model at {model_dir}. Train one (`make train`) or set FRAUD_HF_REPO_ID to pull from the Hub."
            )
        download_model(repo_id, revision or config.HF_REVISION, model_dir)
    import joblib

    pipeline = joblib.load(model_dir / MODEL_FILE)
    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    return pipeline, metadata
