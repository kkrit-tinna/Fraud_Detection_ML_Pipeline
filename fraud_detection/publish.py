"""Publish the trained model to the Hugging Face Hub as a tagged version.

    huggingface-cli login            # once
    python -m fraud_detection.publish --repo-id <user>/credit-card-fraud-xgboost

Uploads model.joblib, metadata.json and a generated model card, then tags the
commit with the model version (e.g. v2.0.0). Pull a pinned version with
FRAUD_HF_REPO_ID=<repo> FRAUD_HF_REVISION=v2.0.0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .registry import METADATA_FILE, MODEL_FILE

GITHUB_URL = "https://github.com/kkrit-tinna/Fraud_Detection_ML_Pipeline"


def operating_point_rows(meta: dict) -> str:
    rows = []
    for name, op in (meta.get("operating_points") or {}).items():
        m = op["test"]
        rows.append(f"| {name} ({op['rule']}) | {op['threshold']:.3f} | {m['f1']:.4f} | {m['precision']:.4f} | "
                    f"{m['recall']:.4f} | {m['type_ii_error_rate']:.2%} | {m['type_i_error_rate']:.3%} |")
    return "\n".join(rows)


def model_card(meta: dict, repo_id: str) -> str:
    t, v = meta["metrics"]["test"], meta["metrics"]["validation"]
    names = {"xgboost": "XGBoost", "random_forest": "Random Forest", "logistic_regression": "Logistic Regression"}
    return f"""---
license: mit
library_name: sklearn
tags:
- tabular-classification
- fraud-detection
- xgboost
- imbalanced-learn
metrics:
- f1
- precision
- recall
- roc_auc
---

# Credit card fraud detection ({names.get(meta['model_name'], meta['model_name'])}, {meta['version']})

Binary classifier that scores a card transaction's probability of fraud. The artifact is a
scikit-learn / imbalanced-learn `Pipeline` (preprocessing + classifier) saved with joblib.
Training, evaluation and serving code: {GITHUB_URL}

## Results on the held-out test set

Test set: `fraudTest.csv` (555,719 transactions, Jun-Dec 2020, 0.39% fraud). It was never used for training,
tuning or threshold selection. Decision threshold: **{meta['decision_threshold']:.3f}** (tuned for F1 on a
time-based validation split of the training data).

| Metric | Test | Validation |
|---|---|---|
| F1 (fraud class) | {t['f1']:.4f} | {v['f1']:.4f} |
| Precision | {t['precision']:.4f} | {v['precision']:.4f} |
| Recall | {t['recall']:.4f} | {v['recall']:.4f} |
| ROC-AUC | {t['roc_auc']:.4f} | {v['roc_auc']:.4f} |
| PR-AUC | {t['pr_auc']:.4f} | {v['pr_auc']:.4f} |
| Type II error rate (missed frauds / all frauds) | {t['type_ii_error_rate']:.2%} | {v['type_ii_error_rate']:.2%} |
| Missed frauds / all transactions | {t['missed_fraud_share_of_all']:.3%} | {v['missed_fraud_share_of_all']:.3%} |
| Type I error rate (false alarms / legitimate) | {t['type_i_error_rate']:.3%} | {v['type_i_error_rate']:.3%} |

Confusion matrix (test): TP {t['tp']:,} | FP {t['fp']:,} | FN {t['fn']:,} | TN {t['tn']:,}

### Operating points

| Operating point | Threshold | Test F1 | Precision | Recall | Type II error rate | Type I error rate |
|---|---|---|---|---|---|---|
{operating_point_rows(meta)}

## Intended use and limitations

* Trained on the public **simulated** Sparkov credit card transactions dataset (Kaggle:
  `kartik2112/fraud-detection`), not on real bank data. Use it for learning and demos, not for production
  decisions.
* Inputs include per-card history values (transaction counts in the last 1h/24h/7d, spend in the last 24h,
  time since the previous transaction, the card's prior average amount) that the caller must compute.
* The pickled pipeline needs the library versions listed in `metadata.json`. Only load joblib files from
  sources you trust.

## Usage

```python
from huggingface_hub import snapshot_download
import joblib, json

path = snapshot_download("{repo_id}", revision="{meta['version']}")
pipeline = joblib.load(f"{{path}}/{MODEL_FILE}")
meta = json.load(open(f"{{path}}/{METADATA_FILE}"))
# features: build them with fraud_detection.features.row_features (see the GitHub repo)
proba = pipeline.predict_proba(features[meta["features"]])[:, 1]
is_fraud = proba >= meta["decision_threshold"]
```

## Training details

* Features: {len(meta['features'])} ({', '.join(meta['features'])})
* Hyperparameters: `{json.dumps(meta['params'])}`
* Training rows: {meta['training_rows']:,}, trained {meta['trained_at']}, code commit `{meta.get('git_commit')}`
* Libraries: {', '.join(f'{k} {v}' for k, v in meta['libraries'].items())}
"""


def publish(repo_id: str, model_dir: Path | None = None, private: bool = False) -> str:
    from huggingface_hub import HfApi

    model_dir = Path(model_dir or config.MODEL_DIR)
    meta = json.loads((model_dir / METADATA_FILE).read_text())
    if meta.get("quick_run"):
        raise SystemExit("Refusing to publish a --quick model. Run the full `make train` first.")
    (model_dir / "README.md").write_text(model_card(meta, repo_id))

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id, folder_path=str(model_dir),
        allow_patterns=[MODEL_FILE, METADATA_FILE, "README.md"],
        commit_message=f"Upload {meta['model_name']} {meta['version']}",
    )
    tags = {t.name for t in api.list_repo_refs(repo_id).tags}
    if meta["version"] in tags:
        print(f"Tag {meta['version']} already exists; bump fraud_detection.__version__ for a new version.")
    else:
        api.create_tag(repo_id, tag=meta["version"], revision=commit.oid,
                       tag_message=f"test F1 {meta['metrics']['test']['f1']:.4f}")
    url = f"https://huggingface.co/{repo_id}"
    print(f"Published {meta['version']} to {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=config.HF_REPO_ID or None, required=not config.HF_REPO_ID)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    publish(args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
