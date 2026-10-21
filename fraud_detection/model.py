"""Model definitions.

Each model is one imblearn ``Pipeline``: preprocessing -> optional SMOTE ->
classifier. SMOTE lives inside the pipeline, so it only ever resamples the data
a model is being fitted on; validation and test data are scored untouched.
The fitted pipeline is the single artifact that is saved, published and served.
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .data import KNOWN_CATEGORIES
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler() if scale_numeric else "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(categories=[KNOWN_CATEGORIES], handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def smote(ratio: float = 0.1) -> SMOTE:
    """Oversample frauds to ``ratio`` x the number of legitimate transactions."""
    return SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=RANDOM_STATE)


def make_pipeline(name: str, params: dict | None = None) -> Pipeline:
    params = dict(params or {})
    sampler = params.pop("sampler", "smote")
    sampler_step = smote(params.pop("smote_ratio", 0.1)) if sampler == "smote" else "passthrough"

    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE, **params)
        prep = make_preprocessor(scale_numeric=True)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, **params)
        prep = make_preprocessor(scale_numeric=False)
    elif name == "xgboost":
        clf = XGBClassifier(
            tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=RANDOM_STATE, **params
        )
        prep = make_preprocessor(scale_numeric=False)
    else:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    return Pipeline([("prep", prep), ("sampler", sampler_step), ("clf", clf)])


# Small, deliberate search spaces: each candidate is scored on the time-based
# validation split (PR-AUC), then the winner is refitted on the full training file.
SEARCH_SPACES: dict[str, list[dict]] = {
    "logistic_regression": [
        {"C": 0.1},
        {"C": 1.0},
        {"C": 1.0, "sampler": "none", "class_weight": "balanced"},
    ],
    "random_forest": [
        {"n_estimators": 150, "max_depth": 16, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 150, "max_depth": 24, "min_samples_leaf": 3, "max_features": "sqrt"},
    ],
    "xgboost": [
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
         "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0, "sampler": "none"},
        {"n_estimators": 600, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8,
         "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 5.0, "sampler": "none"},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
         "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0, "sampler": "smote"},
    ],
}
