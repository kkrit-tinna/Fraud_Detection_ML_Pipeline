"""Train, tune and compare Logistic Regression, Random Forest and XGBoost.

    python -m fraud_detection.train [--models xgboost random_forest ...] [--quick]

Protocol (no information from fraudTest.csv is used for any choice):
1. Split fraudTrain.csv by time: the last 15% of transactions are validation.
2. For each model, score each candidate in its search space on validation
   (PR-AUC); candidates are fitted on all frauds + a sample of legitimate
   transactions to keep tuning fast.
3. Refit the best candidate on the full training portion, then pick the
   decision threshold that maximises fraud-class F1 on validation.
4. Score every model once on the untouched fraudTest.csv.
5. Ship the model with the best validation F1: save the artifact, metadata and
   comparison report/figures.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, config
from .data import load_table
from .evaluate import best_f1_threshold, classification_metrics, recall_target_threshold
from .features import FEATURE_COLUMNS, TARGET
from .model import MODEL_NAMES, SEARCH_SPACES, make_pipeline
from .registry import git_commit, library_versions, save_model

HIGH_RECALL_TARGET = 0.97  # validation recall the "high_recall" operating point must reach


def time_split(df: pd.DataFrame, frac: float = config.VALIDATION_FRACTION):
    df = df.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)
    cut = int(len(df) * (1 - frac))
    return df.iloc[:cut], df.iloc[cut:]


def tuning_sample(train: pd.DataFrame, legit_frac: float) -> pd.DataFrame:
    fraud = train[train[TARGET] == 1]
    legit = train[train[TARGET] == 0].sample(frac=legit_frac, random_state=config.RANDOM_STATE)
    return pd.concat([fraud, legit]).sort_index()


def tune(name: str, sample: pd.DataFrame, valid: pd.DataFrame, space: list[dict]) -> tuple[dict, list[dict]]:
    trials = []
    for params in space:
        t0 = time.time()
        pipe = make_pipeline(name, params).fit(sample[FEATURE_COLUMNS], sample[TARGET])
        prob = pipe.predict_proba(valid[FEATURE_COLUMNS])[:, 1]
        score = classification_metrics(valid[TARGET], prob)["pr_auc"]
        trials.append({"params": params, "val_pr_auc": score, "seconds": round(time.time() - t0, 1)})
        print(f"  [{name}] {params} -> val PR-AUC {score:.4f} ({time.time() - t0:.0f}s)")
    best = max(trials, key=lambda t: t["val_pr_auc"])
    return best["params"], trials


def plot_reports(curves: dict, best_name: str, best_metrics: dict, importances: pd.Series | None, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    out.mkdir(parents=True, exist_ok=True)
    labels = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label=labels[name])
        axes[1].plot(rec, prec, label=labels[name])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_pr_curves.png", dpi=150)
    plt.close(fig)

    plot_confusion(best_name, best_metrics, out / "confusion_matrix.png")

    if importances is not None:
        top = importances.sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.barh(top.index, top.values, color="#4C72B0")
        ax.set(title=f"{labels[best_name]} feature importance (top 15)", xlabel="Importance")
        fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=150)
        plt.close(fig)


def plot_confusion(name: str, m: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    shade = np.log10(cm + 1)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(shade, cmap="Blues", vmin=0, vmax=shade.max() * 1.15)
    for (i, j), v in np.ndenumerate(cm):
        dark = shade[i, j] > shade.max() * 0.6
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12, color="white" if dark else "black")
    ax.set_xticks([0, 1], ["Legit", "Fraud"])
    ax.set_yticks([0, 1], ["Legit", "Fraud"])
    ax.set(xlabel="Predicted", ylabel="Actual",
           title=f"{labels[name]} confusion matrix\n(test set, threshold {m['threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def feature_importances(pipe) -> pd.Series | None:
    clf = pipe.named_steps["clf"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=names)
    if hasattr(clf, "coef_"):
        return pd.Series(np.abs(clf.coef_[0]), index=names)
    return None


def run(models=MODEL_NAMES, quick: bool = False, processed_dir: Path | None = None,
        model_dir: Path | None = None, reports_dir: Path | None = None) -> dict:
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    t_start = time.time()

    train_all = load_table(processed_dir / "train_features")
    test = load_table(processed_dir / "test_features")
    train_all["trans_date_trans_time"] = pd.to_datetime(train_all["trans_date_trans_time"])
    train, valid = time_split(train_all)
    sample = tuning_sample(train, legit_frac=0.1 if quick else 0.25)
    print(f"train {len(train):,} | valid {len(valid):,} | test {len(test):,} | tuning sample {len(sample):,}")

    results, fitted, curves = {}, {}, {}
    for name in models:
        print(f"\n=== {name} ===")
        space = SEARCH_SPACES[name][:1] if quick else SEARCH_SPACES[name]
        best_params, trials = tune(name, sample, valid, space)
        t0 = time.time()
        fit_data = sample if quick else train
        pipe = make_pipeline(name, best_params).fit(fit_data[FEATURE_COLUMNS], fit_data[TARGET])
        fit_seconds = time.time() - t0

        p_valid = pipe.predict_proba(valid[FEATURE_COLUMNS])[:, 1]
        threshold = best_f1_threshold(valid[TARGET], p_valid)
        p_test = pipe.predict_proba(test[FEATURE_COLUMNS])[:, 1]

        results[name] = {
            "best_params": best_params,
            "tuning_trials": trials,
            "fit_seconds": round(fit_seconds, 1),
            "validation": classification_metrics(valid[TARGET], p_valid, threshold),
            "test": classification_metrics(test[TARGET], p_test, threshold),
            "test_at_0.5": classification_metrics(test[TARGET], p_test, 0.5),
        }
        hr_thr = recall_target_threshold(valid[TARGET], p_valid, HIGH_RECALL_TARGET)
        results[name]["operating_points"] = {
            "f1_optimal": {"threshold": threshold, "rule": "maximise F1 on validation",
                           "validation": results[name]["validation"], "test": results[name]["test"]},
            "high_recall": {"threshold": hr_thr,
                            "rule": f"highest threshold with validation recall >= {HIGH_RECALL_TARGET}",
                            "validation": classification_metrics(valid[TARGET], p_valid, hr_thr),
                            "test": classification_metrics(test[TARGET], p_test, hr_thr)},
        }
        fitted[name] = pipe
        curves[name] = (test[TARGET].to_numpy(), p_test)
        v, t = results[name]["validation"], results[name]["test"]
        print(f"  threshold {threshold:.3f} | val F1 {v['f1']:.4f} | test F1 {t['f1']:.4f} "
              f"P {t['precision']:.4f} R {t['recall']:.4f} AUC {t['roc_auc']:.4f} "
              f"TypeII {t['type_ii_error_rate']:.2%}")

    best_name = max(results, key=lambda n: results[n]["validation"]["f1"])
    best = results[best_name]
    print(f"\nSelected model: {best_name} (validation F1 {best['validation']['f1']:.4f})")

    metadata = {
        "model_name": best_name,
        "version": f"v{__version__}",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "decision_threshold": best["validation"]["threshold"],
        "operating_points": {k: {"threshold": v["threshold"], "rule": v["rule"], "test": v["test"]}
                             for k, v in best["operating_points"].items()},
        "params": best["best_params"],
        "features": FEATURE_COLUMNS,
        "metrics": {"validation": best["validation"], "test": best["test"]},
        "data": json.loads((processed_dir / "manifest.json").read_text()) if (processed_dir / "manifest.json").exists() else None,
        "training_rows": int(len(sample if quick else train)),
        "libraries": library_versions(),
        "quick_run": quick,
    }
    saved = save_model(fitted[best_name], metadata, model_dir)
    print(f"Saved model artifact to {saved}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "model_comparison.json").write_text(json.dumps(
        {"selected_model": best_name, "results": results}, indent=2, default=str))
    rows = []
    for name, r in results.items():
        t = r["test"]
        rows.append({"model": name, "threshold": round(t["threshold"], 3), "f1": t["f1"], "precision": t["precision"],
                     "recall": t["recall"], "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"],
                     "type_ii_error_rate": t["type_ii_error_rate"],
                     "missed_fraud_share_of_all": t["missed_fraud_share_of_all"],
                     "type_i_error_rate": t["type_i_error_rate"], "fn": t["fn"], "fp": t["fp"]})
    table = pd.DataFrame(rows)
    hr = best["operating_points"]["high_recall"]["test"]
    print(f"\n{best_name} high-recall operating point (threshold {hr['threshold']:.4f}): test F1 {hr['f1']:.4f} "
          f"P {hr['precision']:.4f} R {hr['recall']:.4f} TypeII {hr['type_ii_error_rate']:.2%} "
          f"TypeI {hr['type_i_error_rate']:.3%}")
    table.to_csv(reports_dir / "model_comparison.csv", index=False)
    plot_reports(curves, best_name, best["test"], feature_importances(fitted[best_name]), reports_dir / "figures")
    print(table.round(4).to_string(index=False))
    print(f"\nTotal training time {time.time() - t_start:.0f}s")
    return {"selected_model": best_name, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--quick", action="store_true", help="one candidate per model on a small sample (CI)")
    args = parser.parse_args()
    run(args.models, quick=args.quick)


if __name__ == "__main__":
    main()
