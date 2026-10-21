"""Model performance and drift monitoring.

    python -m fraud_detection.monitor            # replay the test period month by month
    python -m fraud_detection.monitor --log      # also summarise the API prediction log

Replay: scores fraudTest.csv (Jun-Dec 2020, after the training period) one
month at a time, as if labels arrived monthly. For each month it reports F1,
precision, recall and Type II error at the shipped threshold, plus the
Population Stability Index (PSI) of every numeric feature against the training
distribution. Months where recall falls more than 5 points below validation or
any feature PSI exceeds 0.2 are flagged.

Prediction log: summarises what the live API has scored (volume, alert rate,
score distribution and feature PSI against training).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import load_table
from .evaluate import classification_metrics
from .features import FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET
from .registry import load_model

PSI_ALERT = 0.2
RECALL_DROP_ALERT = 0.05


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index of ``current`` against ``reference`` (quantile bins)."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature: compare category shares instead
        ref_vals, cur_vals = pd.Series(reference).round(6), pd.Series(current).round(6)
        ref_p = ref_vals.value_counts(normalize=True)
        cur_p = cur_vals.value_counts(normalize=True).reindex(ref_p.index, fill_value=0)
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        ref_p = pd.Series(np.histogram(reference, edges)[0] / len(reference))
        cur_p = pd.Series(np.histogram(current, edges)[0] / max(len(current), 1))
    ref_p, cur_p = ref_p.clip(lower=1e-4), cur_p.clip(lower=1e-4)
    return float(((cur_p - ref_p) * np.log(cur_p / ref_p)).sum())


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    return {c: round(psi(reference[c], current[c]), 4) for c in NUMERIC_FEATURES}


def replay(model_dir=None, processed_dir=None, out_dir=None) -> pd.DataFrame:
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    out_dir = Path(out_dir or (config.REPORTS_DIR / "monitoring"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline, meta = load_model(model_dir)
    thr = float(meta["decision_threshold"])
    val_recall = meta["metrics"]["validation"]["recall"]

    reference = load_table(processed_dir / "train_features")
    test = load_table(processed_dir / "test_features")
    test["month"] = pd.to_datetime(test["trans_date_trans_time"]).dt.to_period("M").astype(str)
    test["score"] = pipeline.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    rows, drift = [], {}
    for month, g in test.groupby("month"):
        m = classification_metrics(g[TARGET], g["score"], thr)
        d = feature_psi(reference, g)
        drift[month] = d
        worst = max(d, key=d.get)
        alerts = []
        if m["recall"] < val_recall - RECALL_DROP_ALERT:
            alerts.append(f"recall {m['recall']:.3f} < validation {val_recall:.3f} - {RECALL_DROP_ALERT}")
        if d[worst] > PSI_ALERT:
            alerts.append(f"PSI {worst}={d[worst]:.2f}")
        rows.append({"month": month, "transactions": m["n"], "frauds": m["tp"] + m["fn"],
                     "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                     "type_ii_error_rate": m["type_ii_error_rate"], "max_feature_psi": d[worst],
                     "max_psi_feature": worst, "alerts": "; ".join(alerts) or "none"})
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "monthly_performance.csv", index=False)
    pd.DataFrame(drift).T.to_csv(out_dir / "feature_psi_by_month.csv")
    _plot(report, val_recall, out_dir / "monthly_performance.png")
    print(report.round(4).to_string(index=False))
    return report


def _plot(report: pd.DataFrame, val_recall: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col, label in (("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")):
        ax.plot(report["month"], report[col], marker="o", label=label)
    ax.axhline(val_recall - RECALL_DROP_ALERT, color="red", ls="--", lw=1, label="Recall alert line")
    ax.set(ylim=(0.5, 1.02), title="Monthly model performance on the test period", ylabel="Score")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize_log(log_path=None, processed_dir=None) -> dict:
    log_path = Path(log_path or (Path(config.LOG_DIR) / "predictions.jsonl"))
    if not log_path.exists():
        print(f"No prediction log at {log_path}")
        return {}
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(records)
    feats = pd.DataFrame(list(df["features"]))
    summary = {
        "predictions": len(df),
        "first": df["logged_at"].min(), "last": df["logged_at"].max(),
        "alert_rate": float(df["is_fraud"].mean()),
        "mean_score": float(df["fraud_probability"].mean()),
        "model_versions": df["model_version"].value_counts().to_dict(),
        "p50_latency_ms": float(df["latency_ms"].median()),
    }
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    try:
        reference = load_table(processed_dir / "train_features")
        summary["feature_psi"] = feature_psi(reference, feats) if len(feats) >= 50 else "need >= 50 predictions"
    except FileNotFoundError:
        summary["feature_psi"] = "reference data not available"
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", action="store_true", help="also summarise the API prediction log")
    args = parser.parse_args()
    replay()
    if args.log:
        summarize_log()


if __name__ == "__main__":
    main()
