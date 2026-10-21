"""Metrics for imbalanced fraud classification.

Definitions used throughout the project (and in the README):

* F1 / precision / recall are for the fraud class (is_fraud = 1).
* Type II error rate = FN / (FN + TP): the share of frauds the model misses
  (= 1 - recall).
* Missed-fraud share of all transactions = FN / N: the share of all scored
  transactions that are frauds the model let through.
* Type I error rate = FP / (FP + TN): the share of legitimate transactions flagged.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def classification_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, prob)) if y_true.any() else float("nan"),
        "type_ii_error_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "missed_fraud_share_of_all": float(fn / n) if n else 0.0,
        "type_i_error_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "n": int(n),
    }


def recall_target_threshold(y_true, prob, target_recall: float) -> float:
    """Highest threshold whose validation recall is at least ``target_recall``."""
    y_true, prob = np.asarray(y_true).astype(int), np.asarray(prob, dtype=float)
    fraud_scores = np.sort(prob[y_true == 1])
    return float(fraud_scores[int(np.floor((1 - target_recall) * len(fraud_scores)))])


def best_f1_threshold(y_true, prob) -> float:
    """Threshold that maximises fraud-class F1 (tuned on validation data only)."""
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])
