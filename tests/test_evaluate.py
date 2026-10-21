import pytest

from fraud_detection.evaluate import best_f1_threshold, classification_metrics


def test_error_rate_definitions():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1]
    m = classification_metrics(y, p, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["type_ii_error_rate"] == pytest.approx(1 / 4)      # missed / all frauds
    assert m["missed_fraud_share_of_all"] == pytest.approx(1 / 10)
    assert m["type_i_error_rate"] == pytest.approx(1 / 6)
    assert m["recall"] == pytest.approx(1 - m["type_ii_error_rate"])


def test_best_threshold_separates_perfectly():
    y = [0, 0, 0, 1, 1]
    p = [0.1, 0.2, 0.3, 0.8, 0.9]
    thr = best_f1_threshold(y, p)
    assert classification_metrics(y, p, thr)["f1"] == 1.0
