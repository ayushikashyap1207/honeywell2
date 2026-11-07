import numpy as np

from evaluation import _top_k_metrics


def test_top_k_metrics_uses_the_alert_budget():
    labels = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.99, 0.3, 0.98, 0.4, 0.5, 0.6, 0.7, 0.8])

    metrics = _top_k_metrics(labels, scores, fraction=0.2)

    assert metrics["k"] == 2
    assert metrics["precision_at_1"] == 1.0
    assert metrics["recall_at_1"] == 1.0
    assert metrics["fpr_at_1"] == 0.0
