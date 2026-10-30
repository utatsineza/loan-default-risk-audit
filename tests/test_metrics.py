"""Unit tests: evaluation metrics."""
import numpy as np
import pytest

from loan_audit.metrics import EvaluationMetrics


def test_perfect_predictions_score_one():
    y = np.array([0, 0, 1, 1])
    m = EvaluationMetrics.from_predictions("m", y, np.array([0.1, 0.2, 0.9, 0.8]))
    assert m.roc_auc == 1.0 and m.f1 == 1.0 and m.pr_auc == pytest.approx(1.0)


def test_rejects_single_class_labels():
    with pytest.raises(ValueError, match="both classes"):
        EvaluationMetrics.from_predictions("m", [0, 0, 0], [0.1, 0.2, 0.3])


def test_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        EvaluationMetrics.from_predictions("m", [0, 1], [0.1])


def test_delta_vs_reports_differences():
    y = np.array([0, 0, 1, 1])
    good = EvaluationMetrics.from_predictions("good", y, np.array([0.1, 0.2, 0.9, 0.8]))
    bad = EvaluationMetrics.from_predictions("bad", y, np.array([0.6, 0.2, 0.4, 0.8]))
    assert good.delta_vs(bad)["roc_auc"] > 0
