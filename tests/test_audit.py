"""Unit + integration tests: agreement metrics and the stability auditor."""
import numpy as np
import pytest

from loan_audit.audit import StabilityAuditor, jaccard_top_k, spearman_rank_correlation
from loan_audit.config import Config
from loan_audit.data import DataPreprocessor
from loan_audit.models import XGBConfig, XGBoostModel


def test_jaccard_identical_and_disjoint_and_partial():
    a = np.array([5, 4, 3, 2, 1, 0], float)
    assert jaccard_top_k(a, a, 3) == 1.0
    assert jaccard_top_k(a, a[::-1], 3) == 0.0
    b = np.array([5, 4, 0, 3, 2, 1], float)  # shares 2 of top-3
    assert jaccard_top_k(a, b, 3) == pytest.approx(2 / 4)


def test_jaccard_ignores_sign():
    a = np.array([-5, 4, 3, 2, 1, 0], float)
    b = np.array([5, -4, 3, 2, 1, 0], float)
    assert jaccard_top_k(a, b, 3) == 1.0


def test_spearman_identical_and_reversed():
    a = np.array([5, 4, 3, 2, 1], float)
    assert spearman_rank_correlation(a, a) == pytest.approx(1.0)
    assert spearman_rank_correlation(a, a[::-1]) == pytest.approx(-1.0)


def test_spearman_constant_profiles_do_not_return_nan():
    flat = np.zeros(5)
    assert spearman_rank_correlation(flat, flat) == 1.0
    assert spearman_rank_correlation(flat, np.arange(5.0)) == 0.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        jaccard_top_k(np.ones(4), np.ones(5), 2)
    with pytest.raises(ValueError):
        jaccard_top_k(np.ones(4), np.ones(4), 9)
    with pytest.raises(ValueError):
        spearman_rank_correlation(np.ones(3), np.ones(4))


@pytest.fixture(scope="module")
def fitted(synthetic_df):
    pre = DataPreprocessor(Config())
    train, test = pre.split(synthetic_df)
    pre.fit_imputer(train)
    X_train, y_train = pre.xy(pre.transform(train))
    X_test, _ = pre.xy(pre.transform(test))
    model_a = XGBoostModel("A", XGBConfig(n_estimators=30)).fit(X_train, y_train)
    model_b = XGBoostModel("B", XGBConfig(n_estimators=30, max_depth=6, random_state=7)).fit(
        X_train, y_train)
    return model_a, model_b, X_test


def test_identical_models_agree_perfectly(fitted):
    model_a, _, X_test = fitted
    report = StabilityAuditor(model_a, model_a).run(X_test.head(25))
    summary = report.summary()
    assert summary["mean_jaccard_top_k"] == pytest.approx(1.0)
    assert summary["mean_spearman"] == pytest.approx(1.0)
    assert summary["pct_divergent"] == 0.0


def test_different_models_produce_valid_scores(fitted):
    model_a, model_b, X_test = fitted
    report = StabilityAuditor(model_a, model_b).run(X_test.head(25))
    df = report.to_dataframe()
    assert len(df) == 25
    assert df["jaccard_top_k"].between(0, 1).all()
    assert df["spearman"].between(-1, 1).all()


def test_empty_input_raises(fitted):
    model_a, model_b, X_test = fitted
    with pytest.raises(ValueError, match="No borrowers"):
        StabilityAuditor(model_a, model_b).run(X_test.head(0))
