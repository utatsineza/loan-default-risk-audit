"""Unit tests: model classes and the matched-model selector."""
import numpy as np
import pandas as pd
import pytest

from loan_audit.models import (LogisticBaseline, MatchedModelSelector, XGBConfig,
                               XGBoostModel)


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((600, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(0, 0.3, 600) > 0.85).astype(int))
    return X, y


def test_scale_pos_weight_is_inverse_class_ratio(xy):
    X, y = xy
    model = XGBoostModel("m", XGBConfig(n_estimators=10)).fit(X, y)
    assert model.scale_pos_weight == pytest.approx((y == 0).sum() / (y == 1).sum())


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        XGBoostModel("m").predict_proba(pd.DataFrame({"a": [1.0]}))


def test_single_class_training_labels_rejected(xy):
    X, _ = xy
    with pytest.raises(ValueError, match="both classes"):
        XGBoostModel("m").fit(X, pd.Series(np.zeros(len(X), dtype=int)))


def test_models_share_interface(xy):
    X, y = xy
    for model in (LogisticBaseline(), XGBoostModel("m", XGBConfig(n_estimators=10))):
        probabilities = model.fit(X, y).predict_proba(X)
        assert probabilities.shape == (len(X),) and ((0 <= probabilities) & (probabilities <= 1)).all()


def test_save_and_load_round_trip(xy, tmp_path):
    X, y = xy
    model = XGBoostModel("m", XGBConfig(n_estimators=10)).fit(X, y)
    path = str(tmp_path / "model.json")
    model.save(path)
    loaded = XGBoostModel.load(path)
    np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X), rtol=1e-5)


def test_candidates_differ_in_depth_and_seed(xy):
    X, y = xy
    reference = XGBoostModel("A", XGBConfig(n_estimators=10))
    selector = MatchedModelSelector(reference)
    configs = list(selector.candidate_configs())
    assert configs
    assert all(c.max_depth != reference.config.max_depth for c in configs)
    assert all(c.random_state != reference.config.random_state for c in configs)


def test_selector_returns_a_model_and_reports_match(xy):
    X, y = xy
    reference = XGBoostModel("A", XGBConfig(n_estimators=20)).fit(X, y)
    result = MatchedModelSelector(reference, tolerance=0.5, max_candidates=3).select(X, y, X, y)
    assert result.matched is True and result.candidates_tried >= 1
    assert result.model.model_id == "model_B"
