"""Unit tests: the cost-sensitive vs resampling comparison."""
import numpy as np
import pandas as pd
import pytest

from loan_audit.config import Config
from loan_audit.data import DataPreprocessor
from loan_audit.models import XGBConfig, XGBoostModel
from loan_audit.resampling import ResamplingComparison, best_f1


@pytest.fixture(scope="module")
def splits(synthetic_df):
    pre = DataPreprocessor(Config())
    train, test = pre.split(synthetic_df)
    pre.fit_imputer(train)
    return pre.xy(pre.transform(train)) + pre.xy(pre.transform(test))


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        ResamplingComparison(XGBConfig(n_estimators=10), methods=("bogus",))


def test_cost_sensitive_switch_changes_class_weight(splits):
    X, y, _, _ = splits
    config = XGBConfig(n_estimators=10)
    weighted = XGBoostModel("w", config).fit(X, y)
    plain = XGBoostModel("p", config, cost_sensitive=False).fit(X, y)
    assert weighted.scale_pos_weight > 5
    assert plain.scale_pos_weight == 1.0


def test_resampling_balances_training_data_only(splits):
    X, y, _, _ = splits
    comparison = ResamplingComparison(XGBConfig(n_estimators=10), ("smote",))
    X_res, y_res = comparison.resample("smote", X, y)
    assert abs(y_res.mean() - 0.5) < 0.01
    assert len(X_res) > len(X)
    assert list(X_res.columns) == list(X.columns)


def test_comparison_table_has_every_strategy(splits):
    X, y, X_test, y_test = splits
    table = ResamplingComparison(XGBConfig(n_estimators=15), ("smote", "adasyn")).run(
        X, y, X_test, y_test)
    assert list(table["model_id"]) == ["xgb_cost_sensitive", "xgb_no_imbalance_handling",
                                       "xgb_smote", "xgb_adasyn"]
    assert table["roc_auc"].between(0, 1).all()
    # resampled models train on more rows and a balanced class mix; weighted ones do not
    assert table.loc[table.model_id == "xgb_smote", "n_train_rows"].iloc[0] > len(y)
    assert table.loc[table.model_id == "xgb_cost_sensitive", "train_default_rate"].iloc[0] < 0.1


def test_best_f1_is_one_for_perfect_scores():
    assert best_f1([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_best_f1_handles_no_positive_predictions_gracefully():
    assert 0 <= best_f1([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) <= 1
