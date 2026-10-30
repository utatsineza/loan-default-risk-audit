"""Unit tests: data loading, validation, splitting and imputation."""
import pandas as pd
import pytest

from loan_audit.config import Config, TARGET
from loan_audit.data import DataPreprocessor


def test_validate_schema_rejects_missing_column(synthetic_df):
    broken = synthetic_df.drop(columns=["age"])
    with pytest.raises(ValueError, match="Missing required columns"):
        DataPreprocessor().validate_schema(broken)


def test_validate_schema_rejects_non_binary_target(synthetic_df):
    broken = synthetic_df.copy()
    broken.loc[broken.index[0], TARGET] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        DataPreprocessor().validate_schema(broken)


def test_validate_schema_rejects_missing_target_values(synthetic_df):
    broken = synthetic_df.copy()
    broken.loc[broken.index[0], TARGET] = None
    with pytest.raises(ValueError, match="missing values"):
        DataPreprocessor().validate_schema(broken)


def test_split_is_stratified(synthetic_df):
    train, test = DataPreprocessor().split(synthetic_df)
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.01
    assert len(train) + len(test) == len(synthetic_df)


def test_imputer_uses_training_medians_only(synthetic_df):
    pre = DataPreprocessor(Config())
    train, test = pre.split(synthetic_df)
    pre.fit_imputer(train)
    cleaned = pre.transform(test)
    assert cleaned[list(pre.IMPUTE_COLUMNS)].isnull().sum().sum() == 0
    assert pre.medians_["MonthlyIncome"] == pytest.approx(train["MonthlyIncome"].median())


def test_transform_before_fit_raises(synthetic_df):
    with pytest.raises(RuntimeError):
        DataPreprocessor().transform(synthetic_df)


def test_load_drops_unnamed_index_column(tmp_path, synthetic_df):
    path = tmp_path / "cs-training.csv"
    synthetic_df.to_csv(path)  # writes an unnamed index column like Kaggle's file
    loaded = DataPreprocessor().load(str(path))
    assert "Unnamed: 0" not in loaded.columns
    assert len(loaded) == len(synthetic_df)
