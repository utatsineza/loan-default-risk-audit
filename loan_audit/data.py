"""Loading, validating, splitting and imputing the Give Me Some Credit data."""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config


class DataPreprocessor:
    """Prepares data without leaking test-set information.

    Medians used for imputation are learned from training data only and then applied
    to every split (encapsulated in ``medians_``).
    """

    IMPUTE_COLUMNS = ("MonthlyIncome", "NumberOfDependents")

    def __init__(self, config: Config = Config()):
        self.config = config
        self.medians_ = {}

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        first_column = df.columns[0]
        if str(first_column).startswith("Unnamed"):  # Kaggle's unnamed index column
            df = df.drop(columns=[first_column])
        self.validate_schema(df)
        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        required = set(self.config.features) | {self.config.target}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        target = df[self.config.target]
        if target.isnull().any():
            raise ValueError("Target column contains missing values")
        if not set(target.unique()) <= {0, 1}:
            raise ValueError("Target column must contain only 0 and 1")
        if len(df) < 50:
            raise ValueError("Dataset is too small to split and model")

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified train/test split that preserves the default rate."""
        return train_test_split(
            df, test_size=self.config.test_size,
            stratify=df[self.config.target], random_state=self.config.random_state,
        )

    def split_validation(self, train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carve a validation set out of the training data (used for model matching)."""
        return train_test_split(
            train, test_size=self.config.val_size,
            stratify=train[self.config.target], random_state=self.config.random_state,
        )

    def fit_imputer(self, train: pd.DataFrame) -> "DataPreprocessor":
        self.medians_ = {c: float(train[c].median()) for c in self.IMPUTE_COLUMNS}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.medians_:
            raise RuntimeError("Call fit_imputer() before transform()")
        out = df.copy()
        for column, median in self.medians_.items():
            out[column] = out[column].fillna(median)
        return out

    def xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return df[list(self.config.features)], df[self.config.target]
