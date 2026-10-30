"""SHAP explanations as small value objects plus an explainer wrapper."""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import shap

from .models import XGBoostModel


@dataclass(frozen=True)
class SHAPResult:
    borrower_index: object
    model_id: str
    feature_names: tuple
    values: np.ndarray
    base_value: float

    @property
    def ranking(self) -> List[str]:
        """Feature names ordered from most to least influential (by |SHAP value|)."""
        order = np.argsort(-np.abs(self.values), kind="stable")
        return [self.feature_names[i] for i in order]

    def top_k(self, k: int) -> List[str]:
        return self.ranking[:k]


class SHAPExplainer:
    """Wraps shap.TreeExplainer for one XGBoost model."""

    def __init__(self, model: XGBoostModel):
        self.model = model
        self._explainer = shap.TreeExplainer(model.native)

    def explain(self, X: pd.DataFrame) -> List[SHAPResult]:
        values = self._explainer.shap_values(X)
        if isinstance(values, list):  # older SHAP versions return one array per class
            values = values[1]
        base = float(np.ravel(self._explainer.expected_value)[0])
        names = tuple(X.columns)
        return [SHAPResult(X.index[i], self.model.model_id, names,
                           np.asarray(values[i], dtype=float), base)
                for i in range(len(X))]

    def explain_one(self, row: pd.DataFrame) -> SHAPResult:
        return self.explain(row.iloc[[0]])[0]
