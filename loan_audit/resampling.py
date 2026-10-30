"""Cost-sensitive weighting versus resampling (SMOTE / ADASYN) on the same data and split.

The thesis argues that cost-sensitive weighting is preferable to synthetic resampling. This
module tests that claim on this project's own data: identical XGBoost settings, identical
held-out test set; only the way class imbalance is handled changes.
"""
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import precision_recall_curve

from .metrics import EvaluationMetrics
from .models import XGBConfig, XGBoostModel

SAMPLERS = {"smote": SMOTE, "adasyn": ADASYN}


def best_f1(y_true, proba) -> float:
    """Highest F1 over all decision thresholds, so models are not penalised for how their
    score scale shifts after resampling or weighting (F1 at a fixed 0.5 is not comparable)."""
    precision, recall, _ = precision_recall_curve(np.asarray(y_true), np.asarray(proba))
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    return float(f1.max())


class ResamplingComparison:
    """Trains one XGBoost per imbalance strategy and evaluates all on the same test set."""

    def __init__(self, xgb_config: XGBConfig, methods: Tuple[str, ...] = ("smote", "adasyn"),
                 random_state: int = 42):
        unknown = [m for m in methods if m not in SAMPLERS]
        if unknown:
            raise ValueError(f"Unknown resampling method(s): {unknown}. Use {list(SAMPLERS)}")
        self.xgb_config = xgb_config
        self.methods = methods
        self.random_state = random_state
        self.skipped: List[str] = []

    def resample(self, method: str, X: pd.DataFrame, y: pd.Series):
        """Oversample the minority class. Applied to the training data only, never to the
        validation or test data."""
        return SAMPLERS[method](random_state=self.random_state).fit_resample(X, y)

    def _evaluate(self, label: str, model: XGBoostModel, X_fit, y_fit, X_test, y_test,
                  seconds: float) -> dict:
        proba = model.predict_proba(X_test)
        row = EvaluationMetrics.from_predictions(label, y_test, proba).to_dict()
        row.pop("threshold")
        row["best_f1"] = best_f1(y_test, proba)
        row["n_train_rows"] = int(len(y_fit))
        row["train_default_rate"] = float(np.mean(np.asarray(y_fit)))
        row["seconds"] = round(seconds, 2)
        return row

    def run(self, X_fit, y_fit, X_test, y_test,
            cost_sensitive_model: Optional[XGBoostModel] = None) -> pd.DataFrame:
        rows = []

        def timed_fit(label, model, X, y):
            start = time.perf_counter()
            model.fit(X, y)
            rows.append(self._evaluate(label, model, X, y, X_test, y_test,
                                       time.perf_counter() - start))

        if cost_sensitive_model is None:
            timed_fit("xgb_cost_sensitive", XGBoostModel("xgb_cost_sensitive", self.xgb_config),
                      X_fit, y_fit)
        else:  # reuse Model A rather than training an identical model again
            rows.append(self._evaluate("xgb_cost_sensitive", cost_sensitive_model, X_fit, y_fit,
                                       X_test, y_test, 0.0))
        timed_fit("xgb_no_imbalance_handling",
                  XGBoostModel("xgb_no_imbalance_handling", self.xgb_config, cost_sensitive=False),
                  X_fit, y_fit)
        self.skipped = []
        for method in self.methods:
            try:
                X_res, y_res = self.resample(method, X_fit, y_fit)
            except (RuntimeError, ValueError) as error:  # e.g. ADASYN finds no hard cases
                self.skipped.append(f"{method}: {error}")
                continue
            timed_fit(f"xgb_{method}", XGBoostModel(f"xgb_{method}", self.xgb_config,
                                                    cost_sensitive=False), X_res, y_res)
        return pd.DataFrame(rows)
