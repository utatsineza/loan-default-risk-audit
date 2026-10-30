"""Evaluation metrics for the imbalanced default-prediction task."""
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (auc, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


@dataclass(frozen=True)
class EvaluationMetrics:
    model_id: str
    pr_auc: float
    roc_auc: float
    f1: float
    precision: float
    recall: float
    threshold: float

    @classmethod
    def from_predictions(cls, model_id: str, y_true, proba, threshold: float = 0.5):
        y_true, proba = np.asarray(y_true), np.asarray(proba)
        if len(y_true) != len(proba):
            raise ValueError("y_true and proba must have the same length")
        if len(np.unique(y_true)) < 2:
            raise ValueError("y_true must contain both classes to compute AUC metrics")
        y_pred = (proba >= threshold).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, proba)
        return cls(
            model_id=model_id,
            pr_auc=float(auc(recall, precision)),
            roc_auc=float(roc_auc_score(y_true, proba)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            threshold=threshold,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def delta_vs(self, other: "EvaluationMetrics") -> dict:
        """Metric differences (self minus other), e.g. XGBoost minus baseline."""
        return {k: getattr(self, k) - getattr(other, k)
                for k in ("pr_auc", "roc_auc", "f1", "precision", "recall")}
