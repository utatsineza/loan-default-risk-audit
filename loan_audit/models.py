"""Model classes (abstraction + polymorphism) and the matched-accuracy selector."""
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BaseModel(ABC):
    """Common interface so the pipeline can treat every model the same way."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def fit(self, X, y) -> "BaseModel":
        ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Probability of default (class 1) as a 1-D array."""

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class LogisticBaseline(BaseModel):
    """Class-weighted, scaled logistic regression used as the baseline."""

    def __init__(self, model_id: str = "baseline_lr", random_state: int = 42):
        super().__init__(model_id)
        # Scaling avoids convergence problems from features on very different scales.
        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)[:, 1]


@dataclass(frozen=True)
class XGBConfig:
    max_depth: int = 4
    n_estimators: int = 200
    learning_rate: float = 0.1
    # Row/column subsampling makes the random seed matter; without it XGBoost is
    # near-deterministic and two seeds would give identical models.
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


class XGBoostModel(BaseModel):
    """XGBoost with cost-sensitive weighting (scale_pos_weight), no resampling."""

    def __init__(self, model_id: str, config: XGBConfig = XGBConfig()):
        super().__init__(model_id)
        self.config = config
        self.scale_pos_weight: Optional[float] = None
        self._model: Optional[xgb.XGBClassifier] = None

    def fit(self, X, y):
        positives, negatives = int((y == 1).sum()), int((y == 0).sum())
        if positives == 0 or negatives == 0:
            raise ValueError("Training labels must contain both classes")
        self.scale_pos_weight = negatives / positives  # inverse class ratio
        c = self.config
        self._model = xgb.XGBClassifier(
            max_depth=c.max_depth, n_estimators=c.n_estimators,
            learning_rate=c.learning_rate, subsample=c.subsample,
            colsample_bytree=c.colsample_bytree, random_state=c.random_state,
            scale_pos_weight=self.scale_pos_weight, eval_metric="aucpr",
            tree_method="hist",
        )
        self._model.fit(X, y)
        return self

    def _require_fitted(self) -> xgb.XGBClassifier:
        if self._model is None:
            raise RuntimeError(f"{self.model_id} has not been fitted or loaded yet")
        return self._model

    def predict_proba(self, X):
        return self._require_fitted().predict_proba(X)[:, 1]

    @property
    def native(self) -> xgb.XGBClassifier:
        """Underlying XGBClassifier (needed by SHAP's TreeExplainer)."""
        return self._require_fitted()

    def save(self, path: str) -> None:
        self._require_fitted().save_model(path)

    @classmethod
    def load(cls, path: str, model_id: str = "loaded_model") -> "XGBoostModel":
        instance = cls(model_id)
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(path)
        return instance


@dataclass
class MatchResult:
    model: XGBoostModel
    reference_auc: float
    candidate_auc: float
    difference: float
    matched: bool
    candidates_tried: int


class MatchedModelSelector:
    """Finds Model B: a differently configured XGBoost with (nearly) the same ROC-AUC as Model A.

    Algorithm: enumerate candidate configurations that differ from the reference in BOTH
    tree depth and random seed, train each, and accept the first whose validation ROC-AUC is
    within ``tolerance`` of the reference. If none qualifies within ``max_candidates``, the
    closest candidate is returned with ``matched=False`` so the caller can report it honestly.
    Selection uses validation data only, never the test set.
    """

    DEPTHS = (3, 5, 6, 7)
    SEEDS = (7, 13, 21)

    def __init__(self, reference: XGBoostModel, tolerance: float = 0.005,
                 max_candidates: int = 24):
        self.reference = reference
        self.tolerance = tolerance
        self.max_candidates = max_candidates

    def candidate_configs(self) -> Iterator[XGBConfig]:
        ref = self.reference.config
        depths = [d for d in self.DEPTHS if d != ref.max_depth]
        seeds = [s for s in self.SEEDS if s != ref.random_state]
        n_estimators = sorted({max(10, int(ref.n_estimators * f)) for f in (1.0, 0.75, 1.5)})
        rates = (ref.learning_rate, ref.learning_rate / 2)
        grid = itertools.product(depths, seeds, n_estimators, rates)
        for depth, seed, n_est, rate in itertools.islice(grid, self.max_candidates):
            yield XGBConfig(max_depth=depth, n_estimators=n_est, learning_rate=rate,
                            subsample=ref.subsample, colsample_bytree=ref.colsample_bytree,
                            random_state=seed)

    def select(self, X_fit, y_fit, X_val, y_val) -> MatchResult:
        reference_auc = roc_auc_score(y_val, self.reference.predict_proba(X_val))
        best: Optional[MatchResult] = None
        tried = 0
        for config in self.candidate_configs():
            tried += 1
            candidate = XGBoostModel("model_B", config).fit(X_fit, y_fit)
            auc_value = roc_auc_score(y_val, candidate.predict_proba(X_val))
            difference = abs(auc_value - reference_auc)
            if best is None or difference < best.difference:
                best = MatchResult(candidate, reference_auc, auc_value, difference,
                                   difference <= self.tolerance, tried)
            if difference <= self.tolerance:
                break
        if best is None:
            raise RuntimeError("No candidate configurations were generated")
        best.candidates_tried = tried
        return best
