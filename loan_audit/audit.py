"""The SHAP explanation stability audit (the thesis' core contribution)."""
from dataclasses import asdict, dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .explain import SHAPExplainer, SHAPResult
from .models import XGBoostModel


def spearman_rank_correlation(values_a, values_b) -> float:
    """Spearman correlation between two feature-importance profiles (|SHAP| values).

    Constant profiles have undefined correlation; they score 1.0 if identical, else 0.0.
    """
    a, b = np.abs(np.asarray(values_a, float)), np.abs(np.asarray(values_b, float))
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays of the same length")
    if len(a) < 2:
        raise ValueError("At least two features are needed")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(spearmanr(a, b)[0])


def jaccard_top_k(values_a, values_b, k: int) -> float:
    """Jaccard index of the sets of top-k features (by |SHAP|) of two profiles."""
    a, b = np.abs(np.asarray(values_a, float)), np.abs(np.asarray(values_b, float))
    if a.shape != b.shape:
        raise ValueError("Inputs must have the same length")
    if not 1 <= k <= len(a):
        raise ValueError("k must be between 1 and the number of features")
    top_a = set(np.argsort(-a, kind="stable")[:k])
    top_b = set(np.argsort(-b, kind="stable")[:k])
    return len(top_a & top_b) / len(top_a | top_b)


@dataclass(frozen=True)
class AgreementScore:
    borrower_index: object
    spearman: float
    jaccard_top_k: float
    top_feature_match: bool
    divergent: bool
    top_features_a: str = ""   # "|"-joined top-k features of Model A (CSV friendly)
    top_features_b: str = ""


class AuditReport:
    def __init__(self, scores: List[AgreementScore], top_k: int):
        self.scores = scores
        self.top_k = top_k

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in self.scores])

    def summary(self) -> dict:
        df = self.to_dataframe()
        return {
            "n_borrowers": int(len(df)),
            "top_k": self.top_k,
            "mean_spearman": float(df["spearman"].mean()),
            "median_spearman": float(df["spearman"].median()),
            "mean_jaccard_top_k": float(df["jaccard_top_k"].mean()),
            "pct_top_feature_match": float(df["top_feature_match"].mean() * 100),
            "pct_divergent": float(df["divergent"].mean() * 100),
        }

    def divergent_cases(self) -> List[AgreementScore]:
        return [s for s in self.scores if s.divergent]


class StabilityAuditor:
    """Compares the SHAP explanations of two matched-accuracy models borrower by borrower."""

    def __init__(self, model_a: XGBoostModel, model_b: XGBoostModel,
                 top_k: int = 3, divergence_jaccard: float = 0.5):
        self.explainer_a = SHAPExplainer(model_a)
        self.explainer_b = SHAPExplainer(model_b)
        self.top_k = top_k
        self.divergence_jaccard = divergence_jaccard

    def score_pair(self, a: SHAPResult, b: SHAPResult) -> AgreementScore:
        jaccard = jaccard_top_k(a.values, b.values, self.top_k)
        return AgreementScore(
            borrower_index=a.borrower_index,
            spearman=spearman_rank_correlation(a.values, b.values),
            jaccard_top_k=jaccard,
            top_feature_match=a.ranking[0] == b.ranking[0],
            divergent=jaccard < self.divergence_jaccard,
            top_features_a="|".join(a.top_k(self.top_k)),
            top_features_b="|".join(b.top_k(self.top_k)),
        )

    def run(self, X: pd.DataFrame) -> AuditReport:
        if X.empty:
            raise ValueError("No borrowers supplied for the audit")
        results_a = self.explainer_a.explain(X)
        results_b = self.explainer_b.explain(X)
        scores = [self.score_pair(a, b) for a, b in zip(results_a, results_b)]
        return AuditReport(scores, self.top_k)

    def sensitivity(self, X: pd.DataFrame, ks=(2, 3, 5),
                    thresholds=(0.3, 0.5, 0.7, 1.0)) -> pd.DataFrame:
        """How the share of divergent cases changes with the two analyst-chosen settings.

        ``top_k`` (how many features are compared) and the Jaccard threshold (how little overlap
        counts as divergent) are not given by theory, so the headline "% divergent" is reported
        across a grid of them. A threshold of 1.0 means any difference in the top-k set.
        SHAP values are computed once and reused for every grid cell.
        """
        if X.empty:
            raise ValueError("No borrowers supplied for the audit")
        results_a = self.explainer_a.explain(X)
        results_b = self.explainer_b.explain(X)
        rows = []
        for k in ks:
            jaccards = np.array([jaccard_top_k(a.values, b.values, k)
                                 for a, b in zip(results_a, results_b)])
            for threshold in thresholds:
                rows.append({"top_k": k, "jaccard_threshold": threshold,
                             "pct_divergent": float((jaccards < threshold).mean() * 100),
                             "mean_jaccard": float(jaccards.mean())})
        return pd.DataFrame(rows)
