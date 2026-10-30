"""RQ3: what characterises the borrowers whose explanations diverge between the two models?"""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .config import FEATURES


@dataclass
class DivergenceProfile:
    summary: dict
    feature_table: pd.DataFrame


class DivergenceAnalyzer:
    """Compares divergent and agreeing borrowers on score, borderline status and each feature.

    Input is the pooled audit table: one row per audited borrower (per model pair) with the
    columns ``divergent``, ``mean_proba``, ``top_features_a``, ``top_features_b`` and one column
    per feature. The tests here are exploratory (many features, few divergent cases), so
    p-values are reported as a guide and are not corrected for multiple comparisons.
    """

    def __init__(self, features=FEATURES, borderline_range=(0.4, 0.6)):
        self.features = list(features)
        self.low, self.high = borderline_range

    def profile(self, scores: pd.DataFrame) -> DivergenceProfile:
        required = {"divergent", "mean_proba", "top_features_a", "top_features_b", *self.features}
        missing = required - set(scores.columns)
        if missing:
            raise ValueError(f"Audit table is missing columns: {sorted(missing)}")
        divergent = scores[scores["divergent"].astype(bool)]
        agreeing = scores[~scores["divergent"].astype(bool)]

        summary = {"n_divergent": int(len(divergent)), "n_agreeing": int(len(agreeing))}
        if divergent.empty or agreeing.empty:
            summary["note"] = "Both divergent and agreeing cases are needed to compare them."
            return DivergenceProfile(summary, pd.DataFrame())

        in_border = lambda d: float(d["mean_proba"].between(self.low, self.high).mean() * 100)
        summary.update({
            "mean_score_divergent": float(divergent["mean_proba"].mean()),
            "mean_score_agreeing": float(agreeing["mean_proba"].mean()),
            "pct_borderline_divergent": in_border(divergent),
            "pct_borderline_agreeing": in_border(agreeing),
            "most_contested_features": self.contested_features(divergent),
        })
        return DivergenceProfile(summary, self.feature_table(divergent, agreeing))

    def feature_table(self, divergent: pd.DataFrame, agreeing: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for feature in self.features:
            d, a = divergent[feature].astype(float), agreeing[feature].astype(float)
            pooled_std = np.sqrt((d.var(ddof=1) + a.var(ddof=1)) / 2) if len(d) > 1 else np.nan
            effect = (d.mean() - a.mean()) / pooled_std if pooled_std and pooled_std > 0 else np.nan
            try:
                p_value = float(mannwhitneyu(d, a, alternative="two-sided").pvalue)
            except ValueError:  # e.g. every value identical
                p_value = float("nan")
            rows.append({"feature": feature, "median_divergent": float(d.median()),
                         "median_agreeing": float(a.median()),
                         "standardised_difference": float(effect), "mann_whitney_p": p_value})
        table = pd.DataFrame(rows)
        order = table["standardised_difference"].abs().sort_values(ascending=False).index
        return table.loc[order].reset_index(drop=True)

    @staticmethod
    def contested_features(divergent: pd.DataFrame, n: int = 3) -> List[dict]:
        """Features that sit in one model's top-k but not the other's, most often."""
        counts: dict = {}
        for a, b in zip(divergent["top_features_a"], divergent["top_features_b"]):
            for feature in set(a.split("|")) ^ set(b.split("|")):
                counts[feature] = counts.get(feature, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:n]
        return [{"feature": f, "cases": c} for f, c in ranked]
