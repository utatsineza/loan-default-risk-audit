"""Unit tests: profiling divergent borrowers, and the audit sensitivity grid."""
import numpy as np
import pandas as pd
import pytest

from loan_audit.config import FEATURES
from loan_audit.divergence import DivergenceAnalyzer


def make_scores(n_divergent=6, n_agreeing=24, seed=0):
    rng = np.random.default_rng(seed)
    n = n_divergent + n_agreeing
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    df["divergent"] = [True] * n_divergent + [False] * n_agreeing
    df["mean_proba"] = np.where(df["divergent"], 0.5, 0.2)
    df["top_features_a"] = "age|DebtRatio|MonthlyIncome"
    df["top_features_b"] = np.where(df["divergent"], "age|NumberOfDependents|DebtRatio",
                                    "age|DebtRatio|MonthlyIncome")
    return df


def test_profile_compares_groups():
    profile = DivergenceAnalyzer().profile(make_scores())
    s = profile.summary
    assert s["n_divergent"] == 6 and s["n_agreeing"] == 24
    assert s["pct_borderline_divergent"] == 100.0 and s["pct_borderline_agreeing"] == 0.0
    assert s["mean_score_divergent"] > s["mean_score_agreeing"]
    assert len(profile.feature_table) == len(FEATURES)


def test_contested_features_are_the_ones_that_differ():
    contested = DivergenceAnalyzer().profile(make_scores()).summary["most_contested_features"]
    assert {c["feature"] for c in contested} == {"MonthlyIncome", "NumberOfDependents"}
    assert all(c["cases"] == 6 for c in contested)


def test_no_divergent_cases_is_handled():
    profile = DivergenceAnalyzer().profile(make_scores(n_divergent=0))
    assert profile.summary["n_divergent"] == 0 and "note" in profile.summary
    assert profile.feature_table.empty


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        DivergenceAnalyzer().profile(make_scores().drop(columns=["age"]))


def test_constant_feature_does_not_crash():
    scores = make_scores()
    scores["age"] = 40.0
    table = DivergenceAnalyzer().profile(scores).feature_table
    assert table.loc[table.feature == "age", "mann_whitney_p"].iloc[0] != 0.0


def test_sensitivity_grid_is_monotonic_in_threshold(synthetic_df):
    from loan_audit.audit import StabilityAuditor
    from loan_audit.config import Config
    from loan_audit.models import XGBConfig, XGBoostModel
    from loan_audit.pipeline import ExperimentPipeline

    data = ExperimentPipeline(Config(), "unused").prepare(synthetic_df)
    a = XGBoostModel("a", XGBConfig(n_estimators=20, random_state=1)).fit(data.X_fit, data.y_fit)
    b = XGBoostModel("b", XGBConfig(n_estimators=20, max_depth=6, random_state=2)).fit(
        data.X_fit, data.y_fit)
    grid = StabilityAuditor(a, b).sensitivity(data.X_test.iloc[:50], ks=(2, 3), thresholds=(0.3, 0.7, 1.0))
    assert len(grid) == 6
    for _, group in grid.groupby("top_k"):
        shares = group.sort_values("jaccard_threshold")["pct_divergent"].tolist()
        assert shares == sorted(shares)  # a looser threshold can only flag more cases
