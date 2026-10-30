"""Unit tests: audit sampling algorithm."""
import numpy as np
import pandas as pd
import pytest

from loan_audit.sampling import AuditSampler


@pytest.fixture
def population():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f": rng.random(1000)})
    y = pd.Series((rng.random(1000) < 0.1).astype(int), index=X.index)
    proba = pd.Series(rng.random(1000), index=X.index)
    return X, y, proba


def test_sample_has_requested_size_and_no_duplicates(population):
    X, y, proba = population
    chosen = AuditSampler(size=200, random_state=1).sample(X, y, proba)
    assert len(chosen) == 200 and chosen.is_unique


def test_borderline_cases_are_over_represented(population):
    X, y, proba = population
    chosen = AuditSampler(size=200, borderline_share=0.5, random_state=1).sample(X, y, proba)
    share = ((proba[chosen] >= 0.4) & (proba[chosen] <= 0.6)).mean()
    assert share >= 0.5  # base rate of the 0.4-0.6 band is about 0.2


def test_handles_too_few_borderline_cases(population):
    X, y, _ = population
    proba = pd.Series(0.05, index=X.index)  # nobody is borderline
    chosen = AuditSampler(size=100, random_state=1).sample(X, y, proba)
    assert len(chosen) == 100


def test_rejects_oversized_sample(population):
    X, y, proba = population
    with pytest.raises(ValueError, match="Cannot sample"):
        AuditSampler(size=5000).sample(X, y, proba)


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        AuditSampler(size=0)
    with pytest.raises(ValueError):
        AuditSampler(size=10, borderline_share=1.5)
