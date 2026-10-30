import pytest

from loan_audit.synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic(n=2500, seed=1)
