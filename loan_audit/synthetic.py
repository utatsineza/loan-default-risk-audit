"""Synthetic borrower data with the same schema as Give Me Some Credit.

Used by the tests and for a quick smoke test of the pipeline; never for reported results.
"""
import numpy as np
import pandas as pd

from .config import FEATURES, TARGET


def make_synthetic(n: int = 4000, default_rate: float = 0.067, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "RevolvingUtilizationOfUnsecuredLines": np.clip(rng.beta(1, 3, n) * 1.2, 0, 3),
        "age": rng.integers(21, 85, n),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.poisson(0.3, n),
        "DebtRatio": rng.lognormal(-1, 1, n),
        "MonthlyIncome": rng.lognormal(8.5, 0.6, n),
        "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, n),
        "NumberOfTimes90DaysLate": rng.poisson(0.1, n),
        "NumberRealEstateLoansOrLines": rng.poisson(1, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.1, n),
        "NumberOfDependents": rng.poisson(0.8, n).astype(float),
    }
    df = pd.DataFrame(data)[list(FEATURES)]
    z = (2.2 * df.iloc[:, 0] + 0.5 * df["NumberOfTime30-59DaysPastDueNotWorse"]
         + 0.8 * df["NumberOfTime60-89DaysPastDueNotWorse"]
         + 1.2 * df["NumberOfTimes90DaysLate"] - 0.02 * (df["age"] - 45)
         + 0.3 * np.log1p(df["DebtRatio"]))
    low, high = -15.0, 5.0
    for _ in range(50):  # bisection: pick the intercept that hits the target default rate
        mid = (low + high) / 2
        rate = (1 / (1 + np.exp(-(mid + z)))).mean()
        low, high = (mid, high) if rate < default_rate else (low, mid)
    p = 1 / (1 + np.exp(-(mid + z)))
    df[TARGET] = (rng.random(n) < p).astype(int)
    df.loc[rng.random(n) < 0.20, "MonthlyIncome"] = np.nan
    df.loc[rng.random(n) < 0.026, "NumberOfDependents"] = np.nan
    return df
