"""Selecting the borrowers used in the stability audit."""
import numpy as np
import pandas as pd


class AuditSampler:
    """Stratified sample that deliberately over-represents borderline cases.

    Algorithm: (1) take up to ``size * borderline_share`` cases whose probability lies in
    ``borderline_range``; (2) fill the remaining slots from all other cases, stratified by
    the true label so the default rate is preserved. No case is selected twice.
    """

    def __init__(self, size: int = 300, borderline_range=(0.4, 0.6),
                 borderline_share: float = 0.5, random_state: int = 42):
        if size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= borderline_share <= 1:
            raise ValueError("borderline_share must be between 0 and 1")
        self.size = size
        self.low, self.high = borderline_range
        self.borderline_share = borderline_share
        self.rng = np.random.default_rng(random_state)

    def sample(self, X: pd.DataFrame, y: pd.Series, proba) -> pd.Index:
        if self.size > len(X):
            raise ValueError(f"Cannot sample {self.size} cases from {len(X)} rows")
        proba = pd.Series(np.asarray(proba), index=X.index)
        labels = pd.Series(np.asarray(y), index=X.index)

        borderline = proba.index[(proba >= self.low) & (proba <= self.high)]
        n_border = min(int(round(self.size * self.borderline_share)), len(borderline))
        chosen = list(self.rng.choice(borderline, size=n_border, replace=False)) if n_border else []

        remaining = labels.drop(index=chosen)
        slots = self.size - len(chosen)
        shares = remaining.value_counts(normalize=True)
        for label, share in shares.items():
            pool = remaining.index[remaining == label]
            take = min(int(round(slots * share)), len(pool))
            chosen.extend(self.rng.choice(pool, size=take, replace=False))
        # rounding can leave us a few short: top up from whatever is unused
        if len(chosen) < self.size:
            unused = labels.index.difference(chosen)
            extra = self.rng.choice(unused, size=self.size - len(chosen), replace=False)
            chosen.extend(extra)
        return pd.Index(chosen[: self.size])
