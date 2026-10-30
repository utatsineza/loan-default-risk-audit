"""Running the stability audit on one matched model pair, and on many pairs.

One matched pair could be a lucky or unlucky draw. Auditing several independently configured
pairs shows how much of the explanation (dis)agreement is a property of the method and how
much is a property of one particular pair of models.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .audit import AuditReport, StabilityAuditor
from .config import Config
from .models import MatchedModelSelector, MatchResult, XGBConfig, XGBoostModel
from .sampling import AuditSampler

# Model A depths used for the extra pairs; the seed is derived from the pair number.
DEPTH_CYCLE = (3, 5, 4, 6, 7)


@dataclass
class SplitData:
    """The three disjoint partitions every pair uses: fit, validation (matching), test."""
    X_fit: pd.DataFrame
    y_fit: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


@dataclass
class PairOutcome:
    pair_id: int
    model_a: XGBoostModel
    model_b: XGBoostModel
    match: MatchResult
    report: AuditReport
    audited: pd.DataFrame      # feature rows of the audited borrowers
    mean_proba: np.ndarray     # average A/B score of those borrowers, same order

    def scores_table(self) -> pd.DataFrame:
        """Audit scores joined with the borrowers' features, for pooled analysis."""
        table = self.report.to_dataframe()
        table.insert(0, "pair_id", self.pair_id)
        table["mean_proba"] = self.mean_proba
        return pd.concat([table, self.audited.reset_index(drop=True)], axis=1)


@dataclass(frozen=True)
class PairResult:
    pair_id: int
    a_depth: int
    a_seed: int
    b_depth: int
    b_seed: int
    matched: bool
    val_auc_difference: float
    test_roc_auc_a: float
    test_roc_auc_b: float
    n_borrowers: int
    mean_spearman: float
    mean_jaccard_top_k: float
    pct_top_feature_match: float
    pct_divergent: float


class PairExperiment:
    """Fit Model A, find a matched Model B, audit them on a borderline-weighted sample."""

    def __init__(self, config: Config):
        self.config = config

    def run(self, pair_id: int, model_a: XGBoostModel, data: SplitData,
            tolerance: Optional[float] = None) -> PairOutcome:
        cfg = self.config
        tolerance = cfg.match_tolerance if tolerance is None else tolerance
        match = MatchedModelSelector(model_a, tolerance).select(
            data.X_fit, data.y_fit, data.X_val, data.y_val)
        model_b = match.model
        mean_proba = (model_a.predict_proba(data.X_test) + model_b.predict_proba(data.X_test)) / 2
        sampler = AuditSampler(min(cfg.audit_sample_size, len(data.X_test)), cfg.borderline_range,
                               cfg.borderline_share, cfg.random_state + pair_id)
        audit_index = sampler.sample(data.X_test, data.y_test, mean_proba)
        audited = data.X_test.loc[audit_index]
        report = StabilityAuditor(model_a, model_b, cfg.top_k, cfg.divergence_jaccard).run(audited)
        position = data.X_test.index.get_indexer(audit_index)
        return PairOutcome(pair_id, model_a, model_b, match, report, audited, mean_proba[position])

    def summarise(self, outcome: PairOutcome, data: SplitData) -> PairResult:
        a, b = outcome.model_a.config, outcome.model_b.config
        summary = outcome.report.summary()
        return PairResult(
            pair_id=outcome.pair_id, a_depth=a.max_depth, a_seed=a.random_state,
            b_depth=b.max_depth, b_seed=b.random_state, matched=outcome.match.matched,
            val_auc_difference=outcome.match.difference,
            test_roc_auc_a=float(roc_auc_score(data.y_test, outcome.model_a.predict_proba(data.X_test))),
            test_roc_auc_b=float(roc_auc_score(data.y_test, outcome.model_b.predict_proba(data.X_test))),
            n_borrowers=summary["n_borrowers"], mean_spearman=summary["mean_spearman"],
            mean_jaccard_top_k=summary["mean_jaccard_top_k"],
            pct_top_feature_match=summary["pct_top_feature_match"],
            pct_divergent=summary["pct_divergent"])


class MultiPairAuditor:
    """Audits the main pair plus ``n_pairs - 1`` additional, differently configured pairs."""

    def __init__(self, config: Config):
        if config.n_pairs < 1:
            raise ValueError("n_pairs must be at least 1")
        self.config = config
        self.experiment = PairExperiment(config)

    def extra_a_configs(self) -> List[XGBConfig]:
        """Model A variants for pairs 1..n-1: different depth and seed from the main Model A."""
        cfg = self.config
        configs = []
        for i in range(1, cfg.n_pairs):
            depth = DEPTH_CYCLE[(i - 1) % len(DEPTH_CYCLE)]
            configs.append(XGBConfig(max_depth=depth, n_estimators=cfg.n_estimators,
                                     random_state=100 * i + 1))
        return configs

    def run_extra_pairs(self, data: SplitData, progress=None) -> List[PairOutcome]:
        outcomes = []
        for pair_id, a_config in enumerate(self.extra_a_configs(), start=1):
            if progress:
                progress(f"Auditing model pair {pair_id + 1} of {self.config.n_pairs} ...")
            model_a = XGBoostModel("model_A", a_config).fit(data.X_fit, data.y_fit)
            outcomes.append(self.experiment.run(pair_id, model_a, data))
        return outcomes

    def tolerance_sweep(self, main: PairOutcome, data: SplitData, progress=None) -> pd.DataFrame:
        """Repeat the main pair's matching and audit at several tolerances."""
        rows = []
        for tolerance in self.config.tolerance_sweep:
            if tolerance == self.config.match_tolerance:
                outcome = main  # already done
            else:
                if progress:
                    progress(f"Matching Model B at tolerance {tolerance} ...")
                outcome = self.experiment.run(main.pair_id, main.model_a, data, tolerance)
            summary = outcome.report.summary()
            rows.append({"tolerance": tolerance, "matched": outcome.match.matched,
                         "val_auc_difference": outcome.match.difference,
                         "candidates_tried": outcome.match.candidates_tried,
                         "b_depth": outcome.model_b.config.max_depth,
                         "b_seed": outcome.model_b.config.random_state,
                         "mean_spearman": summary["mean_spearman"],
                         "mean_jaccard_top_k": summary["mean_jaccard_top_k"],
                         "pct_divergent": summary["pct_divergent"]})
        return pd.DataFrame(rows)

    @staticmethod
    def aggregate(results: List[PairResult]) -> dict:
        """Mean, spread and range of the headline audit numbers across pairs.

        Only pairs that met the matching tolerance count, because the research question is
        about *equally accurate* models. If no pair matched, all pairs are used and
        ``used_matched_pairs_only`` is False so the caller can say so.
        """
        df = pd.DataFrame([r.__dict__ for r in results])
        matched = df[df["matched"]]
        used = matched if len(matched) else df
        out = {"n_pairs": int(len(df)), "n_matched_pairs": int(len(matched)),
               "used_matched_pairs_only": bool(len(matched))}
        for column in ("mean_spearman", "mean_jaccard_top_k", "pct_top_feature_match",
                       "pct_divergent"):
            out[column] = {"mean": float(used[column].mean()),
                           "std": float(used[column].std(ddof=0)),
                           "min": float(used[column].min()), "max": float(used[column].max())}
        return out
