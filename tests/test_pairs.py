"""Unit tests: auditing several matched model pairs."""
import pytest

from loan_audit.config import Config
from loan_audit.pairs import MultiPairAuditor, PairExperiment, PairResult
from loan_audit.pipeline import ExperimentPipeline

CFG = Config(n_estimators=20, audit_sample_size=40, n_pairs=3)


def test_extra_pairs_differ_from_main_model_a_and_each_other():
    configs = MultiPairAuditor(CFG).extra_a_configs()
    assert len(configs) == 2
    assert all(c.random_state != CFG.random_state for c in configs)
    assert len({(c.max_depth, c.random_state) for c in configs}) == 2


def test_single_pair_has_no_extra_configs():
    assert MultiPairAuditor(Config(n_pairs=1)).extra_a_configs() == []


def test_invalid_pair_count_rejected():
    with pytest.raises(ValueError):
        MultiPairAuditor(Config(n_pairs=0))


def test_pair_experiment_audits_requested_sample(synthetic_df):
    pipeline = ExperimentPipeline(CFG, "unused")
    data = pipeline.prepare(synthetic_df)
    from loan_audit.models import XGBConfig, XGBoostModel
    model_a = XGBoostModel("model_A", XGBConfig(n_estimators=20)).fit(data.X_fit, data.y_fit)
    outcome = PairExperiment(CFG).run(0, model_a, data)
    assert outcome.report.summary()["n_borrowers"] == 40
    table = outcome.scores_table()
    assert {"pair_id", "mean_proba", "top_features_a", "age"} <= set(table.columns)
    assert len(table) == 40


def _result(pair_id, matched, divergent):
    return PairResult(pair_id, 4, 42, 3, 7, matched, 0.001, 0.8, 0.8, 100, 0.9, 0.8, 90.0,
                      divergent)


def test_aggregate_uses_matched_pairs_only():
    out = MultiPairAuditor.aggregate([_result(0, True, 2.0), _result(1, True, 4.0),
                                      _result(2, False, 50.0)])
    assert out["n_pairs"] == 3 and out["n_matched_pairs"] == 2
    assert out["pct_divergent"]["mean"] == pytest.approx(3.0)
    assert out["pct_divergent"]["max"] == 4.0
    assert out["used_matched_pairs_only"] is True


def test_aggregate_falls_back_and_says_so_when_nothing_matched():
    out = MultiPairAuditor.aggregate([_result(0, False, 2.0), _result(1, False, 6.0)])
    assert out["used_matched_pairs_only"] is False
    assert out["pct_divergent"]["mean"] == pytest.approx(4.0)


def test_tolerance_sweep_reports_every_tolerance(synthetic_df):
    cfg = Config(n_estimators=20, audit_sample_size=40, n_pairs=1,
                 tolerance_sweep=(0.001, 0.005, 0.02))
    pipeline = ExperimentPipeline(cfg, "unused")
    data = pipeline.prepare(synthetic_df)
    from loan_audit.models import XGBConfig, XGBoostModel
    model_a = XGBoostModel("model_A", XGBConfig(n_estimators=20)).fit(data.X_fit, data.y_fit)
    main = PairExperiment(cfg).run(0, model_a, data)
    table = MultiPairAuditor(cfg).tolerance_sweep(main, data)
    assert list(table["tolerance"]) == [0.001, 0.005, 0.02]
    assert table["pct_divergent"].between(0, 100).all()
