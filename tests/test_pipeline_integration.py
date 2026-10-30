"""Integration tests: the whole pipeline, plus error handling."""
import json

import pytest

from loan_audit.config import Config
from loan_audit.pipeline import ExperimentPipeline


def test_pipeline_end_to_end_writes_all_outputs(synthetic_df, tmp_path):
    cfg = Config(n_estimators=25, audit_sample_size=60)
    result = ExperimentPipeline(cfg, str(tmp_path)).run(synthetic_df)

    for name in ("metrics.csv", "audit_scores.csv", "audit_summary.json", "audit_agreement.png",
                 "models/model_A.json", "models/model_B.json"):
        assert (tmp_path / name).exists(), f"missing output {name}"
    assert result["audit"]["n_borrowers"] == 60
    assert 0 <= result["audit"]["mean_jaccard_top_k"] <= 1
    assert "matched" in result["matching"]
    assert json.loads((tmp_path / "audit_summary.json").read_text())["audit"]["n_borrowers"] == 60


def test_pipeline_reports_three_models(synthetic_df, tmp_path):
    result = ExperimentPipeline(Config(n_estimators=25, audit_sample_size=40), str(tmp_path)).run(
        synthetic_df)
    assert [m["model_id"] for m in result["metrics"]] == ["baseline_lr", "model_A", "model_B"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExperimentPipeline(output_dir=str(tmp_path)).run_from_csv(str(tmp_path / "nope.csv"))


def test_malformed_csv_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n3,4\n")
    with pytest.raises(ValueError):
        ExperimentPipeline(output_dir=str(tmp_path)).run_from_csv(str(bad))
