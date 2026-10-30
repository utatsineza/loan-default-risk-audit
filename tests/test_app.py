"""Functional / UI tests: drive the Streamlit app headlessly through its main states."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from loan_audit.config import Config  # noqa: E402
from loan_audit.pipeline import ExperimentPipeline  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def results_folder(tmp_path_factory, synthetic_df):
    folder = tmp_path_factory.mktemp("results")
    ExperimentPipeline(Config(n_estimators=25, audit_sample_size=50), str(folder)).run(synthetic_df)
    return str(folder)


def open_app(monkeypatch, folder):
    monkeypatch.setenv("LOAN_RESULTS_DIR", folder)
    return AppTest.from_file(APP_PATH, default_timeout=90).run()


def test_predict_page_shows_score_and_classification(monkeypatch, results_folder):
    app = open_app(monkeypatch, results_folder)
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert [m.label for m in app.metric] == ["Default risk score", "Classification"]
    assert app.metric[1].value in ("LOW RISK", "HIGH RISK")


def test_extreme_inputs_do_not_crash(monkeypatch, results_folder):
    app = open_app(monkeypatch, results_folder)
    for widget in app.number_input:
        if widget.label == "Times 90+ days late":
            widget.set_value(30)
    app.button[0].click().run()
    assert not app.exception


def test_audit_page_shows_agreement_metrics(monkeypatch, results_folder):
    app = open_app(monkeypatch, results_folder)
    app.sidebar.radio[0].set_value("Stability Audit Results").run()
    assert not app.exception
    assert "Mean Spearman" in [m.label for m in app.metric]


def test_missing_model_shows_friendly_error(monkeypatch, tmp_path):
    app = open_app(monkeypatch, str(tmp_path / "empty"))
    assert not app.exception and app.error


def test_missing_audit_results_shows_info(monkeypatch, tmp_path):
    app = open_app(monkeypatch, str(tmp_path / "empty"))
    app.sidebar.radio[0].set_value("Stability Audit Results").run()
    assert not app.exception and app.info
