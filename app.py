"""Streamlit demo interface: predict + explain one borrower, and browse the stability audit.

Run from the repository root:  streamlit run app.py
Set LOAN_RESULTS_DIR to read results from somewhere other than ./results.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from loan_audit.config import FEATURES
from loan_audit.explain import SHAPExplainer
from loan_audit.models import XGBoostModel

RISK_THRESHOLD = 0.5

st.set_page_config(page_title="Loan Default Risk Predictor", layout="centered")


def results_dir() -> Path:
    return Path(os.environ.get("LOAN_RESULTS_DIR", "results"))


@st.cache_resource
def load_model_and_explainer(model_path: str, modified_time: float):
    """Cached per file version, so re-running the pipeline refreshes the app."""
    model = XGBoostModel.load(model_path, "model_A")
    return model, SHAPExplainer(model)


def build_borrower(values: dict) -> pd.DataFrame:
    return pd.DataFrame([values])[list(FEATURES)]


def render_predict_page() -> None:
    st.title("Loan Default Risk Predictor")
    st.caption("XGBoost with cost-sensitive class weighting, explained with SHAP.")
    st.info("Demo only: please enter hypothetical values, not real personal or financial information.")

    model_path = results_dir() / "models" / "model_A.json"
    if not model_path.exists():
        st.error(f"Model file not found at '{model_path}'. Run "
                 "`python -m loan_audit.pipeline --data cs-training.csv` first.")
        return

    st.subheader("Borrower information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 35)
        income = st.number_input("Monthly income (USD)", 0, 1_000_000, 3000, step=100)
        debt_ratio = st.slider("Debt ratio", 0.0, 5.0, 0.3, step=0.01)
        utilization = st.slider("Revolving credit utilization", 0.0, 3.0, 0.3, step=0.01)
        dependents = st.number_input("Number of dependents", 0, 20, 0)
    with col2:
        open_lines = st.number_input("Open credit lines/loans", 0, 60, 5)
        real_estate = st.number_input("Real estate loans/lines", 0, 30, 1)
        late_30 = st.number_input("Times 30-59 days past due", 0, 30, 0)
        late_60 = st.number_input("Times 60-89 days past due", 0, 30, 0)
        late_90 = st.number_input("Times 90+ days late", 0, 30, 0)

    if not st.button("Predict default risk", type="primary"):
        return

    borrower = build_borrower({
        "RevolvingUtilizationOfUnsecuredLines": utilization, "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30, "DebtRatio": debt_ratio,
        "MonthlyIncome": income, "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": late_90, "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60, "NumberOfDependents": dependents,
    })
    model, explainer = load_model_and_explainer(str(model_path), model_path.stat().st_mtime)
    probability = float(model.predict_proba(borrower)[0])
    high_risk = probability >= RISK_THRESHOLD

    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Default risk score", f"{probability:.1%}")
    c2.metric("Classification", "HIGH RISK" if high_risk else "LOW RISK")
    st.caption("The model is trained with cost-sensitive weighting, so this score is not a "
               "calibrated probability. Use it to rank risk, not as an exact chance of default.")

    st.subheader("Why this prediction? (SHAP explanation)")
    result = explainer.explain_one(borrower)
    explanation = shap.Explanation(values=result.values, base_values=result.base_value,
                                   data=borrower.iloc[0].values,
                                   feature_names=list(result.feature_names))
    plt.close("all")
    shap.plots.waterfall(explanation, show=False)
    figure = plt.gcf()
    figure.set_size_inches(9, 5)
    st.pyplot(figure)
    plt.close(figure)


def render_audit_page() -> None:
    st.title("Explanation Stability Audit")
    st.caption("Do two equally accurate models give the same reasons for the same borrower?")
    summary_path = results_dir() / "audit_summary.json"
    if not summary_path.exists():
        st.info("No audit results yet. Run `python -m loan_audit.pipeline --data cs-training.csv`.")
        return

    summary = json.loads(summary_path.read_text())
    audit, matching = summary["audit"], summary["matching"]

    st.subheader("Model matching")
    st.write(f"Model B was {'matched' if matching['matched'] else 'NOT matched'} to Model A "
             f"(validation ROC-AUC difference {matching['difference']:.4f}, "
             f"tolerance {matching['tolerance']}), after {matching['candidates_tried']} candidate(s).")

    st.subheader(f"Agreement across {audit['n_borrowers']} audited borrowers")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Spearman", f"{audit['mean_spearman']:.2f}")
    c2.metric(f"Mean Jaccard (top {audit['top_k']})", f"{audit['mean_jaccard_top_k']:.2f}")
    c3.metric("Divergent cases", f"{audit['pct_divergent']:.1f}%")
    st.write(f"Same most-important feature in {audit['pct_top_feature_match']:.1f}% of cases.")

    plot_path = results_dir() / "audit_agreement.png"
    if plot_path.exists():
        st.image(str(plot_path))

    metrics_path = results_dir() / "metrics.csv"
    if metrics_path.exists():
        st.subheader("Model performance on the held-out test set")
        st.dataframe(pd.read_csv(metrics_path).round(4), hide_index=True)

    scores_path = results_dir() / "audit_scores.csv"
    if scores_path.exists():
        divergent = pd.read_csv(scores_path).query("divergent")
        st.subheader(f"Divergent borrowers ({len(divergent)})")
        st.dataframe(divergent.round(3), hide_index=True)

    render_extended_results(summary)


def read_csv_if_present(name: str):
    path = results_dir() / name
    return pd.read_csv(path) if path.exists() else None


def render_extended_results(summary: dict) -> None:
    """Resampling comparison, many model pairs, sensitivity and divergence profile."""
    resampling = read_csv_if_present("resampling_comparison.csv")
    if resampling is not None:
        st.subheader("Cost-sensitive weighting vs resampling")
        st.caption("Same XGBoost settings and the same test set; only the way class imbalance "
                   "is handled differs. PR-AUC, ROC-AUC and best F1 do not depend on the 0.5 "
                   "decision threshold, so they are the fair comparison.")
        st.dataframe(resampling.round(4), hide_index=True)
        for note in summary.get("resampling", {}).get("skipped", []):
            st.warning(f"Skipped: {note}")

    pairs = read_csv_if_present("pairs_summary.csv")
    if pairs is not None:
        aggregate = summary.get("pairs", {}).get("aggregate", {})
        st.subheader(f"Stability across {len(pairs)} model pair(s)")
        if aggregate:
            c1, c2, c3 = st.columns(3)
            div = aggregate["pct_divergent"]
            c1.metric("Pairs meeting the tolerance", f"{aggregate['n_matched_pairs']} of {aggregate['n_pairs']}")
            c2.metric("Mean Spearman (across pairs)", f"{aggregate['mean_spearman']['mean']:.2f}")
            c3.metric("Divergent cases (range)", f"{div['min']:.1f}% to {div['max']:.1f}%")
            if not aggregate.get("used_matched_pairs_only", True):
                st.warning("No pair met the matching tolerance, so all pairs are summarised.")
        st.dataframe(pairs.round(4), hide_index=True)
        plot = results_dir() / "pairs_agreement.png"
        if plot.exists():
            st.image(str(plot))

    thresholds = read_csv_if_present("sensitivity_thresholds.csv")
    tolerance = read_csv_if_present("sensitivity_tolerance.csv")
    if thresholds is not None or tolerance is not None:
        st.subheader("How sensitive is the divergence figure?")
    if thresholds is not None:
        st.caption("Share of audited borrowers flagged as divergent for different top-k sizes "
                   "and Jaccard thresholds (a threshold of 1.0 flags any difference).")
        st.dataframe(thresholds.pivot(index="top_k", columns="jaccard_threshold",
                                      values="pct_divergent").round(1))
    if tolerance is not None:
        st.caption("Effect of how strictly Model B must match Model A's accuracy.")
        st.dataframe(tolerance.round(4), hide_index=True)

    profile = read_csv_if_present("divergence_profile.csv")
    info = summary.get("divergence", {}).get("summary", {})
    if info:
        st.subheader("What characterises divergent borrowers?")
        if "mean_score_divergent" in info:
            c1, c2 = st.columns(2)
            c1.metric("Mean score: divergent vs agreeing",
                      f"{info['mean_score_divergent']:.2f} vs {info['mean_score_agreeing']:.2f}")
            c2.metric("In the 0.4-0.6 band",
                      f"{info['pct_borderline_divergent']:.0f}% vs {info['pct_borderline_agreeing']:.0f}%")
            contested = ", ".join(f"{c['feature']} ({c['cases']})"
                                  for c in info.get("most_contested_features", []))
            st.write(f"Based on {info['n_divergent']} divergent and {info['n_agreeing']} agreeing "
                     f"cases pooled over all pairs. Features most often in one model's top-k "
                     f"but not the other's: {contested}.")
        else:
            st.info(info.get("note", "Not enough divergent cases to compare."))
        if profile is not None:
            st.caption("Exploratory comparison: standardised difference in each feature between "
                       "divergent and agreeing borrowers. p-values are not corrected for "
                       "multiple comparisons.")
            st.dataframe(profile.round(4), hide_index=True)


def main() -> None:
    page = st.sidebar.radio("Page", ["Predict & Explain", "Stability Audit Results"])
    if page == "Predict & Explain":
        render_predict_page()
    else:
        render_audit_page()


main()
