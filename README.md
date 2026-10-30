# Loan Default Risk Prediction with a SHAP Explanation Stability Audit

**Author:** Henriette Utatsineza (Feza) | BSc (Hons) Software Engineering, ALU | ML Track
**Supervisor:** Elvira Khwatenge

## Project overview

This project predicts whether a borrower will become seriously delinquent (90+ days late) within two years, using the public *Give Me Some Credit* dataset (Kaggle, about 150,000 borrowers). Class imbalance (about 6.7% defaults) is handled with **cost-sensitive weighting** in XGBoost rather than synthetic resampling such as SMOTE.

The research question is whether SHAP explanations can be trusted: if two differently configured XGBoost models reach the same accuracy, do they give the same reasons for the same borrower? The system trains **Model A**, searches for a **Model B** with matching ROC-AUC but a different tree depth and random seed, and measures how much their SHAP explanations agree (Spearman rank correlation and top-k Jaccard index) on a sample of borrowers.

**Links**
- Walkthrough video: `[ADD LINK]`
- Technical report (PDF): `[ADD LINK]`
- Live demo (optional): `[ADD LINK IF DEPLOYED]`

## Features

| Feature | Where |
|---|---|
| Data validation, stratified splits, leak-free median imputation | `loan_audit/data.py` |
| Baseline (logistic regression) and cost-sensitive XGBoost models | `loan_audit/models.py` |
| Matched-accuracy search for Model B | `MatchedModelSelector` in `loan_audit/models.py` |
| PR-AUC, ROC-AUC, F1, precision, recall | `loan_audit/metrics.py` |
| SHAP explanations as value objects | `loan_audit/explain.py` |
| Audit sampling with borderline-case oversampling | `loan_audit/sampling.py` |
| Stability audit (Spearman, Jaccard, divergent cases) | `loan_audit/audit.py` |
| End-to-end pipeline and command line | `loan_audit/pipeline.py` |
| Web interface: predict and explain a borrower; browse audit results | `app.py` (Streamlit) |
| Performance and environment benchmark | `scripts/benchmark.py` |
| Unit, integration and UI tests | `tests/` (pytest) |

## Technical requirements

- Python 3.10 or newer (tested on 3.12.3)
- Libraries (see `requirements.txt`). Tested with: xgboost 3.4.1, shap 0.54.1, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, streamlit (latest at test time), pytest
- Hardware: runs on CPU only; the full dataset needs roughly 1 GB of RAM
- Browser (for the web interface): any current browser

## Installation

```bash
git clone [REPO URL]
cd loan-default-risk-audit
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the data

Download `cs-training.csv` from [Kaggle: Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) and place it in the repository root. The dataset is not redistributed in this repository, so please check Kaggle's data terms.

## Usage

```bash
# 1. Run the full experiment (trains models, matches Model B, runs the audit, saves outputs)
python -m loan_audit.pipeline --data cs-training.csv --out results

# 2. Open the web interface (reads ./results)
streamlit run app.py

# 3. Run the tests
python -m pytest

# 4. Benchmark speed and record the environment
python scripts/benchmark.py --data cs-training.csv
```

Quick check of the code without the dataset: `python -m loan_audit.pipeline --synthetic --out /tmp/smoke`. This uses made-up data and must not be used for reported results.

### Outputs written to `results/`

| File | Contents |
|---|---|
| `metrics.csv` | PR-AUC, ROC-AUC, F1, precision, recall for the baseline, Model A and Model B on the test set |
| `audit_summary.json` | Model matching details and audit summary (mean Spearman, mean Jaccard, % divergent) |
| `audit_scores.csv` | Agreement scores for each audited borrower |
| `audit_agreement.png` | Histograms of the agreement scores |
| `models/model_A.json`, `models/model_B.json` | Trained XGBoost models |

## Repository structure

```
loan-default-risk-audit/
├── app.py                     Streamlit interface
├── loan_audit/                Core package
│   ├── config.py              Settings in one place
│   ├── data.py                DataPreprocessor
│   ├── models.py              BaseModel, LogisticBaseline, XGBoostModel, MatchedModelSelector
│   ├── metrics.py             EvaluationMetrics
│   ├── explain.py             SHAPExplainer, SHAPResult
│   ├── sampling.py            AuditSampler
│   ├── audit.py               StabilityAuditor, AgreementScore, AuditReport, metric functions
│   ├── pipeline.py            ExperimentPipeline and CLI
│   ├── plots.py               Audit plots
│   └── synthetic.py           Synthetic data for tests and smoke runs only
├── tests/                     pytest suite (unit, integration, UI)
├── scripts/benchmark.py       Performance benchmark
├── notebooks/                 Initial exploration notebook (exploratory, not the code of record)
├── results/                   Pipeline outputs
└── requirements.txt
```

## Design notes

- **Abstraction and polymorphism:** `BaseModel` defines `fit` and `predict_proba`; `LogisticBaseline` and `XGBoostModel` implement it, so the pipeline treats every model alike.
- **Encapsulation:** `DataPreprocessor` keeps the learned medians internally and refuses to transform before fitting; `XGBoostModel` guards access to the unfitted model.
- **Separation of concerns:** loading, modelling, metrics, explanation, sampling, auditing and orchestration each live in their own module; `Config` holds every threshold.
- **Value objects:** `SHAPResult`, `AgreementScore` and `EvaluationMetrics` are immutable dataclasses.
- The classes follow the class diagram in the thesis (Chapter 3).

## Algorithms

1. **Cost-sensitive weighting:** `scale_pos_weight` is set to the ratio of non-defaults to defaults in the training data, so missing a default costs more during training. No synthetic data is created.
2. **Matched-accuracy selection:** candidate configurations that differ from Model A in both depth and seed are trained in turn and compared on a validation set. The first with ROC-AUC within 0.005 of Model A is accepted; if none qualifies, the closest is returned and flagged as not matched. The test set is never used for matching.
3. **Audit sampling:** half of the audit sample comes from borderline cases (average predicted score between 0.4 and 0.6); the rest is filled in proportion to the true label, with no borrower chosen twice.
4. **Agreement metrics:** for each borrower, Spearman correlation between the two models' absolute SHAP values, and the Jaccard index of their top-3 features. A case is divergent when the Jaccard index is below 0.5.

## Testing

`python -m pytest` runs 40 tests:
- **Unit:** data validation and imputation, metrics, sampling, model behaviour, agreement metrics (including edge cases such as constant profiles and invalid inputs)
- **Integration:** the whole pipeline writes every output, and fails clearly on missing files and malformed data
- **Functional/UI:** the Streamlit app is driven headlessly through prediction, extreme inputs, the audit page, and missing-file states

## Limitations and ethics

- This is a research prototype, not a system for real lending decisions.
- The dataset is a 2011 competition dataset with no documented Rwandan origin, so findings are not claimed to generalise to Rwandan borrowers.
- The model's scores come from cost-sensitive training and are not calibrated probabilities.
- The dataset includes age, a sensitive attribute in credit decisions; results should be reviewed by age group before any wider use.

## Acknowledgements

Dataset: Kaggle, *Give Me Some Credit* (2011). Libraries: XGBoost, SHAP, scikit-learn, pandas, NumPy, SciPy, matplotlib, Streamlit.
