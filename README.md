# Loan Default Risk Prediction with a SHAP Explanation Stability Audit

**Author:** Henriette Utatsineza (Feza) | BSc (Hons) Software Engineering, ALU | ML Track
**Supervisor:** Elvira Khwatenge

## Project overview

This project predicts whether a borrower will become seriously delinquent (90+ days late) within two years, using the public *Give Me Some Credit* dataset (Kaggle, about 150,000 borrowers). Class imbalance (about 6.7% defaults) is handled with **cost-sensitive weighting** in XGBoost rather than synthetic resampling such as SMOTE, and this choice is tested against SMOTE and ADASYN on the same data.

The research question is whether SHAP explanations can be trusted: if two differently configured XGBoost models reach the same accuracy, do they give the same reasons for the same borrower? The system trains a **Model A**, searches for a **Model B** with matching ROC-AUC but a different tree depth and random seed, and measures how much their SHAP explanations agree (Spearman rank correlation and top-k Jaccard index) on a sample of borrowers. The whole audit is repeated over **several independent model pairs**, and its headline figure is tested for sensitivity to the choices an analyst has to make.

**Links**
- Walkthrough video: `[ADD LINK]`
- Technical report (PDF): `[ADD LINK]`

## Key definitions

**Matched accuracy.** Model B is *matched* to Model A when the absolute difference of their ROC-AUC on a **validation set** is at most **0.005** (`Config.match_tolerance`). ROC-AUC is used because it does not depend on a decision threshold, so it compares the models' ranking quality directly. 0.005 is small next to the gap between the logistic baseline and XGBoost (about 0.065), so matched models are practically indistinguishable in accuracy. The test set is never used for matching. If no candidate meets the tolerance, the closest is returned and flagged as *not matched*, and unmatched pairs are left out of the cross-pair summary. The tolerance is compared at 0.002, 0.005 and 0.01 (`sensitivity_tolerance.csv`).

**Divergent case.** A borrower is *divergent* when the **Jaccard index of the two models' top-3 SHAP features is below 0.5**, meaning they share at most one of their three most influential features (`Config.top_k`, `Config.divergence_jaccard`). With k=3 the Jaccard index can only take the values 0, 0.2, 0.5 and 1.0, so the figure is sensitive to where the line is drawn. The share of divergent cases is therefore also reported for k = 2, 3, 5 and thresholds 0.3, 0.5, 0.7 and 1.0 (any difference at all) in `sensitivity_thresholds.csv`, and should be read together with the mean Spearman and top-feature agreement.

## Features

| Feature | Where |
|---|---|
| Data validation, stratified splits, leak-free median imputation | `loan_audit/data.py` |
| Baseline (logistic regression) and cost-sensitive XGBoost models | `loan_audit/models.py` |
| Matched-accuracy search for Model B | `MatchedModelSelector` in `loan_audit/models.py` |
| Cost-sensitive weighting vs SMOTE / ADASYN comparison | `loan_audit/resampling.py` |
| PR-AUC, ROC-AUC, F1, precision, recall | `loan_audit/metrics.py` |
| SHAP explanations as value objects | `loan_audit/explain.py` |
| Audit sampling with borderline-case oversampling | `loan_audit/sampling.py` |
| Stability audit (Spearman, Jaccard, divergent cases, sensitivity grid) | `loan_audit/audit.py` |
| Audit of many model pairs, tolerance sweep | `loan_audit/pairs.py` |
| Profile of divergent borrowers (RQ3) | `loan_audit/divergence.py` |
| End-to-end pipeline and command line | `loan_audit/pipeline.py` |
| Web interface: predict and explain a borrower; browse all audit results | `app.py` (Streamlit) |
| Performance and environment benchmark | `scripts/benchmark.py` |
| Unit, integration and UI tests | `tests/` (pytest) |

## Technical requirements

- Python 3.10 or newer. Tested on Python 3.12.3 (Linux) and Python 3.13.2 (Windows 11, 8 cores).
- Libraries (see `requirements.txt`). Windows 11 run: xgboost 3.4.1, shap 0.54.1, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, streamlit 1.66.1, pytest 9.2.0, imbalanced-learn (for SMOTE / ADASYN)
- Hardware: runs on CPU only; peak memory of about 250 MB was measured for the benchmark on Windows 11
- Browser (for the web interface): any current browser

## Installation

```bash
git clone https://github.com/utatsineza/loan-default-risk-audit.git
cd loan-default-risk-audit
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the data

Download `cs-training.csv` from [Kaggle: Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) and place it in the repository root. The dataset is not redistributed in this repository, so please check Kaggle's data terms.

## Usage

```bash
# 1. Run the full experiment (trains models, matches Model B, compares resampling,
#    audits several model pairs, runs the sensitivity study, saves outputs)
python -m loan_audit.pipeline --data cs-training.csv --out results

#    Options: --pairs 1 audits only the main pair; --no-resampling skips SMOTE / ADASYN
#    (the full run takes several minutes because it trains many XGBoost models)

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
| `audit_summary.json` | Everything below in one file: model matching, audit summary, resampling, pairs, sensitivity, divergence profile |
| `audit_scores.csv` | Agreement scores for each audited borrower of the main pair |
| `audit_agreement.png` | Histograms of the main pair's agreement scores |
| `resampling_comparison.csv` | Cost-sensitive vs no handling vs SMOTE vs ADASYN: PR-AUC, ROC-AUC, F1, best F1, training size and time |
| `pairs_summary.csv`, `pairs_agreement.png` | Matching and audit results for every model pair |
| `audit_scores_all_pairs.csv` | Per-borrower agreement scores and features, pooled over all pairs |
| `sensitivity_thresholds.csv` | % divergent for each top-k and Jaccard threshold |
| `sensitivity_tolerance.csv` | Audit results for different matching tolerances |
| `divergence_profile.csv` | Divergent vs agreeing borrowers, feature by feature |
| `models/model_A.json`, `models/model_B.json` | Trained XGBoost models |
| `benchmark.json` | Speed, memory and environment from `scripts/benchmark.py` |

## Repository structure

```
loan-default-risk-audit/
├── app.py                     Streamlit interface
├── loan_audit/                Core package
│   ├── config.py              Settings in one place
│   ├── data.py                DataPreprocessor
│   ├── models.py              BaseModel, LogisticBaseline, XGBoostModel, MatchedModelSelector
│   ├── resampling.py          ResamplingComparison (SMOTE / ADASYN vs cost-sensitive)
│   ├── metrics.py             EvaluationMetrics
│   ├── explain.py             SHAPExplainer, SHAPResult
│   ├── sampling.py            AuditSampler
│   ├── audit.py               StabilityAuditor, AgreementScore, AuditReport, metric functions
│   ├── pairs.py               PairExperiment, MultiPairAuditor, SplitData, PairResult
│   ├── divergence.py          DivergenceAnalyzer
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
- **Separation of concerns:** loading, modelling, resampling, metrics, explanation, sampling, auditing, multi-pair orchestration, divergence analysis and the pipeline each live in their own module; `Config` holds every threshold.
- **Reuse:** `PairExperiment` runs one matched-pair audit and is used for the main pair, the extra pairs and the tolerance sweep alike.
- **Value objects:** `SHAPResult`, `AgreementScore`, `EvaluationMetrics`, `PairResult` and `XGBConfig` are immutable dataclasses.

## Algorithms

1. **Cost-sensitive weighting:** `scale_pos_weight` is set to the ratio of non-defaults to defaults in the training data, so missing a default costs more during training. No synthetic data is created.
2. **Resampling comparison:** the same XGBoost settings are trained with no imbalance handling, with SMOTE and with ADASYN (applied to the training data only), and all are scored on the same held-out test set. Because F1 at a fixed 0.5 threshold is not comparable across strategies, the table also reports the best F1 over all thresholds, alongside the threshold-free PR-AUC and ROC-AUC.
3. **Matched-accuracy selection:** candidate configurations that differ from Model A in both depth and seed are trained in turn and compared on a validation set. The first with ROC-AUC within the tolerance of Model A is accepted; if none qualifies, the closest is returned and flagged as not matched.
4. **Many model pairs:** the main pair uses depth 4 and seed 42. Further pairs start from a Model A with a different depth and seed, each matched to its own Model B, and are audited on their own sample. Cross-pair means, spreads and ranges use matched pairs only.
5. **Audit sampling:** half of the audit sample comes from borderline cases (average predicted score between 0.4 and 0.6); the rest is filled in proportion to the true label, with no borrower chosen twice.
6. **Agreement metrics:** for each borrower, Spearman correlation between the two models' absolute SHAP values, and the Jaccard index of their top-3 features.
7. **Sensitivity study:** SHAP values are computed once and the share of divergent cases is recomputed over a grid of top-k sizes and Jaccard thresholds; the matching and audit are repeated at three tolerances.
8. **Divergence profile:** divergent and agreeing borrowers (pooled over pairs) are compared on predicted score, share of borderline cases and each feature (standardised difference and a Mann-Whitney test, reported as exploratory), and the features most often in one model's top-k but not the other's are counted.

## Testing

`python -m pytest` runs 62 tests:
- **Unit (50):** data validation and imputation, metrics, sampling, model behaviour, agreement metrics (including edge cases such as constant profiles and invalid inputs), the resampling comparison, multi-pair aggregation and the tolerance sweep, the divergence profile and the sensitivity grid
- **Integration (6):** the whole pipeline writes every output, the extended outputs are consistent with the settings, the single-pair and no-resampling modes work, and the pipeline fails clearly on missing files and malformed data
- **Functional/UI (6):** the Streamlit app is driven headlessly through prediction, extreme inputs, the audit page including the extended sections, and missing-file states

## Limitations and ethics

- This is a research prototype, not a system for real lending decisions.
- The dataset is a 2011 competition dataset with no documented Rwandan origin, so findings are not claimed to generalise to Rwandan borrowers.
- The model's scores come from cost-sensitive training and are not calibrated probabilities.
- The dataset includes age, a sensitive attribute in credit decisions; results should be reviewed by age group before any wider use.
- The divergence figure depends on the analyst's choices (top-k, Jaccard threshold, matching tolerance); the sensitivity study shows how much.
- Divergent cases are few, so the divergence profile is exploratory and its p-values are not corrected for multiple comparisons.

## Acknowledgements

Dataset: Kaggle, *Give Me Some Credit* (2011). Libraries: XGBoost, SHAP, scikit-learn, imbalanced-learn, pandas, NumPy, SciPy, matplotlib, Streamlit.
