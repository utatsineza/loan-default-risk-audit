"""Central configuration so no magic numbers are scattered through the code."""
from dataclasses import dataclass

TARGET = "SeriousDlqin2yrs"
FEATURES = (
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
)


@dataclass(frozen=True)
class Config:
    """Experiment settings (defaults follow thesis Chapter 3)."""

    target: str = TARGET
    features: tuple = FEATURES
    test_size: float = 0.2          # stratified 80:20 train/test split
    val_size: float = 0.2           # share of the training set used only to match Model B to Model A
    random_state: int = 42
    n_estimators: int = 200         # Model A
    max_depth: int = 4              # Model A
    match_tolerance: float = 0.005  # max ROC-AUC gap between Model A and Model B on validation data
    audit_sample_size: int = 300
    borderline_range: tuple = (0.4, 0.6)
    borderline_share: float = 0.5   # share of the audit sample drawn from borderline cases
    top_k: int = 3                  # features compared by the Jaccard index
    divergence_jaccard: float = 0.5  # Jaccard below this marks a case as divergent

    # --- extended experiment (supervisor feedback) ---
    n_pairs: int = 5                 # matched model pairs audited (1 = the single main pair only)
    tolerance_sweep: tuple = (0.002, 0.005, 0.01)   # matching tolerances compared in the sensitivity study
    sensitivity_ks: tuple = (2, 3, 5)                # top-k sizes compared in the sensitivity study
    sensitivity_thresholds: tuple = (0.3, 0.5, 0.7, 1.0)  # Jaccard thresholds for "divergent"
    run_resampling: bool = True      # SMOTE / ADASYN comparison against cost-sensitive weighting
    resampling_methods: tuple = ("smote", "adasyn")
