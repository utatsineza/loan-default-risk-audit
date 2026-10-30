"""End-to-end experiment: preprocess -> train -> match -> evaluate -> audit -> save."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .audit import StabilityAuditor
from .config import Config
from .data import DataPreprocessor
from .metrics import EvaluationMetrics
from .models import LogisticBaseline, MatchedModelSelector, XGBConfig, XGBoostModel
from .plots import save_agreement_plots
from .sampling import AuditSampler
from .synthetic import make_synthetic


class ExperimentPipeline:
    def __init__(self, config: Config = Config(), output_dir: str = "results"):
        self.config = config
        self.output_dir = Path(output_dir)

    def run_from_csv(self, path: str) -> dict:
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return self.run(DataPreprocessor(self.config).load(path))

    def run(self, df: pd.DataFrame) -> dict:
        cfg = self.config
        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)

        pre = DataPreprocessor(cfg)
        pre.validate_schema(df)
        train_full, test = pre.split(df)
        train_fit, val = pre.split_validation(train_full)
        pre.fit_imputer(train_fit)  # medians from training data only
        train_fit, val, test = (pre.transform(d) for d in (train_fit, val, test))
        X_fit, y_fit = pre.xy(train_fit)
        X_val, y_val = pre.xy(val)
        X_test, y_test = pre.xy(test)

        baseline = LogisticBaseline(random_state=cfg.random_state).fit(X_fit, y_fit)
        model_a = XGBoostModel("model_A", XGBConfig(
            max_depth=cfg.max_depth, n_estimators=cfg.n_estimators,
            random_state=cfg.random_state)).fit(X_fit, y_fit)
        match = MatchedModelSelector(model_a, cfg.match_tolerance).select(
            X_fit, y_fit, X_val, y_val)
        model_b = match.model

        metrics = [EvaluationMetrics.from_predictions(m.model_id, y_test, m.predict_proba(X_test))
                   for m in (baseline, model_a, model_b)]
        pd.DataFrame([m.to_dict() for m in metrics]).to_csv(
            self.output_dir / "metrics.csv", index=False)

        mean_proba = (model_a.predict_proba(X_test) + model_b.predict_proba(X_test)) / 2
        sampler = AuditSampler(min(cfg.audit_sample_size, len(X_test)), cfg.borderline_range,
                               cfg.borderline_share, cfg.random_state)
        audit_index = sampler.sample(X_test, y_test, mean_proba)
        report = StabilityAuditor(model_a, model_b, cfg.top_k, cfg.divergence_jaccard).run(
            X_test.loc[audit_index])

        scores_df = report.to_dataframe()
        scores_df.to_csv(self.output_dir / "audit_scores.csv", index=False)
        save_agreement_plots(scores_df, str(self.output_dir / "audit_agreement.png"))
        model_a.save(str(self.output_dir / "models" / "model_A.json"))
        model_b.save(str(self.output_dir / "models" / "model_B.json"))

        result = {
            "metrics": [m.to_dict() for m in metrics],
            "matching": {
                "matched": match.matched, "reference_val_roc_auc": match.reference_auc,
                "candidate_val_roc_auc": match.candidate_auc, "difference": match.difference,
                "tolerance": cfg.match_tolerance, "candidates_tried": match.candidates_tried,
                "model_b_config": asdict(model_b.config),
            },
            "audit": report.summary(),
            "config": {k: v for k, v in asdict(cfg).items() if k != "features"},
        }
        (self.output_dir / "audit_summary.json").write_text(json.dumps(result, indent=2))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the loan default + SHAP stability audit")
    parser.add_argument("--data", help="Path to cs-training.csv")
    parser.add_argument("--synthetic", action="store_true",
                        help="Smoke test on synthetic data (not for reported results)")
    parser.add_argument("--out", default="results", help="Output folder")
    args = parser.parse_args()
    pipeline = ExperimentPipeline(output_dir=args.out)
    if args.synthetic:
        result = pipeline.run(make_synthetic(n=6000))
    elif args.data:
        result = pipeline.run_from_csv(args.data)
    else:
        parser.error("Provide --data PATH or --synthetic")
    print(json.dumps(result["audit"], indent=2))
    print(f"Outputs saved to: {args.out}/")


if __name__ == "__main__":
    main()
