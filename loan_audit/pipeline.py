"""End-to-end experiment: preprocess -> train -> match -> evaluate -> audit -> save.

Beyond the single main pair (Model A / Model B) the pipeline also
  * compares cost-sensitive weighting with SMOTE / ADASYN resampling,
  * audits several independently configured model pairs,
  * checks how sensitive the divergence figure is to the matching tolerance, top-k and
    Jaccard threshold, and
  * profiles the borrowers whose explanations diverge.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .audit import StabilityAuditor
from .config import Config
from .data import DataPreprocessor
from .divergence import DivergenceAnalyzer
from .metrics import EvaluationMetrics
from .models import LogisticBaseline, XGBConfig, XGBoostModel
from .pairs import MultiPairAuditor, PairExperiment, SplitData
from .plots import save_agreement_plots, save_pairs_plot
from .resampling import ResamplingComparison
from .synthetic import make_synthetic


def _records(df: pd.DataFrame) -> List[dict]:
    """DataFrame rows as plain dicts with NaN turned into None (valid JSON)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


class ExperimentPipeline:
    def __init__(self, config: Config = Config(), output_dir: str = "results", progress=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.progress = progress or (lambda message: None)

    def run_from_csv(self, path: str) -> dict:
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return self.run(DataPreprocessor(self.config).load(path))

    def prepare(self, df: pd.DataFrame) -> SplitData:
        pre = DataPreprocessor(self.config)
        pre.validate_schema(df)
        train_full, test = pre.split(df)
        train_fit, val = pre.split_validation(train_full)
        pre.fit_imputer(train_fit)  # medians from training data only
        train_fit, val, test = (pre.transform(d) for d in (train_fit, val, test))
        return SplitData(*pre.xy(train_fit), *pre.xy(val), *pre.xy(test))

    def run(self, df: pd.DataFrame) -> dict:
        cfg = self.config
        out = self.output_dir
        (out / "models").mkdir(parents=True, exist_ok=True)
        data = self.prepare(df)

        self.progress("Training baseline and Model A ...")
        baseline = LogisticBaseline(random_state=cfg.random_state).fit(data.X_fit, data.y_fit)
        a_config = XGBConfig(max_depth=cfg.max_depth, n_estimators=cfg.n_estimators,
                             random_state=cfg.random_state)
        model_a = XGBoostModel("model_A", a_config).fit(data.X_fit, data.y_fit)

        self.progress("Matching Model B and auditing the main pair ...")
        experiment = PairExperiment(cfg)
        main = experiment.run(0, model_a, data)
        model_b, match, report = main.model_b, main.match, main.report

        metrics = [EvaluationMetrics.from_predictions(m.model_id, data.y_test,
                                                      m.predict_proba(data.X_test))
                   for m in (baseline, model_a, model_b)]
        pd.DataFrame([m.to_dict() for m in metrics]).to_csv(out / "metrics.csv", index=False)

        scores_df = report.to_dataframe()
        scores_df.to_csv(out / "audit_scores.csv", index=False)
        save_agreement_plots(scores_df, str(out / "audit_agreement.png"))
        model_a.save(str(out / "models" / "model_A.json"))
        model_b.save(str(out / "models" / "model_B.json"))

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

        result["resampling"] = self._resampling(data, model_a)
        pooled, result["pairs"] = self._pairs(main, data)
        result["sensitivity"] = self._sensitivity(main, data)
        result["divergence"] = self._divergence(pooled)

        (out / "audit_summary.json").write_text(
            json.dumps(result, indent=2, default=_json_default))
        return result

    # ---- extended experiment ----------------------------------------------------------------

    def _resampling(self, data: SplitData, model_a: XGBoostModel) -> dict:
        if not self.config.run_resampling:
            return {"table": [], "skipped": ["resampling comparison switched off"]}
        self.progress("Comparing cost-sensitive weighting with SMOTE / ADASYN ...")
        comparison = ResamplingComparison(model_a.config, self.config.resampling_methods,
                                          self.config.random_state)
        table = comparison.run(data.X_fit, data.y_fit, data.X_test, data.y_test, model_a)
        table.to_csv(self.output_dir / "resampling_comparison.csv", index=False)
        return {"table": _records(table), "skipped": comparison.skipped}

    def _pairs(self, main, data: SplitData):
        multi = MultiPairAuditor(self.config)
        outcomes = [main] + multi.run_extra_pairs(data, self.progress)
        results = [multi.experiment.summarise(o, data) for o in outcomes]
        pairs_df = pd.DataFrame([r.__dict__ for r in results])
        pairs_df.to_csv(self.output_dir / "pairs_summary.csv", index=False)
        save_pairs_plot(pairs_df, str(self.output_dir / "pairs_agreement.png"))
        pooled = pd.concat([o.scores_table() for o in outcomes], ignore_index=True)
        pooled.to_csv(self.output_dir / "audit_scores_all_pairs.csv", index=False)
        return pooled, {"results": _records(pairs_df), "aggregate": multi.aggregate(results)}

    def _sensitivity(self, main, data: SplitData) -> dict:
        cfg = self.config
        auditor = StabilityAuditor(main.model_a, main.model_b, cfg.top_k, cfg.divergence_jaccard)
        thresholds = auditor.sensitivity(main.audited, cfg.sensitivity_ks,
                                         cfg.sensitivity_thresholds)
        thresholds.to_csv(self.output_dir / "sensitivity_thresholds.csv", index=False)
        tolerance = pd.DataFrame()
        if cfg.tolerance_sweep:
            tolerance = MultiPairAuditor(cfg).tolerance_sweep(main, data, self.progress)
            tolerance.to_csv(self.output_dir / "sensitivity_tolerance.csv", index=False)
        return {"thresholds": _records(thresholds), "tolerance": _records(tolerance)}

    def _divergence(self, pooled: pd.DataFrame) -> dict:
        profile = DivergenceAnalyzer(self.config.features, self.config.borderline_range).profile(
            pooled)
        if not profile.feature_table.empty:
            profile.feature_table.to_csv(self.output_dir / "divergence_profile.csv", index=False)
        return {"summary": profile.summary, "features": _records(profile.feature_table)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the loan default + SHAP stability audit")
    parser.add_argument("--data", help="Path to cs-training.csv")
    parser.add_argument("--synthetic", action="store_true",
                        help="Smoke test on synthetic data (not for reported results)")
    parser.add_argument("--out", default="results", help="Output folder")
    parser.add_argument("--pairs", type=int, default=Config().n_pairs,
                        help="Number of matched model pairs to audit (default: %(default)s)")
    parser.add_argument("--no-resampling", action="store_true",
                        help="Skip the SMOTE / ADASYN comparison")
    args = parser.parse_args()
    config = Config(n_pairs=args.pairs, run_resampling=not args.no_resampling)
    pipeline = ExperimentPipeline(config, output_dir=args.out, progress=print)
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
