"""Performance and environment benchmark (evidence for the performance/compatibility section).

Usage: python scripts/benchmark.py [--data cs-training.csv] [--out results/benchmark.json]
Without --data it uses synthetic data (fine for checking the script, not for reported numbers).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loan_audit.config import Config  # noqa: E402
from loan_audit.data import DataPreprocessor  # noqa: E402
from loan_audit.explain import SHAPExplainer  # noqa: E402
from loan_audit.models import XGBConfig, XGBoostModel  # noqa: E402
from loan_audit.synthetic import make_synthetic  # noqa: E402


def peak_memory_mb() -> float:
    """Peak memory of this process in MB, on Windows, macOS and Linux."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / (1024 * 1024)

    import resource  # Unix only

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def timed(function, repeats: int = 1):
    start = time.perf_counter()
    for _ in range(repeats):
        output = function()
    return output, (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--out", default="results/benchmark.json")
    args = parser.parse_args()

    cfg = Config()
    pre = DataPreprocessor(cfg)
    df = pre.load(args.data) if args.data else make_synthetic(n=20000)
    train, test = pre.split(df)
    pre.fit_imputer(train)
    X_train, y_train = pre.xy(pre.transform(train))
    X_test, _ = pre.xy(pre.transform(test))

    model, fit_seconds = timed(lambda: XGBoostModel("bench", XGBConfig()).fit(X_train, y_train))
    _, single_predict = timed(lambda: model.predict_proba(X_test.iloc[[0]]), repeats=200)
    _, batch_predict = timed(lambda: model.predict_proba(X_test))
    explainer, explainer_build = timed(lambda: SHAPExplainer(model))
    _, single_shap = timed(lambda: explainer.explain_one(X_test.iloc[[0]]), repeats=50)
    sample = X_test.head(300)
    _, batch_shap = timed(lambda: explainer.explain(sample))

    report = {
        "data_source": args.data or "synthetic (not for reporting)",
        "train_rows": len(X_train),
        "fit_seconds": round(fit_seconds, 3),
        "single_prediction_ms": round(single_predict * 1000, 3),
        "batch_prediction_rows_per_second": round(len(X_test) / batch_predict),
        "shap_explainer_build_seconds": round(explainer_build, 3),
        "single_shap_explanation_ms": round(single_shap * 1000, 3),
        "shap_300_borrowers_seconds": round(batch_shap, 3),
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "environment": {
            "os": platform.platform(), "python": platform.python_version(),
            "cpu_count": __import__("os").cpu_count(), "xgboost": xgboost.__version__,
            "shap": shap.__version__, "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()