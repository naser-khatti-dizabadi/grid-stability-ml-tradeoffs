#!/usr/bin/env python3
"""
Reproduce the efficiency-benchmark methodology.

Important:
Exact latency and serialized size can vary with hardware, operating system,
Python/scikit-learn/joblib versions, BLAS/OpenMP runtime, and serialization
details. The archived manuscript benchmark remains the paper reference.

Protocol:
- Frozen 80/20 stratified split (random_state=42)
- Final selected model configurations
- Full 2,000-sample held-out test batch
- One warm-up prediction
- 20 timed prediction repetitions
- Median latency and IQR
- Numerical thread pools limited to one thread during timing
- RF and KNN configured with n_jobs=1
- Full fitted pipeline serialized with joblib

Usage:
    python src/05_efficiency_benchmark.py /path/to/Data_for_UCI_named.csv
"""

from pathlib import Path
import argparse
import tempfile
import time
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["tau1","tau2","tau3","tau4","p2","p3","p4","g1","g2","g3","g4"]
N_TIMED_REPS = 20

ARCHIVED = {
    "LR":      {"median_ms":0.763,  "iqr_ms":0.155, "size_kb":1.869},
    "KNN":     {"median_ms":53.737, "iqr_ms":1.039, "size_kb":751.769},
    "RBF-SVM": {"median_ms":66.492, "iqr_ms":1.552, "size_kb":135.803},
    "RF":      {"median_ms":45.764, "iqr_ms":0.543, "size_kb":22047.050},
}

def build_models():
    return {
        "LR": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=1000)),
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=15, weights="distance", n_jobs=1
            )),
        ]),
        "RBF-SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(C=10, gamma=0.05, kernel="rbf", probability=True)),
        ]),
        "RF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=None, min_samples_leaf=1,
                random_state=42, n_jobs=1
            )),
        ]),
    }

def serialized_size_kb(model):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "model.joblib"
        # No compression: measures the full serialized fitted pipeline.
        joblib.dump(model, p, compress=0)
        return p.stat().st_size / 1024.0

def benchmark_prediction(model, X_test):
    # Warm-up is intentionally excluded from timing.
    with threadpool_limits(limits=1):
        model.predict(X_test)

        times_ms = []
        for _ in range(N_TIMED_REPS):
            t0 = time.perf_counter()
            model.predict(X_test)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    q1, q3 = np.percentile(times_ms, [25, 75])
    return float(np.median(times_ms)), float(q3 - q1), times_ms

def main(csv_path):
    df = pd.read_csv(csv_path)
    X = df[FEATURES].copy()
    y = (df["stabf"] == "unstable").astype(int)

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    assert len(X_test) == 2000

    print("=== Runtime environment ===")
    print(f"Python: {platform.python_version()}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"joblib: {joblib.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Timed batch size: {len(X_test):,}")
    print(f"Timed repetitions: {N_TIMED_REPS}")
    print("Thread-pool limit during timing: 1")

    rows = []
    models = build_models()

    print("\n=== Current-environment benchmark ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        median_ms, iqr_ms, times_ms = benchmark_prediction(model, X_test)
        size_kb = serialized_size_kb(model)

        archived = ARCHIVED[name]
        rows.append({
            "model": name,
            "batch_size": len(X_test),
            "timed_repetitions": N_TIMED_REPS,
            "median_latency_ms_current": median_ms,
            "iqr_latency_ms_current": iqr_ms,
            "serialized_size_kb_current": size_kb,
            "archived_median_latency_ms": archived["median_ms"],
            "archived_iqr_latency_ms": archived["iqr_ms"],
            "archived_serialized_size_kb": archived["size_kb"],
        })

        print(
            f"{name:7s}: current median={median_ms:.3f} ms, "
            f"IQR={iqr_ms:.3f} ms, size={size_kb:.3f} KB | "
            f"archived median={archived['median_ms']:.3f} ms, "
            f"IQR={archived['iqr_ms']:.3f} ms, size={archived['size_kb']:.3f} KB"
        )

    out = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "efficiency_benchmark_current_environment.csv"
    out.to_csv(out_path, index=False)

    print("\nPASS: the archived efficiency-benchmark methodology was reproduced.")
    print("NOTE: Exact timing/size equality is not asserted because these quantities are environment/version dependent.")
    print("The archived controlled benchmark remains the manuscript reference.")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    main(args.csv_path)
