#!/usr/bin/env python3
"""
Reproduce the frozen train/test split, selected-model cross-validation,
and clean held-out test performance for the grid-stability study.

Usage:
    python src/02_train_evaluate.py /path/to/Data_for_UCI_named.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

FEATURES = ["tau1","tau2","tau3","tau4","p2","p3","p4","g1","g2","g3","g4"]

EXPECTED_CV = {
    "LR": 0.858369,
    "KNN": 0.896251,
    "RBF-SVM": 0.970828,
    "RF": 0.937431,
}

EXPECTED_TEST = {
    "LR":      (0.8195, 0.839644, 0.886364, 0.862371, 0.892052),
    "KNN":     (0.8685, 0.848110, 0.967085, 0.903698, 0.957871),
    "RBF-SVM": (0.9725, 0.978824, 0.978056, 0.978440, 0.996695),
    "RF":      (0.9285, 0.932773, 0.956897, 0.944681, 0.983253),
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
            ("model", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=1)),
        ]),
        "RBF-SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(C=10, gamma=0.05, kernel="rbf", probability=True)),
        ]),
        "RF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=1,
            )),
        ]),
    }

def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

def main(csv_path):
    df = pd.read_csv(csv_path)
    X = df[FEATURES].copy()
    y = (df["stabf"] == "unstable").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print("=== Frozen split ===")
    print(f"Train: {len(X_train):,} | stable={(y_train==0).sum():,} | unstable={(y_train==1).sum():,}")
    print(f"Test:  {len(X_test):,} | stable={(y_test==0).sum():,} | unstable={(y_test==1).sum():,}")

    assert len(X_train) == 8000 and len(X_test) == 2000
    assert int((y_train==0).sum()) == 2896 and int((y_train==1).sum()) == 5104
    assert int((y_test==0).sum()) == 724 and int((y_test==1).sum()) == 1276

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()

    print("\n=== 5-fold CV F1 on training set ===")
    cv_means = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        cv_means[name] = scores.mean()
        print(f"{name:7s}: folds={np.round(scores, 6)} mean={scores.mean():.6f} sd={scores.std():.6f}")

    print("\n=== Clean held-out test performance ===")
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = positive_scores(model, X_test)

        metrics = (
            accuracy_score(y_test, pred),
            precision_score(y_test, pred),
            recall_score(y_test, pred),
            f1_score(y_test, pred),
            roc_auc_score(y_test, score),
        )
        rows.append([name, *metrics])
        print(
            f"{name:7s}: accuracy={metrics[0]:.6f} precision={metrics[1]:.6f} "
            f"recall={metrics[2]:.6f} f1={metrics[3]:.6f} auc={metrics[4]:.6f}"
        )

    # CV means should reproduce to displayed six-decimal precision.
    for name, expected in EXPECTED_CV.items():
        assert round(cv_means[name], 6) == round(expected, 6), (name, cv_means[name], expected)

    # Clean test metrics should reproduce to the manuscript's frozen precision.
    for row in rows:
        name = row[0]
        observed = row[1:]
        expected = EXPECTED_TEST[name]
        for obs, exp in zip(observed, expected):
            assert round(obs, 6) == round(exp, 6), (name, obs, exp)

    out = pd.DataFrame(rows, columns=["model","accuracy","precision","recall","f1","roc_auc"])
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "clean_performance_reproduced.csv", index=False)

    print("\nPASS: split, selected-model CV, and clean test metrics reproduced the frozen study results.")
    print(f"Saved: {out_dir / 'clean_performance_reproduced.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    main(args.csv_path)
