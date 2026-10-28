#!/usr/bin/env python3
"""
Reproduce MCAR missing-measurement robustness experiments.

Protocol:
- Frozen stratified 80/20 split, random_state=42.
- Models trained only on clean training data.
- Cell-level MCAR feature dropout on held-out test data only.
- Missingness levels: 0.05, 0.10, 0.20, 0.30.
- 30 repetitions.
- Seed = 2000 + repetition.
- One RNG per repetition, consumed sequentially across missingness levels.
- Same missingness mask supplied to all models within each repetition/level.
- Missing values imputed by training-set medians through the fitted pipelines.

Usage:
    python src/04_missingness_robustness.py /path/to/Data_for_UCI_named.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

FEATURES = ["tau1","tau2","tau3","tau4","p2","p3","p4","g1","g2","g3","g4"]
LEVELS = [0.05, 0.10, 0.20, 0.30]
N_REPS = 30

ARCHIVED_MEAN_F1 = {
    0.05: {"KNN":0.89386, "LR":0.85601, "RBF-SVM":0.95407, "RF":0.92742},
    0.10: {"KNN":0.88180, "LR":0.84828, "RBF-SVM":0.93246, "RF":0.91079},
    0.20: {"KNN":0.86211, "LR":0.83419, "RBF-SVM":0.89613, "RF":0.88180},
    0.30: {"KNN":0.84528, "LR":0.82262, "RBF-SVM":0.86748, "RF":0.85796},
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
                n_estimators=200, max_depth=None, min_samples_leaf=1,
                random_state=42, n_jobs=1
            )),
        ]),
    }

def main(csv_path):
    df = pd.read_csv(csv_path)
    X = df[FEATURES].copy()
    y = (df["stabf"] == "unstable").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models = build_models()
    for model in models.values():
        model.fit(X_train, y_train)

    X_test_np = X_test.to_numpy()
    rows = []

    for repetition in range(N_REPS):
        rng = np.random.default_rng(2000 + repetition)

        # IMPORTANT: same RNG continues sequentially through all levels.
        for level in LEVELS:
            mask = rng.random(X_test_np.shape) < level
            X_missing_np = X_test_np.copy()
            X_missing_np[mask] = np.nan
            X_missing = pd.DataFrame(
                X_missing_np, columns=FEATURES, index=X_test.index
            )

            for name, model in models.items():
                pred = model.predict(X_missing)
                rows.append({
                    "condition": "Missing measurements",
                    "level": level,
                    "repetition": repetition,
                    "seed": 2000 + repetition,
                    "model": name,
                    "accuracy": accuracy_score(y_test, pred),
                    "f1": f1_score(y_test, pred),
                })

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["level","model"], as_index=False)
        .agg(mean_f1=("f1","mean"), sd_f1=("f1","std"),
             mean_accuracy=("accuracy","mean"), sd_accuracy=("accuracy","std"))
    )

    print("=== MCAR missingness robustness: reproduced means ===")
    for level in LEVELS:
        print(f"\nMissingness = {100*level:.0f}%")
        block = summary[summary["level"] == level].set_index("model")
        for name in ["KNN","LR","RBF-SVM","RF"]:
            mean = block.loc[name, "mean_f1"]
            sd = block.loc[name, "sd_f1"]
            archived = ARCHIVED_MEAN_F1[level][name]
            diff_pp = 100 * (mean - archived)
            print(
                f"{name:7s}: mean F1={100*mean:.3f}%  SD={100*sd:.3f}%  "
                f"archived={100*archived:.3f}%  difference={diff_pp:+.4f} pp"
            )

    # Archived table values were stored to three decimals in percentage units.
    # Require every model/level to reproduce that manuscript precision.
    for level in LEVELS:
        block = summary[summary["level"] == level].set_index("model")
        for name in ["LR","KNN","RBF-SVM","RF"]:
            obs_pct = 100 * block.loc[name, "mean_f1"]
            exp_pct = 100 * ARCHIVED_MEAN_F1[level][name]
            assert round(obs_pct, 3) == round(exp_pct, 3), (level, name, obs_pct, exp_pct)

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "missingness_robustness_reproduced.csv"
    summary_path = out_dir / "missingness_robustness_summary_reproduced.csv"
    results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nPASS: all MCAR missingness means reproduce the archived manuscript values.")
    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    main(args.csv_path)
