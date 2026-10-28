#!/usr/bin/env python3
"""
Reproduce Gaussian-noise robustness experiments.

Protocol:
- Frozen stratified 80/20 split, random_state=42.
- Models trained only on clean training data.
- Gaussian noise is scaled by each feature's training-set standard deviation.
- Nonzero levels: 0.05, 0.10, 0.20, 0.30 sigma.
- 30 repetitions.
- Seed = 1000 + repetition.
- One RNG is created per repetition and consumed sequentially across noise levels.
- The identical perturbed test set is supplied to all models within a repetition/level.

Usage:
    python src/03_gaussian_robustness.py /path/to/Data_for_UCI_named.csv
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

# Archived manuscript means (F1 proportions) used only for an informative comparison.
ARCHIVED_MEAN_F1 = {
    0.05: {"KNN":0.90424, "LR":0.86288, "RBF-SVM":0.97424, "RF":0.94349},
    0.10: {"KNN":0.90368, "LR":0.86190, "RBF-SVM":0.96620, "RF":0.93945},
    0.20: {"KNN":0.89675, "LR":0.85452, "RBF-SVM":0.93849, "RF":0.92473},
    0.30: {"KNN":0.88833, "LR":0.84735, "RBF-SVM":0.91001, "RF":0.90626},
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

    feature_std = X_train.std(axis=0).to_numpy()
    X_test_np = X_test.to_numpy()

    rows = []
    for repetition in range(N_REPS):
        rng = np.random.default_rng(1000 + repetition)

        # IMPORTANT: the same RNG continues sequentially through all levels.
        for level in LEVELS:
            noise = rng.normal(loc=0.0, scale=level, size=X_test_np.shape)
            X_noisy_np = X_test_np + noise * feature_std
            X_noisy = pd.DataFrame(
                X_noisy_np, columns=FEATURES, index=X_test.index
            )

            for name, model in models.items():
                pred = model.predict(X_noisy)
                rows.append({
                    "condition": "Gaussian noise",
                    "level": level,
                    "repetition": repetition,
                    "seed": 1000 + repetition,
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

    print("=== Gaussian-noise robustness: reproduced means ===")
    for level in LEVELS:
        print(f"\nNoise level = {level:.2f} sigma")
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

    # LR, KNN and SVM were previously shown to reproduce the archived values
    # to manuscript rounding. RF can show tiny environment/version-dependent
    # differences, so we report rather than overwrite the archived results.
    for level in LEVELS:
        block = summary[summary["level"] == level].set_index("model")
        for name in ["LR","KNN","RBF-SVM"]:
            obs = block.loc[name, "mean_f1"]
            exp = ARCHIVED_MEAN_F1[level][name]
            assert abs(obs-exp) < 0.00006, (level, name, obs, exp)

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "gaussian_robustness_reproduced.csv"
    summary_path = out_dir / "gaussian_robustness_summary_reproduced.csv"
    results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nPASS: Gaussian protocol and deterministic non-RF results reproduce the archived study.")
    print("NOTE: Random Forest is reported separately because tiny predictions can vary across sklearn/runtime environments.")
    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    main(args.csv_path)
