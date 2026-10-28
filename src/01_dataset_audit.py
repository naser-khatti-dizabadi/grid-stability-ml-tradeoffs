#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

FINAL_PREDICTORS = ["tau1","tau2","tau3","tau4","p2","p3","p4","g1","g2","g3","g4"]

def main(csv_path):
    df = pd.read_csv(csv_path)
    print("=== Dataset Audit ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing values: {int(df.isna().sum().sum()):,}")
    print(f"Exact duplicate rows: {int(df.duplicated().sum()):,}")

    counts = df["stabf"].value_counts()
    print("\n=== Target distribution ===")
    for label in ["unstable", "stable"]:
        n = int(counts.get(label, 0))
        print(f"{label}: {n:,} ({100*n/len(df):.1f}%)")

    derived = np.where(df["stab"] <= 0, "stable", "unstable")
    match = np.mean(derived == df["stabf"].to_numpy())
    print("\n=== Target leakage audit ===")
    print(f"stab sign -> stabf agreement: {100*match:.6f}%")

    reconstructed = -(df["p2"] + df["p3"] + df["p4"])
    err = np.abs(df["p1"] - reconstructed)
    print("\n=== Feature redundancy audit ===")
    print("Relationship: p1 = -(p2 + p3 + p4)")
    print(f"Maximum absolute error: {err.max():.12g}")
    print(f"np.allclose: {np.allclose(df['p1'], reconstructed)}")

    predictors = [c for c in df.columns if c not in {"stab","stabf","p1"}]
    print("\n=== Final predictor set ===")
    print(f"Number of predictors: {len(predictors)}")
    print(", ".join(predictors))

    assert df.shape == (10000, 14)
    assert df.isna().sum().sum() == 0
    assert df.duplicated().sum() == 0
    assert counts.get("unstable", 0) == 6380
    assert counts.get("stable", 0) == 3620
    assert match == 1.0
    assert np.allclose(df["p1"], reconstructed)
    assert predictors == FINAL_PREDICTORS
    print("\nPASS: all frozen dataset-audit checks were reproduced.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    main(args.csv_path)
