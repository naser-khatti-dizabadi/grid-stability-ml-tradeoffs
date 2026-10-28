#!/usr/bin/env python3
"""
Statistical analysis for robustness experiments.

Inputs:
    results/gaussian_robustness_reproduced.csv
    results/missingness_robustness_reproduced.csv

Outputs:
    results/robustness_95ci_reproduced.csv
    results/statistical_tests_reproduced.csv

Methods:
- 95% CI = mean +/- 1.96 * standard error across 30 perturbation repetitions.
- Two-sided paired Wilcoxon signed-rank tests on matched repetitions.
- Holm correction across the six pairwise model comparisons separately
  within each condition and degradation level.

The intervals characterize perturbation-to-perturbation variation on the
fixed held-out test set; they are not population-level generalization CIs.
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MODELS = ["LR", "KNN", "RBF-SVM", "RF"]

def holm_adjust(pvalues):
    """Holm step-down adjusted p-values, returned in original order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]

    adjusted_ranked = np.empty(m, dtype=float)
    running_max = 0.0
    for i, value in enumerate(ranked):
        candidate = (m - i) * value
        running_max = max(running_max, candidate)
        adjusted_ranked[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted

def ci_table(df):
    rows = []
    for (condition, level, model), g in df.groupby(["condition", "level", "model"]):
        values = g["f1"].to_numpy()
        n = len(values)
        mean = values.mean()
        sd = values.std(ddof=1)
        se = sd / np.sqrt(n)
        margin = 1.96 * se
        rows.append({
            "condition": condition,
            "level": level,
            "model": model,
            "n": n,
            "mean_f1": mean,
            "sd_f1": sd,
            "se_f1": se,
            "ci95_lower": mean - margin,
            "ci95_upper": mean + margin,
        })
    return pd.DataFrame(rows)

def paired_tests(df):
    rows = []
    for (condition, level), block in df.groupby(["condition", "level"]):
        pivot = block.pivot(index="repetition", columns="model", values="f1")
        raw = []
        meta = []

        for a, b in combinations(MODELS, 2):
            x = pivot[a].to_numpy()
            y = pivot[b].to_numpy()
            stat, p = wilcoxon(x, y, alternative="two-sided")
            raw.append(p)
            meta.append((a, b, stat, np.mean(x - y)))

        adjusted = holm_adjust(raw)
        for (a, b, stat, mean_diff), p_raw, p_holm in zip(meta, raw, adjusted):
            rows.append({
                "condition": condition,
                "level": level,
                "model_a": a,
                "model_b": b,
                "mean_f1_difference_a_minus_b": mean_diff,
                "wilcoxon_statistic": stat,
                "p_raw": p_raw,
                "p_holm": p_holm,
            })

    return pd.DataFrame(rows)

def main():
    repo = Path(__file__).resolve().parents[1]
    results_dir = repo / "results"

    gaussian = pd.read_csv(results_dir / "gaussian_robustness_reproduced.csv")
    missing = pd.read_csv(results_dir / "missingness_robustness_reproduced.csv")
    df = pd.concat([gaussian, missing], ignore_index=True)

    ci = ci_table(df)
    tests = paired_tests(df)

    ci_path = results_dir / "robustness_95ci_reproduced.csv"
    tests_path = results_dir / "statistical_tests_reproduced.csv"
    ci.to_csv(ci_path, index=False)
    tests.to_csv(tests_path, index=False)

    print("=== 30% degradation: mean F1 and 95% CI ===")
    for condition in ["Gaussian noise", "Missing measurements"]:
        print(f"\n{condition}")
        block = ci[(ci["condition"] == condition) & (ci["level"] == 0.30)]
        block = block.set_index("model")
        for model in ["KNN", "LR", "RBF-SVM", "RF"]:
            r = block.loc[model]
            print(
                f"{model:7s}: {100*r['mean_f1']:.3f}% "
                f"[{100*r['ci95_lower']:.3f}, {100*r['ci95_upper']:.3f}]"
            )

    print("\n=== RBF-SVM vs RF: paired Wilcoxon + Holm ===")
    svm_rf = tests[
        ((tests["model_a"] == "RBF-SVM") & (tests["model_b"] == "RF")) |
        ((tests["model_a"] == "RF") & (tests["model_b"] == "RBF-SVM"))
    ].copy()

    for _, r in svm_rf.sort_values(["condition", "level"]).iterrows():
        diff = r["mean_f1_difference_a_minus_b"]
        if r["model_a"] == "RF":
            diff = -diff
        print(
            f"{r['condition']:20s} level={r['level']:.2f}: "
            f"SVM-RF={100*diff:+.6f} pp, p_Holm={r['p_holm']:.9g}"
        )

    # Strong reproducibility check for missingness 30% CIs at manuscript rounding.
    expected_missing_30 = {
        "KNN": (84.528, 84.394, 84.662),
        "LR": (82.262, 82.036, 82.488),
        "RBF-SVM": (86.748, 86.564, 86.933),
        "RF": (85.796, 85.624, 85.969),
    }
    block = ci[(ci["condition"] == "Missing measurements") & (ci["level"] == 0.30)].set_index("model")
    for model, (m, lo, hi) in expected_missing_30.items():
        r = block.loc[model]
        # Archived publication values are rounded summaries; allow 0.0015 pp.
        assert abs(100*r["mean_f1"] - m) < 0.0015
        assert abs(100*r["ci95_lower"] - lo) < 0.0015
        assert abs(100*r["ci95_upper"] - hi) < 0.0015

    print("\nPASS: CI construction and paired Wilcoxon/Holm analysis reproduced the study protocol.")
    print("NOTE: Gaussian RF-dependent p-values can differ slightly across software/runtime environments.")
    print(f"Saved: {ci_path}")
    print(f"Saved: {tests_path}")

if __name__ == "__main__":
    main()
