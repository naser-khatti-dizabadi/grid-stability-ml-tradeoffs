#!/usr/bin/env python3
"""
Generate reproducibility figures from the reproduced result tables.

These figures reproduce the analytical content of the manuscript figures.
Exact typography/layout may differ from the archived publication PDFs.

Outputs:
    figures/reproduced_gaussian_robustness.pdf
    figures/reproduced_missing_robustness.pdf
    figures/reproduced_tradeoff.pdf
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["LR", "KNN", "RBF-SVM", "RF"]
LABELS = {"LR":"LR", "KNN":"KNN", "RBF-SVM":"RBF-SVM", "RF":"RF"}

# Archived controlled efficiency values used in the manuscript.
ARCHIVED_LATENCY_MS = {
    "LR": 0.763,
    "KNN": 53.737,
    "RBF-SVM": 66.492,
    "RF": 45.764,
}
ARCHIVED_SIZE_KB = {
    "LR": 1.869,
    "KNN": 751.769,
    "RBF-SVM": 135.803,
    "RF": 22047.050,
}

def robustness_plot(summary_path, out_path, xlabel):
    df = pd.read_csv(summary_path)
    fig, ax = plt.subplots(figsize=(6.6, 4.5))

    for model in MODELS:
        g = df[df["model"] == model].sort_values("level")
        ax.plot(
            g["level"] * 100,
            g["mean_f1"] * 100,
            marker="o",
            label=LABELS[model]
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean F1-score (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def tradeoff_plot(clean_path, out_path):
    df = pd.read_csv(clean_path)
    # Accommodate either "model"/"f1" or the frozen script's column naming.
    model_col = "model" if "model" in df.columns else "Model"
    f1_col = "f1" if "f1" in df.columns else "F1"

    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    for _, row in df.iterrows():
        model = row[model_col]
        latency = ARCHIVED_LATENCY_MS[model]
        f1 = float(row[f1_col]) * 100
        # sqrt-like compression keeps RF visible without overwhelming the plot.
        marker_area = 35 + 10 * (ARCHIVED_SIZE_KB[model] ** 0.25)
        ax.scatter(latency, f1, s=marker_area)
        ax.annotate(model, (latency, f1), xytext=(5, 5),
                    textcoords="offset points")

    ax.set_xlabel("Median inference latency for 2,000 samples (ms)")
    ax.set_ylabel("Clean test F1-score (%)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    repo = Path(__file__).resolve().parents[1]
    results = repo / "results"
    figures = repo / "figures"
    figures.mkdir(exist_ok=True)

    robustness_plot(
        results / "gaussian_robustness_summary_reproduced.csv",
        figures / "reproduced_gaussian_robustness.pdf",
        "Gaussian noise level (% of training-feature SD)",
    )
    robustness_plot(
        results / "missingness_robustness_summary_reproduced.csv",
        figures / "reproduced_missing_robustness.pdf",
        "Missing feature cells (%)",
    )
    tradeoff_plot(
        results / "clean_performance_reproduced.csv",
        figures / "reproduced_tradeoff.pdf",
    )

    print("PASS: reproducibility figures generated.")
    for name in [
        "reproduced_gaussian_robustness.pdf",
        "reproduced_missing_robustness.pdf",
        "reproduced_tradeoff.pdf",
    ]:
        print("Saved:", figures / name)

if __name__ == "__main__":
    main()
