# Reproducibility Notes

## Archived versus reproduced outputs

The manuscript's archived experimental outputs remain the reference values reported in the paper.

The `*_reproduced.csv` files are independent reruns used to verify the computational protocol. They are intentionally kept separate from archived results.

### Gaussian robustness

The Gaussian-noise protocol is reproducible using one RNG per repetition with seed `1000 + repetition`, consumed sequentially across noise levels 0.05, 0.10, 0.20, and 0.30.

Very small differences can occur in a few reproduced values, especially Random Forest predictions, across software/runtime environments. Such differences can propagate to RF-dependent Wilcoxon/Holm p-values. They do not change the qualitative conclusions.

### Missingness robustness

The MCAR missingness experiment uses one RNG per repetition with seed `2000 + repetition`, consumed sequentially across missingness levels 0.05, 0.10, 0.20, and 0.30. The reproduced results match the manuscript values to the reported precision.

### Confidence intervals

The manuscript intervals use `mean ± 1.96 × SE` across 30 perturbation realizations. They describe perturbation-to-perturbation variation on the fixed test set, not population-level generalization uncertainty.

### Efficiency

Latency is environment dependent. Reproduction should preserve the benchmarking protocol rather than expect identical millisecond values on different hardware/runtime conditions.

## Hyperparameter search record

The final selected hyperparameters and their five-fold cross-validation F1 scores were retained and reproduced. The exact candidate GridSearchCV parameter grid was not retained and is therefore not reconstructed in this repository.
