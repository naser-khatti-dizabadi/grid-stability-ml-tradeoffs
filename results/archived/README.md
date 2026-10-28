# Archived Study Results

This directory contains retained result tables from the original study workflow.

These files are the reference outputs underlying the manuscript. They are kept
separate from the independently reproduced outputs in the parent `results/`
directory.

Important notes:

- `lightweight_tuning_results.csv` preserves the retained selected-model tuning
  results. The exact original candidate GridSearchCV parameter grid was not
  retained and is not reconstructed.
- `robustness_all_runs_tuned_30.csv` contains the archived per-repetition
  robustness results used for the manuscript.
- `robustness_summary_tuned_30.csv` contains the corresponding archived
  robustness summaries.
- `efficiency_tuned_controlled.csv` contains the controlled efficiency benchmark
  used as the manuscript reference.
- Reproduced files should not silently overwrite these archived results because
  tiny environment-dependent numerical or timing differences can occur.
