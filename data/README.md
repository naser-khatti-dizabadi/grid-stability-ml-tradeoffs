# Dataset

This repository does not redistribute the raw dataset.

Download **Electrical Grid Stability Simulated Data** from the UCI Machine Learning Repository.

DOI: 10.24432/C5PG66

Expected analysis file:

`Data_for_UCI_named.csv`

The scripts accept the dataset path as a command-line argument, for example:

```bash
python src/01_dataset_audit.py /path/to/Data_for_UCI_named.csv
```

The audit script verifies the expected dataset shape, class distribution, target leakage through `stab`, redundancy of `p1`, and the final 11-predictor feature set.
