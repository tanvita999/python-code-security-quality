# Data preprocessing 
# Preprocessing

This directory contains the preprocessing pipelines for the Python code security and code-quality datasets.

## Files

### `preprocess_security.py`

Preprocesses the DetectVul/Vudenc Python security dataset.

The raw security data contains function-level records with aligned sequences:

- `lines`
- `raw_lines`
- `label`
- `type`

The preprocessing pipeline:

1. Loads the raw training and test Parquet files.
2. Converts the nested statement sequences into one row per statement.
3. Preserves the normalized code, original code, numeric label, and statement type.
4. Adds the dataset split (`train` or `test`).
5. Removes statements with empty code.
6. Saves separate processed training and test CSV files.

Output:

```text
data/processed/security/
├── security_train.csv
└── security_test.csv