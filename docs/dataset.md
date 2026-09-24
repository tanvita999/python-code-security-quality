# Dataset Documentation

## 1. Overview

This project uses separate datasets for two prediction tasks:

1. Python security vulnerability detection
2. Python code-quality / code-smell detection

The raw datasets are stored under `data/raw/` and are kept unchanged.

Processed datasets are stored under `data/processed/`.

The security and quality datasets are maintained as separate task-specific datasets. They are not merged row-by-row because they represent different prediction tasks and different data formats.

---

## 2. Security Dataset

### 2.1 Source

The security dataset used in this project is the **DetectVul/Vudenc Python dataset**.

The dataset is designed for **statement-level vulnerability detection in Python**. DetectVul uses information about code statements and statement types to identify vulnerable statements. The DetectVul work uses VUDENC as one of the publicly available datasets for Python vulnerability detection.

The dataset is available through the Hugging Face Hub under:

**DetectVul/Vudenc**

Dataset source:

https://huggingface.co/datasets/DetectVul/Vudenc

The original VUDENC dataset is also available through Zenodo:

https://zenodo.org/records/3559841

### 2.2 Dataset Version and Format

The version used in this project corresponds to the **DetectVul/Vudenc dataset available on the Hugging Face Hub** at the time of data collection.

Dataset format:

- Parquet
- Python source-code statements
- Sequence-based records

The dataset contains two splits:

- Training split: 12,672 records
- Test split: 3,169 records

The Hugging Face dataset contains the following fields:

- `lines` — normalized code statements
- `raw_lines` — original code statements
- `label` — vulnerability labels aligned with the code statements in each sequence
- `type` — statement type information aligned with the code statements

The original dataset structure stores multiple statements and their corresponding labels and statement types within a record.

### 2.3 Raw Files

The raw security dataset files used in this project are:

```text
data/raw/security/detectvul/
├── train-00000-of-00001-4cfe5148f991ecb4.parquet
└── test-00000-of-00001-7b79e04247817e16.parquet
```

The raw files are preserved without modification.

### 2.4 Original Research Reference

The dataset is associated with the following DetectVul research work:

**Hoai-Chau Tran, Anh-Duy Tran, and Kim-Hung Le.**

*DetectVul: A statement-level code vulnerability detection for Python.*

**Future Generation Computer Systems, Volume 163, February 2025, Article 107504.**

DOI:

```text
10.1016/j.future.2024.107504
```

The paper uses the VUDENC and CVEFixes datasets for statement-level Python vulnerability detection and covers seven vulnerability types.

### 2.5 VUDENC Reference

The original VUDENC dataset is associated with:

**Laura Wartschinski et al.**

*VUDENC: Vulnerability Detection with Deep Learning on a Natural Codebase for Python.*

**Information and Software Technology, Volume 144, 2022.**

Original dataset record:

https://zenodo.org/records/3559841

Zenodo DOI:

```text
10.5281/zenodo.3559841
```

### 2.6 Dataset License and Usage Terms

The DetectVul GitHub repository is released under the **MIT License**.

However, the software repository license should not be assumed to automatically apply to the underlying VUDENC dataset.

Therefore, the dataset is referenced to its original sources, and any redistribution or reuse of the underlying dataset should follow the license and usage terms specified by the original dataset providers.

Original VUDENC dataset:

https://zenodo.org/records/3559841

DetectVul repository:

https://github.com/hchautran/DetectVul

### 2.7 Data Processing

The raw Parquet files are kept unchanged under:

```text
data/raw/security/detectvul/
```

The preprocessing pipeline performs the following steps:

1. Loads the training and test Parquet files.
2. Converts sequence-based records into statement-level rows.
3. Verifies that code statements, labels, and statement types have matching sequence lengths.
4. Removes statements where the normalized code is empty.
5. Preserves the original training and test split.
6. Saves the processed datasets as CSV files under `data/processed/security/`.

Processed files:

```text
data/processed/security/
├── security_train.csv
└── security_test.csv
```

The processed security dataset contains the following columns:

- `code` — normalized code statement
- `raw_code` — original code statement
- `label` — numeric vulnerability label
- `statement_type` — statement type
- `split` — original dataset split

The numeric security labels are preserved from the source dataset. The preprocessing pipeline does not redefine or remap the vulnerability label values.

### 2.8 Security Dataset Preprocessing Statistics

The preprocessing pipeline produced the following statistics:

| Split | Function Records | Statements Before Cleaning | Empty Statements Removed | Statements After Cleaning |
|------|------------------:|----------------------------:|--------------------------:|---------------------------:|
| Training | 12,672 | 124,048 | 442 | 123,606 |
| Test | 3,169 | 29,871 | 112 | 29,759 |

Training label distribution:

```text
{0: 117923, 1: 603, 2: 680, 3: 68, 4: 2375, 5: 946, 6: 632, 7: 379}
```

Test label distribution:

```text
{0: 28338, 1: 189, 2: 141, 3: 13, 4: 545, 5: 238, 6: 214, 7: 81}
```

The preprocessing pipeline checks the label distribution separately for the training and test splits.

The label values are retained from the original DetectVul/Vudenc dataset and are not manually remapped during preprocessing.

---

## 3. Quality Datasets

### 3.1 Overview

Two datasets are used for Python code-quality / code-smell detection:

1. Large Class
2. Long Method

The quality datasets contain **software metrics rather than complete Python source-code statements**.

Therefore, these datasets are treated as metric-based code-quality classification datasets and are not described as direct source-code inputs to the CodeBERT security model.

The datasets contain metrics related to code size, comments, vocabulary, complexity, effort, and other software characteristics.

The raw quality datasets are stored under:

```text
data/raw/quality/
```

### 3.2 Source and Reference

The quality datasets are from the following research work:

**R. Sandouka and H. Aljamaan.**

*Python code smells detection using conventional machine learning models.*

**PeerJ Computer Science, 9, e1370, 2023.**

DOI:

```text
10.7717/peerj-cs.1370
```

The datasets are available through Zenodo:

https://zenodo.org/records/7512516

The quality datasets should be used according to the licensing and usage terms specified by the original Zenodo dataset record.

---

## 4. Large Class Dataset

### 4.1 Source File

The raw Large Class dataset is stored at:

```text
data/raw/quality/Python_LargeClassSmell_Dataset.csv
```

### 4.2 Dataset Structure

The dataset contains 20 columns, including software metrics such as:

- `loc`
- `lloc`
- `scloc`
- `comments`
- `single_comments`
- `multi_comments`
- `blanks`
- `h1`
- `h2`
- `n1`
- `n2`
- `vocabulary`
- `length`
- `calculated_length`
- `volume`
- `difficulty`
- `effort`
- `time`
- `bugs`

The target column is:

```text
LargeClass
```

The original dataset contains:

```text
1,000 rows
20 columns
```

Original target distribution:

```text
{0: 800, 1: 200}
```

Missing values before preprocessing:

```text
0
```

Exact duplicate rows before preprocessing:

```text
153
```

### 4.3 Processed File

The processed dataset is stored at:

```text
data/processed/quality/quality_large_class.csv
```

After preprocessing:

```text
847 rows
20 columns
```

Processed target distribution:

```text
{0: 654, 1: 193}
```

Missing values after preprocessing:

```text
0
```

Exact duplicate rows removed:

```text
153
```

### 4.4 Processing

The preprocessing pipeline:

1. Loads the raw CSV file.
2. Removes leading and trailing whitespace from column names.
3. Identifies the final column as the target column.
4. Checks for missing values.
5. Removes exact duplicate rows.
6. Converts the target column to integer values.
7. Saves the processed dataset.

The raw dataset is kept unchanged.

---

## 5. Long Method Dataset

### 5.1 Source File

The raw Long Method dataset is stored at:

```text
data/raw/quality/Python_LongMethodSmell_Dataset.csv
```

### 5.2 Dataset Structure

The dataset contains 20 columns consisting of software metrics similar to the Large Class dataset.

The original target column is:

```text
Experince_Based 
```

The source dataset contains a trailing space in the target column name.

The preprocessing pipeline removes leading and trailing whitespace from column names. Therefore, the processed dataset uses:

```text
Experince_Based
```

as the target column.

The spelling `Experince_Based` is preserved from the source dataset after whitespace normalization.

The original dataset contains:

```text
894 rows
20 columns
```

Original target distribution:

```text
{0: 685, 1: 209}
```

Missing values before preprocessing:

```text
0
```

Exact duplicate rows before preprocessing:

```text
78
```

### 5.3 Processed File

The processed dataset is stored at:

```text
data/processed/quality/quality_long_method.csv
```

After preprocessing:

```text
816 rows
20 columns
```

Processed target distribution:

```text
{0: 607, 1: 209}
```

Missing values after preprocessing:

```text
0
```

Exact duplicate rows removed:

```text
78
```

### 5.4 Processing

The preprocessing pipeline:

1. Loads the raw CSV file.
2. Removes leading and trailing whitespace from column names.
3. Identifies the final column as the target column.
4. Checks for missing values.
5. Removes exact duplicate rows.
6. Converts the target column to integer values.
7. Saves the processed dataset.

The raw dataset is kept unchanged.

---

## 6. Data Integrity

The preprocessing pipeline keeps the original raw datasets unchanged.

For the security dataset:

- Code, labels, and statement types are checked for aligned sequence lengths.
- Empty normalized-code statements are removed.
- The original train/test split is preserved.
- Original label values are retained.
- No manual label remapping is performed.

For the quality datasets:

- Exact duplicate rows are removed.
- Column-name whitespace is normalized.
- Missing values are checked.
- Target columns are converted to integer values.
- The original raw files are not modified.

The processed datasets are generated using the preprocessing scripts in:

```text
src/preprocessing/
```

The preprocessing code does not modify the files stored under:

```text
data/raw/
```

---

## 7. Preprocessing Scripts

The preprocessing pipeline is implemented in the following files:

```text
src/preprocessing/
├── preprocess_security.py
├── preprocess_quality.py
└── README.md
```

### 7.1 Security Preprocessing Script

The security preprocessing script:

- Reads the raw DetectVul/Vudenc Parquet files.
- Processes both training and test splits.
- Flattens the sequence-based records into statement-level rows.
- Preserves normalized code.
- Preserves original raw code.
- Preserves numeric labels.
- Preserves statement types.
- Preserves the original split.
- Removes empty normalized-code statements.
- Saves the processed training and test CSV files.
- Reports preprocessing statistics and label distributions.

### 7.2 Quality Preprocessing Script

The quality preprocessing script:

- Reads the Large Class and Long Method CSV files.
- Normalizes column names by removing leading and trailing whitespace.
- Detects the final column as the target column.
- Checks for missing values.
- Removes exact duplicate rows.
- Converts the target column to integer values.
- Saves the processed quality datasets.

The target column is detected from the final column rather than being hardcoded to a specific target name. This allows the preprocessing pipeline to handle the source dataset's whitespace variation in the Long Method target column.

---

## 8. Dataset Statistics Summary

The final processed datasets contain the following number of records:

| Dataset | Original Rows/Records | Processed Rows/Records |
|---------|-----------------------:|------------------------:|
| Security Training | 12,672 function records | 123,606 statements |
| Security Test | 3,169 function records | 29,759 statements |
| Large Class | 1,000 | 847 |
| Long Method | 894 | 816 |

For the security dataset, the original function-level records are converted into statement-level rows during preprocessing. Therefore, the original record count and processed statement count represent different levels of data organization.

For the quality datasets, preprocessing removes exact duplicate rows while retaining the original target labels.

---

## 9. Data Integrity Checks

The following checks are performed during preprocessing.

### 9.1 Security Dataset Checks

For each security dataset split:

- The required columns are loaded from the Parquet files.
- The lengths of code, raw code, labels, and statement types are checked for alignment.
- Empty normalized-code statements are identified and removed.
- Labels are retained without manual remapping.
- The original train/test split is preserved.
- Missing values are checked in the final processed output.
- Label distributions are reported.

### 9.2 Quality Dataset Checks

For each quality dataset:

- The CSV file is loaded successfully.
- Column-name whitespace is removed.
- The final column is identified as the target.
- Missing values are checked.
- Exact duplicate rows are identified and removed.
- Target values are converted to integer values.
- The processed output is saved separately from the raw input.

---

## 10. Raw and Processed Data Policy

The project follows a separation between raw and processed data.

Raw datasets are stored under:

```text
data/raw/
```

Processed datasets are stored under:

```text
data/processed/
```

Raw files are not modified by the preprocessing scripts.

This separation ensures that the original downloaded datasets remain available for reproducibility and verification.

Processed datasets can be regenerated from the raw datasets using the preprocessing scripts.

---

## 11. Reproducibility

The preprocessing pipeline can be executed from the project root using:

```bash
python src/preprocessing/preprocess_security.py
```

and:

```bash
python src/preprocessing/preprocess_quality.py
```

The scripts generate the processed datasets under:

```text
data/processed/security/
```

and:

```text
data/processed/quality/
```

The preprocessing code is stored in the repository under:

```text
src/preprocessing/
```

The raw input datasets must be present in the expected directories before running the preprocessing scripts.

---

## 12. Dataset Directory Structure

The dataset-related project structure is:

```text
data/
├── raw/
│   ├── security/
│   │   └── detectvul/
│   │       ├── train-00000-of-00001-4cfe5148f991ecb4.parquet
│   │       └── test-00000-of-00001-7b79e04247817e16.parquet
│   │
│   └── quality/
│       ├── Python_LargeClassSmell_Dataset.csv
│       └── Python_LongMethodSmell_Dataset.csv
│
└── processed/
    ├── security/
    │   ├── security_train.csv
    │   └── security_test.csv
    │
    └── quality/
        ├── quality_large_class.csv
        └── quality_long_method.csv
```

---

## 13. Dataset Usage in the Project

The security dataset is used for **statement-level Python vulnerability detection**.

The security data is structured around Python source-code statements and their corresponding vulnerability labels and statement types.

The quality datasets are used for **metric-based Python code-quality / code-smell classification**.

The quality data is based on software metrics rather than complete source-code statements.

The security and quality datasets are therefore treated as separate prediction tasks within the overall explainable and robust deep learning framework.

The datasets are not merged row-by-row because their input representations, labels, and prediction tasks are different.

The processed security dataset can be used by the model-training component for statement-level vulnerability classification.

The processed quality datasets can be used by the model-training component for metric-based code-smell classification.

---

## 14. Relationship Between Security and Quality Tasks

The project addresses both security and software-quality analysis.

The two tasks use different datasets:

```text
Security Task
    ↓
DetectVul/Vudenc
    ↓
Python source-code statements
    ↓
Statement-level vulnerability labels
```

```text
Quality Task
    ↓
Python Code Smell Datasets
    ↓
Software metrics
    ↓
Code-smell classification
```

The datasets are therefore maintained separately during preprocessing.

A future multi-task or integrated model may use separate task-specific inputs, output heads, or losses rather than assuming that security and quality labels correspond to the same individual samples.

---

## 15. Important Dataset Notes

### 15.1 Security Labels

The security preprocessing pipeline preserves the numeric labels found in the DetectVul/Vudenc dataset.

The preprocessing documentation does not assume that a particular numeric value represents a specific vulnerability category unless that mapping is explicitly established from the source dataset documentation.

Therefore, labels are retained as source-defined numeric values.

### 15.2 Security Statement-Level Representation

The original security data contains multiple statements within individual records.

The preprocessing pipeline converts these sequence-based records into individual statement-level rows.

This results in substantially more processed rows than the original number of function-level records.

### 15.3 Quality Dataset Representation

The Large Class and Long Method datasets are metric-based datasets.

They contain software metrics such as:

- Lines of code
- Logical lines of code
- Comments
- Vocabulary
- Length
- Volume
- Difficulty
- Effort
- Time
- Bugs

They are therefore not treated as raw Python source-code statement datasets.

### 15.4 Duplicate Removal

Exact duplicate rows are removed only from the processed quality datasets.

The raw quality datasets remain unchanged.

For the Large Class dataset, 153 exact duplicate rows were removed.

For the Long Method dataset, 78 exact duplicate rows were removed.

### 15.5 Column Name Normalization

The quality preprocessing pipeline removes leading and trailing whitespace from column names.

The Long Method source dataset contains a trailing space in its target column name:

```text
Experince_Based 
```

After preprocessing, the column name becomes:

```text
Experince_Based
```

The spelling of the source column is otherwise preserved.

---

## 16. Processed Dataset Summary

### 16.1 Security Training

File:

```text
data/processed/security/security_train.csv
```

Columns:

```text
code
raw_code
label
statement_type
split
```

Number of processed statements:

```text
123,606
```

### 16.2 Security Test

File:

```text
data/processed/security/security_test.csv
```

Columns:

```text
code
raw_code
label
statement_type
split
```

Number of processed statements:

```text
29,759
```

### 16.3 Large Class

File:

```text
data/processed/quality/quality_large_class.csv
```

Number of processed rows:

```text
847
```

Number of columns:

```text
20
```

Target:

```text
LargeClass
```

### 16.4 Long Method

File:

```text
data/processed/quality/quality_long_method.csv
```

Number of processed rows:

```text
816
```

Number of columns:

```text
20
```

Target:

```text
Experince_Based
```

---

## 17. References

1. Tran, H.-C., Tran, A.-D., and Le, K.-H. *DetectVul: A statement-level code vulnerability detection for Python.* Future Generation Computer Systems, Volume 163, February 2025, Article 107504. DOI: `10.1016/j.future.2024.107504`.

2. Wartschinski, L. et al. *VUDENC: Vulnerability Detection with Deep Learning on a Natural Codebase for Python.* Information and Software Technology, Volume 144, 2022.

3. Sandouka, R. and Aljamaan, H. *Python code smells detection using conventional machine learning models.* PeerJ Computer Science, 9, e1370, 2023. DOI: `10.7717/peerj-cs.1370`.

4. DetectVul/Vudenc Dataset, Hugging Face Hub:

https://huggingface.co/datasets/DetectVul/Vudenc

5. VUDENC Dataset, Zenodo:

https://zenodo.org/records/3559841

6. Python Code Smell Dataset, Zenodo:

https://zenodo.org/records/7512516

7. DetectVul Source Repository:

https://github.com/hchautran/DetectVul