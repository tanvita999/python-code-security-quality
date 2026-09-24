# Dataset Documentation

## 1. Overview

This project uses separate datasets for two prediction tasks:

1. Python security vulnerability detection
2. Python code-quality / code-smell detection

The raw datasets are stored under `data/raw/` and are kept unchanged.

Processed datasets will be generated under `data/processed/`.

---

## 2. Security Dataset

### 2.1 Source

The security dataset is the DetectVul/Vudenc Python dataset.

The dataset is designed for statement-level vulnerability detection in Python. DetectVul uses information about both code statements and statement types.

Raw files:

```text
data/raw/security/detectvul/
├── train-00000-of-00001-4cfe5148f991ecb4.parquet
└── test-00000-of-00001-7b79e04247817e16.parquet