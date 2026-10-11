# Python Code Security and Quality

**Explainable and Robust Deep Learning Framework for Python Code Security and Quality Anomaly Detection**

## Overview
This project explores deep learning techniques for Python code security and quality classification, with a focus on model explainability and robustness.

## Project Structure
- `data/` — Dataset files
- `docs/` — Project documentation
- `notebooks/` — Experiment notebooks
- `results/` — Evaluation, explainability, and robustness results
- `src/` — Source code for model training, evaluation, explainability, and robustness testing

## Key Components
- Code classification using a CodeBERT-based model
- Token-masking explanations
- Robustness tests under code-formatting changes
- Classification reports and confusion matrix outputs

## Current Limitations
The saved classification report shows that the evaluated model predicts class 0 for every test example. Although overall accuracy is approximately 95.22%, the model has zero recall for classes 1–7. This highlights the challenge of class imbalance.

The robustness experiment maintained the same prediction across three formatting changes for one sample only. Broader testing is needed.

A discrepancy between the balanced model's training-stage evaluation and the separate evaluation script remains unresolved.

## Future Work
- Investigate the evaluation discrepancy
- Improve minority-class detection
- Validate explanation quality
- Evaluate robustness across more samples and transformations

## Results
See `results/` for the generated CSV reports and experiment summaries.
