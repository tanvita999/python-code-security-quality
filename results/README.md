# Experimental Results

## 1. Classification Evaluation
The classification report contains results for eight classes (0-7).
The saved report shows 95.22% accuracy, but the model predicts
class 0 for every test example. Recall and F1-score are zero for
classes 1-7. Therefore, accuracy alone is misleading because the
dataset is highly imbalanced.

## 2. Explainability
Token-masking explanations were generated and saved in
security_explanations.csv. The recorded importance scores are
small and should be interpreted cautiously. Further validation
is needed to establish whether the explanations are meaningful.

## 3. Robustness
Formatting perturbations were tested on one example:
- Leading and trailing whitespace
- Extra blank lines
- Trailing newline

The prediction remained unchanged for all three perturbations.
This result applies only to the tested example and does not
establish robustness across the full dataset.

## 4. Balanced Model Experiment
A balanced-training experiment was completed and saved in
models/security_codebert_balanced. Its training-stage evaluation
reported 41.45% accuracy, macro F1 of 0.1454, and weighted F1
of 0.5519. A separate evaluation script produced results matching
the baseline model, so the discrepancy remains unresolved.

## 5. Limitations and Future Work
- Investigate the discrepancy between the two evaluation results.
- Improve minority-class detection and evaluate per-class metrics.
- Test robustness across more examples and perturbations.
- Validate explanation quality on correctly classified examples.
