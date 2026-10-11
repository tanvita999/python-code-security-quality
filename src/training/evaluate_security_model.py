from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path("models/security_codebert_balanced")

TEST_CSV = Path("data/processed/security/security_test.csv")
RESULTS_DIR = Path("results")

MAX_LENGTH = 256
BATCH_SIZE = 8
SAMPLE_SIZE = None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test data...")
    df = pd.read_csv(TEST_CSV)

    # Use a stratified sample so that each label is represented.
        # Evaluate on the entire untouched test dataset.
    sample = df.copy()

    print(f"Evaluating on {len(sample)} test examples...")
    print("Loading saved model...")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    texts = sample["code"].fillna("").astype(str).tolist()
    true_labels = sample["label"].astype(int).to_numpy()
    predictions = []
    confidences = []

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            batch_predictions = probabilities.argmax(dim=-1)

            predictions.extend(batch_predictions.cpu().numpy().tolist())
            confidences.extend(
                probabilities.max(dim=-1).values.cpu().numpy().tolist()
            )

            if (start // BATCH_SIZE + 1) % 25 == 0:
                print(f"Processed {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

    label_ids = sorted(set(true_labels) | set(predictions))
    label_names = [str(label) for label in label_ids]

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(
        true_labels, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        true_labels, predictions, average="weighted", zero_division=0
    )

    print("\n=== Security Model Evaluation ===")
    print(f"Sample size: {len(sample)}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predictions,
            labels=label_ids,
            target_names=label_names,
            zero_division=0,
        )
    )

    report = classification_report(
        true_labels,
        predictions,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / "security_classification_report.csv"
    )

    cm = confusion_matrix(true_labels, predictions, labels=label_ids)
    pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in label_ids],
        columns=[f"predicted_{label}" for label in label_ids],
    ).to_csv(RESULTS_DIR / "security_confusion_matrix.csv")

    sample = sample.copy()
    sample["predicted_label"] = predictions
    sample["confidence"] = confidences
    sample["correct"] = sample["label"].astype(int) == sample["predicted_label"]
    sample.to_csv(
        RESULTS_DIR / "security_evaluation_predictions.csv", index=False
    )

    print("\nSaved results:")
    print("results/security_classification_report.csv")
    print("results/security_confusion_matrix.csv")
    print("results/security_evaluation_predictions.csv")


if __name__ == "__main__":
    main()
