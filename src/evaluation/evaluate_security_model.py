
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path("models/security_codebert_baseline")
TEST_CSV = Path("data/processed/security/security_test.csv")
RESULTS_DIR = Path("results")
SAMPLE_SIZE = 0
BATCH_SIZE = 8
MAX_LENGTH = 256


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_DIR}")

    df = pd.read_csv(TEST_CSV).dropna(subset=["code", "label"])
    df["label"] = df["label"].astype(int)

    # Sample without replacement; include all available classes where possible.
    sample = df.reset_index(drop=True)

    # Keep the total sample within the requested limit.

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")
    print(f"Evaluation examples: {len(sample)}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    predictions = []
    texts = sample["code"].astype(str).tolist()

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            batch_predictions = torch.argmax(logits, dim=-1).cpu().tolist()

        predictions.extend(batch_predictions)
        print(f"Processed {min(start + len(batch_texts), len(texts))}/{len(texts)}")

    y_true = sample["label"].tolist()
    y_pred = predictions
    labels = list(range(model.config.num_labels))

    print("\nAccuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Macro-F1:", round(f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    ), 4))
    print("Weighted-F1:", round(f1_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    ), 4))

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(RESULTS_DIR / "security_confusion_matrix.csv")

    sample["predicted_label"] = y_pred
    sample.to_csv(RESULTS_DIR / "security_evaluation_predictions.csv", index=False)

    print("\nSaved:")
    print(RESULTS_DIR / "security_confusion_matrix.csv")
    print(RESULTS_DIR / "security_evaluation_predictions.csv")


if __name__ == "__main__":
    main()
