from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "microsoft/codebert-base"

TRAIN_CSV = Path("data/processed/security/security_train.csv")
TEST_CSV = Path("data/processed/security/security_test.csv")
OUTPUT_DIR = Path("models/security_codebert_balanced")

MAX_TRAIN_SAMPLES = 5000
BATCH_SIZE = 4
NUM_EPOCHS = 1
MAX_LENGTH = 256
RANDOM_SEED = 42


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
    }


def main():
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for name, df in [("training", train_df), ("test", test_df)]:
        missing = {"code", "label"} - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: {sorted(missing)}"
            )

    train_df = train_df.dropna(subset=["code", "label"]).copy()
    test_df = test_df.dropna(subset=["code", "label"]).copy()
    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    labels = sorted(train_df["label"].unique().tolist())
    if labels != list(range(max(labels) + 1)):
        raise ValueError(f"Unexpected training labels: {labels}")

    if not set(test_df["label"].unique()).issubset(set(labels)):
        raise ValueError("Test set contains labels absent from training data.")

    # Sample approximately equally from each class.
    per_class = MAX_TRAIN_SAMPLES // len(labels)
    sampled_groups = []

    for label_value, group in train_df.groupby("label"):
        sampled_groups.append(
            group.sample(
                n=min(per_class, len(group)),
                random_state=RANDOM_SEED,
            )
        )

    train_df = pd.concat(sampled_groups, ignore_index=True)
    train_df = train_df.sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Training examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")
    print("Training label counts:")
    print(train_df["label"].value_counts().sort_index())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
    )

    def tokenize_batch(batch):
        texts = [
            str(code) if pd.notna(code) else ""
            for code in batch["code"]
        ]
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = batch["label"]
        return encoded

    train_dataset = Dataset.from_pandas(
        train_df[["code", "label"]].reset_index(drop=True)
    )
    test_dataset = Dataset.from_pandas(
        test_df[["code", "label"]].reset_index(drop=True)
    )

    train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "training_runs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=1,
        dataloader_pin_memory=False,
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting balanced-sample training...")
    trainer.train()

    print("Evaluating on the full test set...")
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        print(f"{key}: {value}")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
