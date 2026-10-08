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

TRAIN_PATH = "data/processed/security/security_train.csv"
TEST_PATH = "data/processed/security/security_test.csv"


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_data = Dataset.from_pandas(
        train_df[["code", "label"]],
        preserve_index=False,
    )

    test_data = Dataset.from_pandas(
        test_df[["code", "label"]],
        preserve_index=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["code"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    train_data = train_data.map(
        tokenize,
        batched=True,
    )

    test_data = test_data.map(
        tokenize,
        batched=True,
    )

    train_data = train_data.rename_column(
        "label",
        "labels",
    )

    test_data = test_data.rename_column(
        "label",
        "labels",
    )

    train_data.set_format("torch")
    test_data.set_format("torch")

    num_labels = int(train_df["label"].nunique())

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir="models/security_codebert",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()

    print("\nFinal Evaluation:")
    print(results)


if __name__ == "__main__":
    main()
