from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_SECURITY_DIR = PROJECT_ROOT / "data" / "raw" / "security" / "detectvul"
PROCESSED_SECURITY_DIR = PROJECT_ROOT / "data" / "processed" / "security"


TRAIN_FILE = "train-00000-of-00001-4cfe5148f991ecb4.parquet"
TEST_FILE = "test-00000-of-00001-7b79e04247817e16.parquet"


def load_security_split(file_path: Path) -> pd.DataFrame:
    """Load one raw DetectVul/Vudenc Parquet split."""
    return pd.read_parquet(file_path)


def flatten_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Convert function-level sequence records into statement-level rows."""

    rows = []

    for _, record in df.iterrows():
        lines = record["lines"]
        raw_lines = record["raw_lines"]
        labels = record["label"]
        statement_types = record["type"]

        # Make sure all sequence fields are aligned.
        if not (
            len(lines)
            == len(raw_lines)
            == len(labels)
            == len(statement_types)
        ):
            raise ValueError("Sequence lengths do not match in a record.")

        for code, raw_code, label, statement_type in zip(
            lines,
            raw_lines,
            labels,
            statement_types,
        ):
            rows.append(
                {
                    "code": code,
                    "raw_code": raw_code,
                    "label": int(label),
                    "statement_type": statement_type,
                    "split": split_name,
                }
            )

    return pd.DataFrame(rows)


def remove_empty_code(df: pd.DataFrame) -> pd.DataFrame:
    """Remove statements where the normalized code is empty."""

    return df[
        df["code"].notna()
        & df["code"].astype(str).str.strip().ne("")
    ].copy()


def main():
    # Create processed output directory.
    PROCESSED_SECURITY_DIR.mkdir(parents=True, exist_ok=True)

    # Raw dataset paths.
    train_path = RAW_SECURITY_DIR / TRAIN_FILE
    test_path = RAW_SECURITY_DIR / TEST_FILE

    # Load raw data.
    train_df = load_security_split(train_path)
    test_df = load_security_split(test_path)

    print("Training function records:", len(train_df))
    print("Test function records:", len(test_df))

    # Convert function-level sequences into statement-level rows.
    train_processed = flatten_split(train_df, "train")
    test_processed = flatten_split(test_df, "test")

    # Record sizes before cleaning.
    train_before = len(train_processed)
    test_before = len(test_processed)

    # Remove empty-code statements.
    train_processed = remove_empty_code(train_processed)
    test_processed = remove_empty_code(test_processed)
    print()
    print("Training label distribution:")
    print(train_processed["label"].value_counts().sort_index().to_dict())
    print()
    print("Test label distribution:")
    print(test_processed["label"].value_counts().sort_index().to_dict())

    # Report cleaning results.
    print()
    print("Training statements before cleaning:", train_before)
    print(
        "Training empty-code statements removed:",
        train_before - len(train_processed),
    )
    print("Training statements after cleaning:", len(train_processed))

    print()
    print("Test statements before cleaning:", test_before)
    print(
        "Test empty-code statements removed:",
        test_before - len(test_processed),
    )
    print("Test statements after cleaning:", len(test_processed))

    # Output paths.
    train_output = PROCESSED_SECURITY_DIR / "security_train.csv"
    test_output = PROCESSED_SECURITY_DIR / "security_test.csv"

    # Save processed datasets.
    train_processed.to_csv(train_output, index=False)
    test_processed.to_csv(test_output, index=False)

    print()
    print("Saved:")
    print(train_output)
    print(test_output)


if __name__ == "__main__":
    main()