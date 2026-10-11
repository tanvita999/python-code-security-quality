from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_QUALITY_DIR = PROJECT_ROOT / "data" / "raw" / "quality"
PROCESSED_QUALITY_DIR = PROJECT_ROOT / "data" / "processed" / "quality"


LARGE_CLASS_FILE = "Python_LargeClassSmell_Dataset.csv"
LONG_METHOD_FILE = "Python_LongMethodSmell_Dataset.csv"


def load_quality_dataset(file_path: Path) -> pd.DataFrame:
    """Load a raw quality dataset."""
    return pd.read_csv(file_path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading and trailing whitespace from column names."""

    df = df.copy()
    df.columns = df.columns.str.strip()

    return df


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows."""

    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    removed = before - len(df)

    return df, removed


def preprocess_dataset(
    file_name: str,
    output_file_name: str,
) -> None:
    """Preprocess one quality dataset."""

    input_path = RAW_QUALITY_DIR / file_name
    output_path = PROCESSED_QUALITY_DIR / output_file_name

    # Load raw data.
    df = load_quality_dataset(input_path)

    print()
    print("=" * 60)
    print("DATASET:", file_name)
    print("=" * 60)

    print("Original shape:", df.shape)

    # Clean column names.
    df = clean_column_names(df)

    # The quality datasets use the final column as the target.
    target_column = df.columns[-1]

    print("Target column:", repr(target_column))

    # Check missing values before cleaning.
    missing_before = int(df.isna().sum().sum())

    # Remove exact duplicate rows.
    df, duplicates_removed = remove_duplicate_rows(df)

    # Check missing values after cleaning.
    missing_after = int(df.isna().sum().sum())

    # Make sure target values are integers.
    df[target_column] = df[target_column].astype(int)

    # Save processed dataset.
    df.to_csv(output_path, index=False)

    print("Missing values before cleaning:", missing_before)
    print("Duplicate rows removed:", duplicates_removed)
    print("Missing values after cleaning:", missing_after)
    print("Processed shape:", df.shape)

    print("Target distribution:")
    print(
        df[target_column]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("Saved:", output_path)


def main():
    # Create processed output directory.
    PROCESSED_QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    # Large Class dataset.
    preprocess_dataset(
        file_name=LARGE_CLASS_FILE,
        output_file_name="quality_large_class.csv",
    )

    # Long Method dataset.
    preprocess_dataset(
        file_name=LONG_METHOD_FILE,
        output_file_name="quality_long_method.csv",
    )


if __name__ == "__main__":
    main()