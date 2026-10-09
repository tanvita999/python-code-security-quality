import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


DATASETS = [
    (
        "data/processed/quality/quality_large_class.csv",
        "LargeClass",
        "large_class",
    ),
    (
        "data/processed/quality/quality_long_method.csv",
        "Experince_Based",
        "long_method",
    ),
]


def train_dataset(path, target_column, name):
    df = pd.read_csv(path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


def main():
    for path, target_column, name in DATASETS:
        train_dataset(
            path,
            target_column,
            name,
        )


if __name__ == "__main__":
    main()
