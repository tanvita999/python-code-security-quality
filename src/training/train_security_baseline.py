import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


TRAIN_PATH = "data/processed/security/security_train.csv"
TEST_PATH = "data/processed/security/security_test.csv"


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df["code"].fillna("")
    y_train = train_df["label"]

    X_test = test_df["code"].fillna("")
    y_test = test_df["label"]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=100000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
