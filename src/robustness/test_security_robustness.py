
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "microsoft/codebert-base"
MODEL_DIR = Path("models/security_codebert")
TEST_PATH = Path("data/processed/security/security_test.csv")
MAX_LENGTH = 256


def load_model():
    model_path = MODEL_DIR if (MODEL_DIR / "config.json").exists() else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    return tokenizer, model


def predict(code, tokenizer, model):
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]

    predicted_class = int(probabilities.argmax())
    confidence = float(probabilities[predicted_class])
    return predicted_class, confidence


def formatting_variants(code):
    return {
        "original": code,
        "leading_trailing_whitespace": f"  {code}  ",
        "extra_blank_lines": "\n\n".join(code.splitlines()),
        "trailing_newline": code.rstrip() + "\n",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/security_robustness.csv",
    )
    args = parser.parse_args()

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {TEST_PATH}")

    test_df = pd.read_csv(TEST_PATH)
    if "code" not in test_df.columns:
        raise ValueError("The test CSV must contain a 'code' column.")

    if not 0 <= args.row < len(test_df):
        raise IndexError(f"Row must be between 0 and {len(test_df) - 1}.")

    code = str(test_df.iloc[args.row]["code"])
    tokenizer, model = load_model()

    original_class, original_confidence = predict(code, tokenizer, model)
    print(f"Original predicted class: {original_class}")
    print(f"Original confidence: {original_confidence:.4f}")
    print("\nFormatting variant results:")

    results = []
    for name, variant in formatting_variants(code).items():
        predicted_class, confidence = predict(variant, tokenizer, model)
        matches = predicted_class == original_class

        print(
            f"{name}: class={predicted_class}, "
            f"confidence={confidence:.4f}, matches_original={matches}"
        )
        results.append({
            "variant": name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "prediction_matches_original": matches,
        })

    stability = sum(
        result["prediction_matches_original"] for result in results
    ) / len(results)
    print(f"\nPrediction stability: {stability:.1%}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

