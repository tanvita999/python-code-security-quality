
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
    """Load a locally fine-tuned model if available; otherwise use base CodeBERT."""
    model_path = MODEL_DIR if (MODEL_DIR / "config.json").exists() else MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    return tokenizer, model


def predict_probabilities(code, tokenizer, model):
    """Return class probabilities for a code snippet."""
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    return probabilities


def explain_code(code, tokenizer, model):
    """Rank code tokens by how much masking each token lowers the predicted-class score."""
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        original_logits = model(**inputs).logits[0]
        original_class = int(original_logits.argmax())
        original_score = float(
            torch.softmax(original_logits, dim=-1)[original_class]
        )

    token_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    special_ids = set(tokenizer.all_special_ids)
    mask_id = tokenizer.mask_token_id

    if mask_id is None:
        raise ValueError("The tokenizer does not provide a mask token.")

    explanations = []

    for index, token_id in enumerate(token_ids.tolist()):
        if token_id in special_ids:
            continue

        masked_inputs = {key: value.clone() for key, value in inputs.items()}
        masked_inputs["input_ids"][0, index] = mask_id

        with torch.no_grad():
            logits = model(**masked_inputs).logits[0]
            masked_score = float(
                torch.softmax(logits, dim=-1)[original_class]
            )

        explanations.append(
            {
                "token": tokens[index],
                "position": index,
                "original_class": original_class,
                "original_score": original_score,
                "masked_score": masked_score,
                "importance": original_score - masked_score,
            }
        )

    explanations.sort(key=lambda item: item["importance"], reverse=True)
    return explanations


def main():
    parser = argparse.ArgumentParser(
        description="Explain CodeBERT predictions using token occlusion."
    )
    parser.add_argument(
        "--code",
        type=str,
        help="Python code snippet to explain.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Test CSV row to explain when --code is not supplied.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/security_explanations.csv",
        help="Path for the explanation CSV.",
    )
    args = parser.parse_args()

    if args.code is None:
        if not TEST_PATH.exists():
            raise FileNotFoundError(
                f"Test dataset not found: {TEST_PATH}"
            )

        test_df = pd.read_csv(TEST_PATH)
        if "code" not in test_df.columns:
            raise ValueError("The test CSV must contain a 'code' column.")
        if not 0 <= args.row < len(test_df):
            raise IndexError(f"Row must be between 0 and {len(test_df) - 1}.")

        code = str(test_df.iloc[args.row]["code"])
    else:
        code = args.code

    tokenizer, model = load_model()
    probabilities = predict_probabilities(code, tokenizer, model)
    predicted_class = int(probabilities.argmax())

    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {probabilities.tolist()}")
    print("\nMost influential tokens (occlusion ranking):")

    explanations = explain_code(code, tokenizer, model)
    for item in explanations[:15]:
        print(
            f"{item['token']:<20} "
            f"importance={item['importance']:+.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(explanations).to_csv(output_path, index=False)
    print(f"\nExplanation saved to: {output_path}")


if __name__ == "__main__":
    main()
