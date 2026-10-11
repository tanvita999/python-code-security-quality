from transformers import AutoModelForSequenceClassification


MODEL_NAME = "microsoft/codebert-base"


def create_model(num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    return model
