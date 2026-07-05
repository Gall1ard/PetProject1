from transformers import pipeline
from dataclasses import dataclass
import os

MODEL_PATH = "./models/age_classifier"

classifier = None

id2label = {
    "LABEL_0": "13-17",
    "LABEL_1": "18-29",
    "LABEL_2": "30-48"
}


class ModelNotFoundError(FileNotFoundError):
    """Raises when the fine-tuned model does not exist (not downloaded/trained)."""
    pass


def _load_classifier():
    if not os.path.isdir(MODEL_PATH):
        raise ModelNotFoundError(
            f"No model found at '{MODEL_PATH}'. This project doesn't ship "
            "pretrained weights in git. Train one first:\n"
            "  1) python src/prepare_data.py\n"
            "  2) python src/fine_tuning.py\n"
            "This trains a full BERT classifier and will take a while "
            "(and benefits a lot from a GPU)."
        )

    return pipeline(
        "text-classification",
        model=MODEL_PATH
    )


@dataclass
class Prediction:
    age_group: str | None = None
    confidence: float | None = None
    text: str | None = None

    def to_dict(self) -> dict:
        return {
            "age_group": self.age_group,
            "confidence": f"{round(self.confidence*100, 2)}%",
            "text": self.text
        }

    def set_params(self, age_group: str = None,
                   confidence: float = None,
                   text: str = None):
        self.age_group = age_group
        self.confidence = confidence
        self.text = text

    def is_valid(self) -> bool:
        return bool(
            self.age_group is not None
            and self.confidence is not None
            and self.text is not None
        )


def predict_age(text: str) -> Prediction:
    global classifier

    if classifier is None:
        classifier = _load_classifier()

    prediction = Prediction()

    if not text.strip():
        return prediction

    result = classifier(text)[0]

    prediction.set_params(
        age_group=id2label[result["label"]],
        confidence=result["score"],
        text=text
    )

    return prediction