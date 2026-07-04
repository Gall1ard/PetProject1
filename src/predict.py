from transformers import pipeline
from dataclasses import dataclass


classifier = pipeline(
    "text-classification",
    model="../models/age_classifier"
)

id2label = {
    "LABEL_0": "13-17",
    "LABEL_1": "18-29",
    "LABEL_2": "30-48"
}

@dataclass
class Prediction:
    age_group: str | None = None
    confidence: float | None = None
    text: str | None = None

    def to_dict(self) -> dict:
        return {
            "age_group": self.age_group,
            "confidence": f"{self.confidence}%",
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