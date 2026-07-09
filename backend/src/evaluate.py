from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from fine_tuning import AgeDataset

MODEL_PATH = "./models/age_classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

df = pd.read_csv("./data/prepared.csv")

_, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["age_group"],
    random_state=42
)

test_dataset = AgeDataset(
    test_df["text"].tolist(),
    test_df["age_group"].tolist(),
    tokenizer
)

trainer = Trainer(model=model)

predictions = trainer.predict(test_dataset)

y_pred = predictions.predictions.argmax(axis=1)
y_true = predictions.label_ids

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["13-17", "18-29", "30-48"]
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=[
        "13-17",
        "18-29",
        "30-48"
    ]
)

plt.show()