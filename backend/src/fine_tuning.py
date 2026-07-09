from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import TrainingArguments, Trainer


MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


""" Dataset class """
class AgeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }


def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(
            labels,
            preds,
            average="macro"
        )
    }

def build_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )


def main():
    """ Dataframe splitting """

    df = pd.read_csv('./data/prepared.csv')
    train_df, test_df = train_test_split(df,
                                         test_size=0.2,
                                         stratify=df['age_group'],
                                         random_state=42)

    train_dataset = AgeDataset(
        train_df["text"].tolist(),
        train_df["age_group"].tolist(),
        tokenizer
    )

    test_dataset = AgeDataset(
        test_df["text"].tolist(),
        test_df["age_group"].tolist(),
        tokenizer
    )

    model = build_model()

    """ Training arguments and trainer"""

    training_args = TrainingArguments(
        output_dir="./results",

        num_train_epochs=3,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=2e-5,

        load_best_model_at_end=True,

        metric_for_best_model="f1_macro"
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=test_dataset,

        compute_metrics=compute_metrics
    )


    trainer.train()

    trainer.save_model("./models/age_classifier")
    tokenizer.save_pretrained(
        "./models/age_classifier"
    )


if __name__ == "__main__":
    main()