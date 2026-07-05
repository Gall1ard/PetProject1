# Age Prediction from Text using BERT

A machine learning project that predicts the author's age group from English text using a fine-tuned BERT model and lets a chosen external LLM explain how the choice was made.

## Overview

This project fine-tunes BERT on a dataset of English texts labeled by age group.

The model classifies a text into one of the following categories:

* 13–17
* 18–29
* 30–48

The goal is to investigate whether writing style, vocabulary, and discussed topics can be used to estimate the author's age.

## Dataset

Dataset fields:

| Column    | Description                                    |
| --------- | ---------------------------------------------- |
| text      | Original text                                  |
| age_group | Target age category                            |

Class distribution:

| Age group | Samples |
| --------- | ------: |
| 13–17     |  17,471 |
| 18–29     |  55,377 |
| 30–48     |  23,351 |

## Model

Base model:

* bert-base-uncased

Training setup:

* Epochs: 3
* Batch size: 16
* Learning rate: 2e-5
* Max sequence length: 128

## Results

Test metrics:

| Metric   | Score |
| -------- | ----: |
| Accuracy |  0.74 |
| Macro F1 |  0.70 |

Classification report:

| Age group | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| 13–17     |      0.80 |   0.68 | 0.74 |
| 18–29     |      0.75 |   0.84 | 0.79 |
| 30–48     |      0.65 |   0.54 | 0.59 |

Confusion matrix:

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/007afe0a-97cf-4a64-b743-022ccb779443" />

## Project Structure

```text
project/
│
├── data/
│   └── prepared.csv
│
├── models/
│   └── age_classifier/
│
├── prompts/
│   └── system_prompt.txt
│
├── src/
│   ├── config.py
│   ├── evaluate.py
│   ├── fine_tuning.py
│   ├── main.py
│   ├── predict.py
│   ├── prepare_data.py
│   └── reasoning.py
│
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone https://github.com/Gall1ard/age-classifier.git
cd age-classifier

pip install -r requirements.txt
```

All commands below are meant to be run from the project root (this directory), not from inside `src/`.

## Training

This repo does not ship pretrained weights — `models/age_classifier/` is gitignored because a fine-tuned BERT checkpoint is hundreds of MB. You need to build it locally before prediction or the API will work:

```bash
python src/prepare_data.py   # downloads + cleans the dataset into data/prepared.csv
python src/fine_tuning.py    # fine-tunes BERT, saves to models/age_classifier/
```

Fine-tuning benefits significantly from a GPU; on CPU expect it to be considerably slower.

## Evaluation

```bash
python src/evaluate.py
```

Prints a classification report and shows the confusion matrix (requires a trained model, see above).

## Running the API

```bash
uvicorn src.main:app --reload --app-dir src
```

Then either open `http://localhost:8000/docs` for interactive Swagger docs, or:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am preparing for my university exams and applying for internships."}'
```

`GET /health` is available for basic liveness checks.

## Optional LLM Reasoning

The project supports optional LLM-powered explanations for age predictions.

When enabled, an external LLM analyzes the input text and explains why the classifier assigned a particular age group. If it isn't configured, `/predict` still works — the `reasoning` field just comes back with `status: "disabled"`.

### Configuration

Create a `.env` file in the project root:

```env
CHUTES_API_KEY=your_api_key
CHUTES_API_BASE=https://your-chutes-endpoint
CHUTES_LLM=chosen_model_name (example: google/gemma-4-31B-turbo-TEE)
```

### Environment Variables

| Variable          | Description                         |
| ----------------- | ------------------------------------ |
| `CHUTES_API_KEY`  | Chutes API key                       |
| `CHUTES_API_BASE` | Chutes API base URL                  |
| `CHUTES_LLM`      | Model identifier used for reasoning  |

All three must be set for reasoning to activate; if only some are set, a warning is logged so the gap is obvious instead of silent.

### Example

Request:

```json
{ "text": "I am preparing for my university exams and applying for internships." }
```

Response:

```json
{
  "prediction": {
    "age_group": "18-29",
    "confidence": "91%",
    "text": "I am preparing for my university exams and applying for internships."
  },
  "reasoning": {
    "status": "ok",
    "text": "The text references university studies and internships, which are commonly associated with young adults.",
    "error": null
  }
}
```

## Future Improvements

* Compare BERT and RoBERTa
* Use industry as an additional feature
* Experiment with larger context lengths
* Publish the fine-tuned model to the Hugging Face Hub so `predict.py` can fetch it instead of requiring a local training run
* Create a web interface (React)
