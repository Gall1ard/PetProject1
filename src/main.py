from predict import predict_age
from reasoning import get_reasoning
from config import InvalidLLMConfigurationError
from fastapi import FastAPI
import logging
from openai import OpenAIError
from pydantic import BaseModel

app = FastAPI()
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    text: str


@app.post("/predict")
def main(request: PredictionRequest):
    user_input = request.text

    prediction = predict_age(user_input)
    pred2dict = prediction.to_dict()

    try:
        reasoning = get_reasoning(prediction)

    except InvalidLLMConfigurationError as e:
        logger.error("Invalid LLM configuration: %s", e)
        reasoning = None

    except ValueError as e:
        logger.error("Prediction results are invalid: %s", e)
        reasoning = None

    except FileNotFoundError as e:
        logger.error("System prompt file not found: %s", e)
        reasoning = None

    except OpenAIError as e:
        logger.error("Unspecified OpenAI error: %s", e)
        reasoning = None

    return {
        "prediction": pred2dict,
        "reasoning": reasoning
    }