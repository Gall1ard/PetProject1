from predict import predict_age, ModelNotFoundError
from reasoning import get_reasoning_result, ReasoningResult

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import logging


app = FastAPI(
    title="Age Classifier API",
    description="Predicts an author's age group from text, with optional LLM-generated reasoning.",
    version="0.1.0",
)

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=500,
        description="Text to classify. Only the first ~128 tokens are used by the model."
    )


class PredictionResponse(BaseModel):
    age_group: str
    confidence: str
    text: str


class ReasoningResponse(BaseModel):
    status: str
    text: str | None = None
    error: str | None = None


class PredictResponse(BaseModel):
    prediction: PredictionResponse
    reasoning: ReasoningResponse


@app.exception_handler(ModelNotFoundError)
def model_not_found_handler(_request: Request, exc: ModelNotFoundError):
    logger.error("Model not available: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def main(request: PredictionRequest) -> PredictResponse:
    user_input = request.text

    prediction = predict_age(user_input)
    reasoning: ReasoningResult = get_reasoning_result(prediction)

    return PredictResponse(
        prediction=PredictionResponse(**prediction.to_dict()),
        reasoning=ReasoningResponse(**reasoning.to_dict())
    )