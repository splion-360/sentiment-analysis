import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))
from config import setup_logger
from training.evaluate import inference

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")
logger = setup_logger(__name__, "YELLOW")


class TextInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    text: str
    prediction: str
    confidence: float


class ErrorResponse(BaseModel):
    error: str
    detail: str


@app.get("/")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "message": "Sentiment Analysis API is running"}


@app.post("/evaluate", response_model=PredictionOutput)
def evaluate(input_data: TextInput) -> PredictionOutput:
    try:
        text = input_data.text.strip()

        if not text:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")

        sentiment, confidence, cleaned_text = inference(text)
        logger.info(f"Predicted sentiment: {sentiment.lower()} with confidence: {confidence}")

        return PredictionOutput(
            text=cleaned_text, prediction=sentiment.lower(), confidence=confidence
        )

    except HTTPException:
        raise

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=503, detail="Model temporarily un-available.") from e

    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
