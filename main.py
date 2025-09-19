import os
import sys

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ValidationError, field_validator

sys.path.append(os.path.dirname(__file__))
from config import setup_logger
from training.evaluate import inference

app = FastAPI(
    title="Sentiment Analysis API",
    version="v0.0.1",
    description="An app that performs binary sentiment classification on tweets",
)
logger = setup_logger(__name__, "YELLOW")

VERBOSITY = False


class TextInput(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not isinstance(v, str):
            raise ValueError('Text must be a string')
        if len(v.strip()) == 0:
            raise ValueError('Text should not be empty')
        return v


class PredictionOutput(BaseModel):
    text: str | None
    prediction: str
    confidence: float


class ErrorResponse(BaseModel):
    error: str
    detail: str


@app.get("/")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "message": "Sentiment Analysis API is running"}


@app.post("/evaluate", response_model=PredictionOutput)
def evaluate(input_data: TextInput, verbose: bool = Query(True)) -> PredictionOutput:
    try:
        text = input_data.text.strip()

        sentiment, confidence, cleaned_text = inference(text)
        if verbose:
            logger.info(f"Predicted sentiment: {sentiment.lower()} with confidence: {confidence}")
            logger.info(f"Cleaned Text: {cleaned_text}")
        return PredictionOutput(
            text=cleaned_text, prediction=sentiment.lower(), confidence=confidence
        )

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail="Unprocessable entity") from e

    except HTTPException:
        raise

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(status_code=503, detail="Model temporarily un-available.") from e

    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
