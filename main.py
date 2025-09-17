from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    text: str
    prediction: str
    confidence: float

@app.get("/")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "message": "Sentiment Analysis API is running"}

@app.post("/evaluate", response_model=PredictionOutput)
async def evaluate(input_data: TextInput) -> PredictionOutput:
    prediction = evaluate_sentiment(input_data.text)
    return prediction

def evaluate_sentiment(text: str) -> PredictionOutput:
    # Placeholder implementation
    return PredictionOutput(
        text=text,
        prediction="positive",
        confidence=0.85
    )