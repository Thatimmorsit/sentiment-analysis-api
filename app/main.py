from fastapi import FastAPI
from pydantic import BaseModel

# This is a placeholder for a real sentiment analysis model.
# In a real-world scenario, you would load a pre-trained model from a library
# like `transformers` and use it to analyze the text.
def analyze_sentiment(text: str):
    """Analyzes the sentiment of the given text."""
    # Placeholder sentiment analysis logic
    if "love" in text or "fantastic" in text:
        return {"sentiment": "positive", "confidence": 0.98}
    elif "hate" in text or "terrible" in text:
        return {"sentiment": "negative", "confidence": 0.95}
    else:
        return {"sentiment": "neutral", "confidence": 0.6}

app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple API for analyzing the sentiment of text.",
    version="1.0.0",
)

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

@app.post("/analyze", response_model=SentimentResponse)
def analyze(request: SentimentRequest):
    """Analyzes the sentiment of a given text."""
    return analyze_sentiment(request.text)
