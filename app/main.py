#!/usr/bin/env python3
"""
This script provides a functional implementation of a sentiment analysis API
using FastAPI and the Hugging Face Transformers library.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List

# --- Model and App Initialization ---

# Use a distilled, faster model for efficient inference
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

try:
    # Initialize the sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model=MODEL_NAME)
except Exception as e:
    # If model loading fails, we can't proceed. Raise an error.
    raise RuntimeError(f"Failed to load sentiment analysis model: {e}") from e

app = FastAPI(
    title="Sentiment Analysis API",
    description="A high-performance API for analyzing text sentiment using distilled BERT.",
    version="2.0.0",
)

# --- API Models ---

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float

class BatchSentimentRequest(BaseModel):
    texts: List[str]

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]

# --- API Endpoints ---

@app.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    """Analyzes the sentiment of a single text string."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        result = sentiment_analyzer(request.text)[0]
        return SentimentResponse(label=result["label"].lower(), score=result["score"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {e}")

@app.post("/analyze/batch", response_model=BatchSentimentResponse)
def analyze_batch_sentiment(request: BatchSentimentRequest):
    """
    Analyzes the sentiment of a batch of texts in a single request for efficiency.
    """
    if not request.texts or all(not text.strip() for text in request.texts):
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")

    try:
        # The pipeline is highly optimized for batch processing
        batch_results = sentiment_analyzer(request.texts)
        
        # Format the results into the response model
        response_results = [
            SentimentResponse(label=res["label"].lower(), score=res["score"])
            for res in batch_results
        ]
        return BatchSentimentResponse(results=response_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during batch analysis: {e}")

@app.get("/health")
def health_check():
    """Provides a simple health check endpoint."""
    return {"status": "ok", "model_name": MODEL_NAME}
