# Sentiment Analysis API

A RESTful API for performing sentiment analysis on text using a pre-trained transformer model. This project provides a simple and efficient way to integrate sentiment analysis capabilities into your applications.

## Features

*   **Fast and Accurate**: Utilizes a fine-tuned transformer model for high-performance sentiment analysis.
*   **Easy to Use**: A simple RESTful API with a single endpoint for analyzing text.
*   **Scalable**: Built with FastAPI for high-performance, asynchronous request handling.
*   **Dockerized**: Comes with a Dockerfile for easy deployment and scaling.

## API Endpoints

### POST /analyze

Analyzes the sentiment of a given text.

**Request Body:**

```json
{
  "text": "I love using this sentiment analysis API! It's fantastic."
}
```

**Response:**

```json
{
  "sentiment": "positive",
  "confidence": 0.98
}
```

## Getting Started

### Prerequisites

*   Python 3.8+
*   Docker (optional, for containerized deployment)

### Installation

```bash
git clone https://github.com/Thatimmorsit/sentiment-analysis-api.git
cd sentiment-analysis-api
pip install -r requirements.txt
```

### Running the API

```bash
uvicorn app.main:app --reload
```
