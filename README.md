# Emotion Recognition System

A modern emotion recognition system that uses RoBERTa model to predict emotions in text

## Features

- Real-time emotion prediction:
  - RoBERTa model with attention mechanism
- Modern interface
- FastAPI backend for model inference
- Support for multiple emotions:
  - Anger
  - Disgust
  - Fear
  - Joy
  - Sadness
  - Surprise
  - Neutral

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```


   
2. Run the FastAPI backend:
```bash
python backend.py
```


## Project Structure

- `backend.py` - FastAPI backend with model inference
- `index.html` - Frontend interface
- `requirements.txt` - Python dependencies
- `Roberta/` - Directory containing RoBERTa model files

## API Endpoints

- `POST /predict` - Send text and get emotion predictions 
  - Input: `{"text": "your text here"}`
  - Output: Predictions from RoBERTa

