# Emotion Recognition System

A modern emotion recognition system that uses both RoBERTa and Hybrid models to predict emotions in text.

## Features

- Real-time emotion prediction using two models:
  - RoBERTa model with attention mechanism
  - Hybrid model combining RoBERTa and Longformer with cross-attention
- Modern and responsive Streamlit interface
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

2. Make sure your pre-trained models are in the correct locations:
   - RoBERTa model: `Roberta/model.pth`
   - Hybrid model: `Hybrid/model.pth`

3. Run the FastAPI backend:
```bash
python backend.py
```

4. In a separate terminal, run the Streamlit frontend:
```bash
streamlit run frontend.py
```

## Project Structure

- `backend.py` - FastAPI backend with model inference
- `frontend.py` - Streamlit frontend interface
- `requirements.txt` - Python dependencies
- `Roberta/` - Directory containing RoBERTa model files
- `Hybrid/` - Directory containing Hybrid model files

## API Endpoints

- `POST /predict` - Send text and get emotion predictions from both models
  - Input: `{"text": "your text here"}`
  - Output: Predictions from both RoBERTa and Hybrid models

## Note

Make sure you have the pre-trained model files in the correct locations before running the application. The models should be saved as:
- `Roberta/model.pth`
- `Hybrid/model.pth` 