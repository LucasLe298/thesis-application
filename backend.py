from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import warnings
from transformers import TrainingArguments

# Suppress FutureWarning for clean_up_tokenization_spaces (optional)
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Serve index.html at root
@app.get("/")
async def read_root():
    return FileResponse("index.html")

# Ekman emotion mapping
ekman_mapping = {
    "anger": [2, 3, 10], # anger, annoyance, disapproval*/
    "disgust": [11], # disgust
    "fear": [14, 19], # fear, nervousness
    "joy": [17, 1, 4, 13, 15, 18, 20, 23, 21, 0, 8, 5], # joy, amusement, approval, excitement, gratitude, love, optimism, pride, relief, caring, admiration, desire
    "sadness": [25, 9, 12, 16, 24], # sadness, disappointment, embarrassment, grief, remorse
    "surprise": [26, 22, 6, 7], # surprise, realization, curiosity, confusion
    "neutral": [27] # neutral
}

# Should use thresholds from your best performing fold
roberta_thresholds = [0.36, 0.44, 0.37, 0.54, 0.39, 0.40, 0.225]
                    # anger, disgust, fear, joy,  sadness, surprise, neutral
                    # 0.31,    0.39,  0.54,  0.56,  0.52,     0.31,   0.20
# RoBERTa + Attention model
class RoBERTa_Attention(nn.Module):
    def __init__(self, num_labels, freeze_layers=0):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.freeze_encoder_layers(freeze_layers)  # Freezes first 2 layers
        self.attention = nn.Linear(768, 1)         # Attention mechanism
        self.fc = nn.Linear(768, num_labels)       # Final classification layer

    def freeze_encoder_layers(self, num_layers):
        for i in range(num_layers):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        weights = torch.softmax(self.attention(hidden_states), dim=1)
        context_vector = torch.sum(weights * hidden_states, dim=1)
        logits = self.fc(context_vector)
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return {"loss": loss, "logits": logits}
        return logits

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load RoBERTa model and tokenizer
ROBERTA_MODEL_CKPT = "./Roberta/fold_1/checkpoint-3255"
ROBERTA_TOKENIZER_CKPT = "./Roberta/fold_1"
roberta_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_TOKENIZER_CKPT)
roberta_model = RoBERTa_Attention(num_labels=len(ekman_mapping))
roberta_model.to(device)
roberta_model.eval()

class TextInput(BaseModel):
    text: str

# Only use RoBERTa thresholds
def predict_emotions(text, model, tokenizer, thresholds):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    all_probs = []
    above_threshold = []
    for i, (emotion, prob) in enumerate(zip(ekman_mapping.keys(), probs)):
        all_probs.append({"emotion": emotion, "probability": float(prob), "threshold": thresholds[i]})
        if prob > thresholds[i]:
            above_threshold.append({"emotion": emotion, "probability": float(prob), "threshold": thresholds[i]})
    return {"all_probabilities": all_probs, "above_threshold": above_threshold}

@app.post("/predict")
async def predict(input_data: TextInput):
    try:
        result = predict_emotions(input_data.text, roberta_model, roberta_tokenizer, roberta_thresholds)
        return {
            "text": input_data.text,
            "model": "roberta",
            "all_probabilities": result["all_probabilities"],
            "above_threshold": result["above_threshold"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
