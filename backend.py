import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files and index
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def read_root():
    return FileResponse("index.html")

# Load config
with open("config/roberta_backend_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

labels = config["labels"]
label_fold_map = config["label_fold_map"]
thresholds = config["thresholds"]
model_paths = config["model_paths"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model class
class RoBERTa_Attention(nn.Module):
    def __init__(self, num_labels=7, freeze_layers=0):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.freeze_encoder_layers(freeze_layers)
        self.attention = nn.Linear(768, 1)
        self.fc = nn.Linear(768, num_labels)

    def freeze_encoder_layers(self, num_layers):
        for i in range(num_layers):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        weights = torch.softmax(self.attention(hidden_states), dim=1)
        context_vector = torch.sum(weights * hidden_states, dim=1)
        logits = self.fc(context_vector)
        return logits

# Load all models from config
models = {}
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
for fold_num, model_path in model_paths.items():
    try:
        model = RoBERTa_Attention(num_labels=len(labels))
        model.roberta = RobertaModel.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )
        model.to(device)
        model.eval()
        models[int(fold_num)] = model
    except Exception as e:
        print(f"Error loading model for fold {fold_num}: {str(e)}")
        raise

# Input class
class TextInput(BaseModel):
    text: str

# Prediction logic
@torch.no_grad()
def predict_emotions(text, models, tokenizer, label_fold_map, thresholds, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    final_probs = []
    above_threshold = []

    for idx, label in enumerate(labels):
        fold = label_fold_map[label]
        model = models[fold]
        logits = model(**inputs)
        prob = torch.sigmoid(logits[0][idx]).item()
        threshold = thresholds[label]
        final_probs.append({"emotion": label, "probability": prob, "threshold": threshold})
        if prob > threshold:
            above_threshold.append({"emotion": label, "probability": prob, "threshold": threshold})

    return {"all_probabilities": final_probs, "above_threshold": above_threshold}

# API endpoint
@app.post("/predict")
async def predict(input_data: TextInput):
    try:
        result = predict_emotions(input_data.text, models, tokenizer, label_fold_map, thresholds, labels)
        return {
            "text": input_data.text,
            "model": "roberta-multi-fold",
            "all_probabilities": result["all_probabilities"],
            "above_threshold": result["above_threshold"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)