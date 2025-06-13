import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
import os
import warnings
import logging
from datetime import datetime
from safetensors.torch import load_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend.log')
    ]
)
logger = logging.getLogger(__name__)

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
    try:
        logger.info("Serving index.html")
        return FileResponse("index.html")
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving index page")

# Input validation
class TextInput(BaseModel):
    username: str
    text: str
    
    @validator('text')
    def text_length(cls, v):
        if len(v) > 512:
            raise ValueError('Text must be less than 512 characters')
        return v

# Load config with error handling
try:
    logger.info("Loading config file...")
    with open("config/roberta_backend_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    labels = config["labels"]
    label_fold_map = config["label_fold_map"]
    thresholds = config["thresholds"]
    model_paths = config["model_paths"]
    logger.info("Config loaded successfully")
except FileNotFoundError:
    logger.error("Config file not found")
    raise
except json.JSONDecodeError:
    logger.error("Invalid JSON in config file")
    raise
except Exception as e:
    logger.error(f"Error loading config: {str(e)}")
    raise

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define model class
class RoBERTa_Attention(nn.Module):
    def __init__(self, num_labels=7, freeze_layers=0):
        super().__init__()
        try:
            self.roberta = RobertaModel.from_pretrained("roberta-base")
            self.freeze_encoder_layers(freeze_layers)
            self.attention = nn.Linear(768, 1)
            self.fc = nn.Linear(768, num_labels)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def freeze_encoder_layers(self, num_layers):
        for i in range(num_layers):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        try:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            weights = torch.softmax(self.attention(hidden_states), dim=1)
            context_vector = torch.sum(weights * hidden_states, dim=1)
            logits = self.fc(context_vector)
            return logits
        except Exception as e:
            logger.error(f"Error in model forward pass: {str(e)}")
            raise

# Load models with error handling
models = {}
try:
    logger.info("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("./Roberta/fold_1/")
    
    logger.info("Loading models...")
    for fold_num, model_path in model_paths.items():
        try:
            logger.info(f"Loading model for fold {fold_num} from {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path not found: {model_path}")
            model = RoBERTa_Attention(num_labels=len(labels))
            state = load_file(model_path)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            models[int(fold_num)] = model
            logger.info(f"Model for fold {fold_num} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model for fold {fold_num}: {str(e)}")
            raise
except Exception as e:
    logger.error(f"Error in model loading process: {str(e)}")
    raise

# In-memory message store
chat_history = []

# Prediction logic with error handling
@torch.no_grad()
def predict_emotions(text, models, tokenizer, label_fold_map, thresholds, labels):
    try:
        logger.info(f"Processing text: {text[:60]}...")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        final_probs = []
        for idx, label in enumerate(labels):
            try:
                fold = label_fold_map[label]
                if fold not in models:
                    raise ValueError(f"Model for fold {fold} not found")
                    
                model = models[fold]
                logits = model(**inputs)
                prob = torch.sigmoid(logits[0][idx]).item()
                threshold = thresholds[label]
                final_probs.append({
                    "emotion": label,
                    "probability": prob,
                    "threshold": threshold
                })
            except Exception as e:
                logger.error(f"Error processing emotion {label}: {str(e)}")
                raise
                
        # Sort by probability descending
        final_probs = sorted(final_probs, key=lambda x: x["probability"], reverse=True)
        logger.info("Prediction completed successfully")
        return final_probs
    except Exception as e:
        logger.error(f"Error in prediction process: {str(e)}")
        raise

# API endpoints with error handling
@app.post("/predict")
async def predict(input_data: TextInput):
    try:
        logger.info(f"Received prediction request from user: {input_data.username}")
        sorted_probs = predict_emotions(
            input_data.text,
            models,
            tokenizer,
            label_fold_map,
            thresholds,
            labels
        )
        message = {
            "username": input_data.username,
            "text": input_data.text,
            "emotions": sorted_probs,
            "timestamp": datetime.now().isoformat()
        }
        chat_history.append(message)
        logger.info("Prediction request completed successfully")
        return message
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/messages")
async def get_messages():
    try:
        logger.info("Retrieving chat history")
        return chat_history
    except Exception as e:
        logger.error(f"Error retrieving messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run locally with error handling
if __name__ == "__main__":
    try:
        import uvicorn
        logger.info("Starting server...")
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise