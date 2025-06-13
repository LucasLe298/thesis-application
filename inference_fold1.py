import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, RobertaModel
from safetensors.torch import load_file
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve
)

# ===== DEVICE & SEED =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
if device.type == "cuda":
    print("GPU IS AVAILABLE")

# ===== EKMAN LABELS & MAPPING =====
ekman_labels = ["anger","disgust","fear","joy","sadness","surprise","neutral"]
ekman_mapping = {
    "anger":    [2,3,10],
    "disgust":  [11],
    "fear":     [14,19],
    "joy":      [17,1,4,13,15,18,20,23,21,0,8,5],
    "sadness":  [25,9,12,16,24],
    "surprise": [26,22,6,7],
    "neutral":  [27]
}
label2ekman = {lid: idx for idx, (_, ids) in enumerate(ekman_mapping.items()) for lid in ids}

def map_to_ekman(label_ids_str: str):
    vec = [0]*7
    for lid_str in label_ids_str.split(","):
        if lid_str.strip().isdigit():
            lid = int(lid_str)
            if lid in label2ekman:
                vec[label2ekman[lid]] = 1
    return vec

def load_dataset(path: str):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: continue
            vec = map_to_ekman(parts[1])
            if sum(vec)==0: continue
            texts.append(parts[0])
            labels.append(vec)
    return texts, labels

def vec2labels(vec):
    return [ekman_labels[i] for i, val in enumerate(vec) if val == 1]

# ===== MODEL DEFINITION =====
class RoBERTa_Attention(nn.Module):
    def __init__(self, num_labels: int, freeze_layers: int = 2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.freeze_encoder_layers(freeze_layers)
        hidden_size = self.roberta.config.hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_labels)

    def freeze_encoder_layers(self, num_layers: int):
        for i in range(num_layers):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        weights = torch.softmax(self.attention(hidden_states), dim=1)
        context_vector = torch.sum(weights * hidden_states, dim=1)
        return self.fc(context_vector)

# ===== DATASET CLASS =====
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):  
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_vec = torch.FloatTensor(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label_vec
        }

# ===== MAIN =====
if __name__ == "__main__":
    # 1. Load & filter dataset
    train_texts, train_labels = load_dataset("/kaggle/input/goemotions-raw/train.tsv")
    print(f"Loaded {len(train_texts)} samples from train.tsv")

    # 2. Split out fold 1 (giống trainer)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_texts), start=1):
        if fold_idx == 1:
            tr_texts = [train_texts[i] for i in train_idx]
            tr_labels = [train_labels[i] for i in train_idx]
            val_texts = [train_texts[i] for i in val_idx]
            val_labels = [train_labels[i] for i in val_idx]
            break
    print(f"Fold 1: {len(val_texts)} validation samples")

    # Kiểm tra lại sample đầu tiên
    print("First 3 validation samples:")
    for i in range(3):
        print(val_texts[i], val_labels[i])

    # 3. Load thresholds.json (đúng fold 1)
    th_path = "/kaggle/input/fold-1-model-summary/pytorch/default/1/thresholds.json"
    with open(th_path, "r") as f:
        threshold_dict = json.load(f)
    thresholds = np.array([threshold_dict[l] for l in ekman_labels])
    print("Thresholds:", thresholds)
    print("Ekman labels:", ekman_labels)

    # 4. Prepare DataLoader
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_len=512)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 5. Load model + weights
    checkpoint = "/kaggle/input/fold-1-model-summary/pytorch/default/1/model.safetensors"
    model = RoBERTa_Attention(num_labels=7, freeze_layers=2)
    state = load_file(checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()

    # 6. Inference & collect
    probs, predictions, val_labels_collect = [], [], []
    examples = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inferencing fold 1"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids, attention_mask)
            probs_batch = torch.sigmoid(logits).cpu().numpy()
            predictions_batch = (probs_batch > thresholds).astype(int)

            probs.append(probs_batch)
            predictions.append(predictions_batch)
            val_labels_collect.append(labels)

            for i in range(len(predictions_batch)):
                if len(examples) < 20:
                    examples.append({
                        "text": val_texts[i][:100] + ("…" if len(val_texts[i]) > 100 else ""),
                        "true": labels[i].tolist(),
                        "pred": predictions_batch[i].tolist(),
                        "probs": probs_batch[i].tolist()
                    })

    # 7. Stack arrays
    probs = np.vstack(probs)
    predictions = np.vstack(predictions)
    val_labels = np.vstack(val_labels_collect)

    # 8. F1 per label via precision_recall_curve
    print("\n=== Best F1 & threshold per label ===")
    for i, lbl in enumerate(ekman_labels):
        p, r, t = precision_recall_curve(val_labels[:, i], probs[:, i])
        f1_scores = 2 * p * r / (p + r + 1e-8)
        best = np.nanargmax(f1_scores)
        print(f"{lbl:10s} | best F1 = {f1_scores[best]:.4f} at thresh = {t[best]:.4f}")

    # 9. Classification report
    print("\n=== Classification Report F1's ===")
    report_dict = classification_report(
        val_labels,
        predictions,
        target_names=ekman_labels,
        zero_division=0,
        output_dict=True
    )
    for lbl in ekman_labels:
        print(f"{lbl:10s} | F1-score = {report_dict[lbl]['f1-score']:.4f}")

    print("\n=== Full Classification Report ===")
    print(classification_report(val_labels, predictions, target_names=ekman_labels, zero_division=0))

    # 10. Match statistics
    exact = np.sum(np.all(val_labels == predictions, axis=1))
    partial = np.sum(np.logical_and(val_labels, predictions).sum(axis=1) > 0) - exact
    total = val_labels.shape[0]
    no = total - exact - partial

    exact_pct = exact / total * 100
    partial_pct = partial / total * 100
    no_pct = no / total * 100

    print("\n=== Match Statistics ===")
    print(f"  Exact Match  : {exact_pct:.2f}% ({exact}/{total})")
    print(f"  Partial Match: {partial_pct:.2f}% ({partial}/{total})")
    print(f"  No Match     : {no_pct:.2f}% ({no}/{total})")
    
    # 11. Example cases
    print("\n=== Some Examples with Labels ===")
    for ex in examples:
        true_vec = np.array(ex["true"])
        pred_vec = np.array(ex["pred"])
        true_names = vec2labels(true_vec)
        pred_names = vec2labels(pred_vec)
        prob_dict = {}
        for i, v in enumerate(pred_vec):
            if v == 1:
                prob_dict[ekman_labels[i]] = round(float(ex.get("probs", [0]*7)[i]), 4)
        if np.array_equal(true_vec, pred_vec):
            match_type = "Exact Match"
        elif np.any(np.logical_and(true_vec, pred_vec)):
            match_type = "Partial Match"
        else:
            match_type = "No Match"
        print(f"Text:  {ex['text']}")
        print(f" True vector : {true_vec.tolist()}  → {true_names}")
        print(f" Pred vector : {pred_vec.tolist()}  → {pred_names}")
        print(f" Probability     : {prob_dict}")
        print(f" Match    : {match_type}\n")

    # 12. Save output
    df_out = pd.DataFrame({
        "text": val_texts,
        "true_label": val_labels.tolist(),
        "pred_label": predictions.tolist()
    })
    df_out.to_csv("/kaggle/working/fold1_val_predictions.csv", index=False)
    print("Saved fold1_val_predictions.csv") 