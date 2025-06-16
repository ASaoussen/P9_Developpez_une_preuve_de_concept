from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from azure.storage.blob import BlobServiceClient
import os

# 🔐 Charger depuis les variables d'environnement (définies dans GitHub Actions)
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# Liste des fichiers nécessaires pour le modèle
BLOB_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "pytorch_model.bin",
    "config.json"
]

# Répertoire temporaire local
LOCAL_MODEL_DIR = "./temp_model"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# --- Fonction pour télécharger les fichiers depuis Azure Blob Storage ---
def download_model_from_azure():
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    for blob_name in BLOB_FILES:
        blob_client = container_client.get_blob_client(blob_name)
        download_path = os.path.join(LOCAL_MODEL_DIR, blob_name)

        with open(download_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

    print("✅ Modèle téléchargé depuis Azure Blob Storage.")

# Télécharger au démarrage
download_model_from_azure()

# --- Modèle personnalisé ---
class CustomClassificationHead(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.3, num_labels=2):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        return self.classifier(self.dropout(x))

class CustomModernBERTModel(nn.Module):
    def __init__(self, model_name_or_path, num_labels=2, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = CustomClassificationHead(hidden_size, dropout_rate, num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

# --- Chargement du modèle et du tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)

model = CustomModernBERTModel(LOCAL_MODEL_DIR)
model.load_state_dict(torch.load(os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin"), map_location="cpu"))
model.eval()

# --- API FastAPI ---
app = FastAPI()

# Endpoint de bienvenue
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de classification de sentiment avec ModernBERT 🚀"}

# Endpoint de prédiction
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: InputText):
    tokens = tokenizer(input.text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        logits = model(**tokens)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
    return {"prediction": prediction, "confidence": round(confidence, 4)}
