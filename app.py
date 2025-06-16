from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from azure.storage.blob import BlobServiceClient
import os

# Azure Blob Storage config
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=p8blob;AccountKey=jtfgMlh2QSMN60CydwDTOWMS1L2726/4N8Dhbvg4cNe/JQ6x8YmwHRdiNP3Igk1GA2AMtM41Fcam+AStl5pRog==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "container2"
BLOB_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "pytorch_model.bin",
    "config.json"
]

# Dossier temporaire local pour stocker les fichiers téléchargés
LOCAL_MODEL_DIR = "./temp_model"

# Créer dossier s'il n'existe pas
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Fonction pour télécharger les blobs
def download_model_from_azure():
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    for blob_name in BLOB_FILES:
        blob_client = container_client.get_blob_client(blob_name)
        download_file_path = os.path.join(LOCAL_MODEL_DIR, blob_name)
        
        with open(download_file_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())
    print("Modèle téléchargé depuis Azure Blob Storage.")

# Télécharger les fichiers au démarrage de l'API
download_model_from_azure()

# Modèle personnalisé (comme dans ton code)
class CustomClassificationHead(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.3, num_labels=2):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        return self.classifier(self.dropout(x))

class CustomModernBERTModel(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout_rate=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = CustomClassificationHead(hidden_size, dropout_rate, num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits

# Charger tokenizer et modèle depuis le dossier temporaire
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
model = CustomModernBERTModel(LOCAL_MODEL_DIR)
model.load_state_dict(torch.load(os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin"), map_location="cpu"))
model.eval()

# FastAPI app
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: InputText):
    tokens = tokenizer(input.text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        logits = model(**tokens)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
    return {"prediction": pred, "confidence": round(confidence, 4)}
