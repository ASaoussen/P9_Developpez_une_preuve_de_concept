from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel, AutoConfig
from azure.storage.blob import BlobServiceClient
import torch
import torch.nn as nn
import os

# --- Configuration Azure depuis variable d’environnement ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

if AZURE_CONNECTION_STRING is None:
    raise RuntimeError("Variable d'environnement AZURE_CONNECTION_STRING manquante.")

# --- Fichiers requis à télécharger ---
BLOB_FILES = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "special_tokens_map.json"
]

# --- Répertoire local temporaire pour stocker les fichiers du modèle ---
LOCAL_MODEL_DIR = "./temp_model"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# --- Téléchargement depuis Azure ---
def download_model_from_azure():
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        for blob_name in BLOB_FILES:
            blob_client = container_client.get_blob_client(blob_name)
            download_path = os.path.join(LOCAL_MODEL_DIR, blob_name)
            with open(download_path, "wb") as f:
                f.write(blob_client.download_blob().readall())

        print(" Modèle téléchargé depuis Azure Blob Storage.")
    except Exception as e:
        print(f" Erreur Azure : {e}")
        raise RuntimeError("Échec du téléchargement depuis Azure.")

# --- Validation des fichiers téléchargés ---
def validate_model_files():
    missing = [f for f in BLOB_FILES if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, f))]
    if missing:
        raise FileNotFoundError(f"Fichiers manquants : {missing}")

# --- Classe du modèle personnalisé ---
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

# --- Initialisation du modèle ---
download_model_from_azure()
validate_model_files()

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, use_fast=False)
config = AutoConfig.from_pretrained(LOCAL_MODEL_DIR)
num_labels = getattr(config, "num_labels", 2)

model = CustomModernBERTModel(LOCAL_MODEL_DIR, num_labels=num_labels)
model_path = os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

print(" Modèle chargé et prêt.")

# --- Initialisation Flask ---
app = Flask(__name__)

label_map = {
    0: "Négatif",
    1: "Positif"
}

# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Bienvenue sur l'API de classification de sentiment."})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data or not data["text"].strip():
        return jsonify({"error": "Le texte ne peut pas être vide."}), 400

    text = data["text"]
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )

    with torch.no_grad():
        logits = model(**tokens)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

    return jsonify({
        "prediction": prediction,
        "label": label_map.get(prediction, "Inconnu"),
        "confidence": round(confidence, 4)
    })

# --- Lancement local ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
