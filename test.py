from azure.storage.blob import BlobServiceClient
import os

# ⚙️ Paramètres
connect_str = "DefaultEndpointsProtocol=https;AccountName=p8blob;AccountKey=jtfgMlh2QSMN60CydwDTOWMS1L2726/4N8Dhbvg4cNe/JQ6x8YmwHRdiNP3Igk1GA2AMtM41Fcam+AStl5pRog==;EndpointSuffix=core.windows.net"
container_name = "container2"
local_model_dir = "C:/Users/attia/mon_dossier_model"
file_name = "tokenizer.json"
blob_name = file_name  # 👈 ligne ajoutée

# 📍 Chemin complet vers le fichier tokenizer.json
tokenizer_path = os.path.join(local_model_dir, file_name)

# 📤 Connexion au blob Azure et upload
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

# 🔁 Upload du fichier
with open(tokenizer_path, "rb") as data:
    blob_client.upload_blob(data, overwrite=True)

print(f"{blob_name} uploadé avec succès.")
