#!/bin/bash

# Activer l'environnement virtuel s'il existe
if [ -d "antenv" ]; then
    source antenv/bin/activate
fi

# Installer les dépendances (en cas de build partiel)
pip install --upgrade pip
pip install -r requirements.txt

# Lancer l'application avec Gunicorn + Uvicorn worker
exec gunicorn -k uvicorn.workers.UvicornWorker main:app --bind=0.0.0.0:8000 --timeout 120
