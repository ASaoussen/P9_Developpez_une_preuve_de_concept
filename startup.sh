#!/bin/bash

echo "Début du script startup.sh"

if [ -d "antenv" ]; then
    source antenv/bin/activate
    echo "Environnement activé"
else
    echo "Environnement virtuel antenv non trouvé"
fi

pip install --upgrade pip
pip install -r requirements.txt

exec gunicorn -k uvicorn.workers.UvicornWorker app:app --bind=0.0.0.0:8000 --timeout 120
