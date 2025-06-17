

# Lancer l'application avec Gunicorn + Uvicorn worker
exec gunicorn -k uvicorn.workers.UvicornWorker app:app --bind=0.0.0.0:8000 --timeout 120
