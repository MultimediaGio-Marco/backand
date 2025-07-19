FROM python:3.12.3-slim

# Versione
ENV Versione=V1
ENV FLASK_ENV=production
ENV PORT=5000

# 1. Dipendenze di sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements e installa dipendenze Python
COPY ./riconoscitore_flask/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    --index-url https://pypi.python.org/simple/ \
    --trusted-host pypi.python.org \
    -r /tmp/requirements.txt

# 4. Copia e scarica tutti i modelli Transformers in /app/models
COPY ./download_models.py /app/download_models.py
WORKDIR /app
RUN python download_models.py

# 5. Copia il resto dell’app e setta WORKDIR
COPY ./riconoscitore_flask /app
WORKDIR /app

EXPOSE 5000

# Avvio con echo per debug e poi con Python
ENTRYPOINT ["sh", "-c", "echo \"🚀 Starting Flask app (Versione $Versione) on port $PORT\" && exec python app.py"]
