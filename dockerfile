FROM python:3.12.3-slim

# Installa dipendenze di sistema minime (aggiungi quelle che servono al tuo progetto)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia l'app
COPY ./riconoscitore_flask /app
WORKDIR /app

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta 5000
EXPOSE 5000

# Comando di avvio
CMD ["python", "app.py"]
