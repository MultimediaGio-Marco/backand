FROM python:3.12.3-slim

# Installa dipendenze di sistema minime
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
COPY ./download_models.py /tmp/download_models.py
WORKDIR /app

# Installa le dipendenze Python usando mirror italiano/europeo
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    --index-url https://pypi.python.org/simple/ \
    --trusted-host pypi.python.org \
    -r requirements.txt


#RUN python /tmp/download_models.py && rm /tmp/download_models.py


# Espone la porta 5000
EXPOSE 5000

# Comando di avvio
CMD ["python", "app.py"]