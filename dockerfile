FROM python:3.12.3-slim
ENV Versione=V1
# 1. Dipendenze di sistema (cambiano raramente)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Copia solo requirements.txt e installa dipendenze Python
COPY ./riconoscitore_flask/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    --index-url https://pypi.python.org/simple/ \
    --trusted-host pypi.python.org \
    -r /tmp/requirements.txt

# 3. Copia script per scaricare modelli e scarica modelli (se necessario)
COPY ./download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py

# 4. Copia tutto il codice app (qui inizia la parte più volatile)
COPY ./riconoscitore_flask /app

WORKDIR /app

EXPOSE 5000

CMD ["python", "app.py"]
