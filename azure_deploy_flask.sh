#!/bin/bash

# Configurazioni
IMAGE_NAME="riconoscitore_flask"
TAG="RV1_latest"
FULL_LOCAL_TAG="${IMAGE_NAME}:${TAG}"
ACR_URI="oligiovi.azurecr.io"
ACR_REPO="${ACR_URI}/${IMAGE_NAME}:${TAG}"

echo "🚧 Costruzione immagine locale: ${FULL_LOCAL_TAG}"
docker build -t ${FULL_LOCAL_TAG} .

if [ $? -ne 0 ]; then
  echo "❌ Build fallita. Interrotto."
  exit 1
fi

echo "🔐 Login su Azure Container Registry: ${ACR_URI}"
docker login ${ACR_URI}

if [ $? -ne 0 ]; then
  echo "❌ Login fallito. Interrotto."
  exit 1
fi

echo "🏷  Tagging immagine per ACR: ${ACR_REPO}"
docker tag ${FULL_LOCAL_TAG} ${ACR_REPO}

echo "📤 Push verso Azure Container Registry: ${ACR_REPO}"
docker push ${ACR_REPO}

if [ $? -eq 0 ]; then
  echo "✅ Push completato con successo!"
else
  echo "❌ Errore nel push!"
fi
