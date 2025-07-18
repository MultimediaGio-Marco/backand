#!/bin/bash

# Configurazioni
IMAGE_NAME="riconoscitore_flask"
TAG="RV1_latest"
FULL_LOCAL_TAG="${IMAGE_NAME}:${TAG}"
DOCKERHUB_REPO="oligiovi/${IMAGE_NAME}:${TAG}"

echo "🚧 Costruzione immagine locale: ${FULL_LOCAL_TAG}"
docker build -t ${FULL_LOCAL_TAG} .

if [ $? -ne 0 ]; then
  echo "❌ Build fallita. Interrotto."
  exit 1
fi

echo "🏷  Tagging immagine: ${DOCKERHUB_REPO}"
docker tag ${FULL_LOCAL_TAG} ${DOCKERHUB_REPO}

echo "📤 Push verso Docker Hub: ${DOCKERHUB_REPO}"
docker push ${DOCKERHUB_REPO}

if [ $? -eq 0 ]; then
  echo "✅ Push completato con successo!"
else
  echo "❌ Errore nel push!"
fi
