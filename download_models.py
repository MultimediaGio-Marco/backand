from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    pipeline
)
import torch
import os

BASE_DIR = "./models"
os.makedirs(BASE_DIR, exist_ok=True)

def download_blip_base():
    print("⬇️ Scarico BLIP base...")
    BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=f"{BASE_DIR}/blip-base")
    BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=f"{BASE_DIR}/blip-base")

def download_blip_large():
    print("⬇️ Scarico BLIP large...")
    BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=f"{BASE_DIR}/blip-large")
    BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=f"{BASE_DIR}/blip-large")

def download_blip2():
    print("⬇️ Scarico BLIP-2 OPT 2.7B...")
    Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=f"{BASE_DIR}/blip2-opt")
    Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=f"{BASE_DIR}/blip2-opt")

def download_instructblip():
    print("⬇️ Scarico InstructBLIP Vicuna 7B...")
    InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b", cache_dir=f"{BASE_DIR}/instructblip")
    InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", cache_dir=f"{BASE_DIR}/instructblip")

def download_vehicle_classifier():
    print("⬇️ Scarico classificatore veicoli (DIT)...")
    _ = pipeline("image-classification", model="microsoft/dit-base-finetuned-rvlcdip", 
                 cache_dir=f"{BASE_DIR}/dit", device=-1)

if __name__ == "__main__":
    download_blip_base()
    download_blip_large()
    download_blip2()
    download_instructblip()
    download_vehicle_classifier()
    print("✅ Tutti i modelli sono stati scaricati in ./models")
