from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    pipeline
)
import torch
import os
import threading

BASE_DIR = "./models"
os.makedirs(BASE_DIR, exist_ok=True)

# Crea un semaforo per limitare il numero di thread attivi
sema = threading.Semaphore(2)  # massimo 2 thread contemporanei

def download_blip_base():
    with sema:
        print("⬇️ Scarico BLIP base...")
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=f"{BASE_DIR}/blip-base")
        BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=f"{BASE_DIR}/blip-base")

def download_blip_large():
    with sema:
        print("⬇️ Scarico BLIP large...")
        BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=f"{BASE_DIR}/blip-large")
        BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=f"{BASE_DIR}/blip-large")

def download_blip2():
    with sema:
        print("⬇️ Scarico BLIP-2 OPT 2.7B...")
        Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=f"{BASE_DIR}/blip2-opt")
        Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=f"{BASE_DIR}/blip2-opt")

def download_instructblip():
    with sema:
        print("⬇️ Scarico InstructBLIP Vicuna 7B...")
        InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b", cache_dir=f"{BASE_DIR}/instructblip")
        InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", cache_dir=f"{BASE_DIR}/instructblip")

if __name__ == "__main__":
    print("🚀 Inizio download modelli...")
    threads = []
    threads.append(threading.Thread(target=download_blip_base))
    threads.append(threading.Thread(target=download_blip_large))
    if os.environ.get("Versione") == "V2":
        threads.append(threading.Thread(target=download_blip2))
        threads.append(threading.Thread(target=download_instructblip))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("✅ Tutti i modelli sono stati scaricati in ./models")
