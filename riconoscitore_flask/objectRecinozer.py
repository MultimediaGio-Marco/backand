from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2
import numpy as np
import re
from find_box_pipeline import pipeline_image_detector, pipeline_image_detector_stamp

class ObjectRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Carica dalla cache in /app/models
        try:
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large",
                cache_dir="models/blip-large",
                use_fast=True
            )
            self.model = (
                BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-large",
                    cache_dir="models/blip-large"
                )
                .to(self.device)
            )
        except Exception:
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir="models/blip-base",
                use_fast=True
            )
            self.model = (
                BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    cache_dir="models/blip-base"
                )
                .to(self.device)
            )
        
        self.model.eval()

    def clean_caption(self, caption):
        """Pulisce la didascalia da ripetizioni e artefatti"""
        if not caption:
            return ""
        
        words = caption.split()
        cleaned_words = []
        prev_word = ""
        repeat_count = 0
        
        for word in words:
            if word.lower() == prev_word.lower():
                repeat_count += 1
                if repeat_count <= 1:
                    cleaned_words.append(word)
            else:
                cleaned_words.append(word)
                repeat_count = 0
            prev_word = word.lower()
        
        cleaned_caption = " ".join(cleaned_words)
        cleaned_caption = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', cleaned_caption, flags=re.IGNORECASE)
        cleaned_caption = re.sub(r'[^\w\s-]', '', cleaned_caption)
        cleaned_caption = re.sub(r'\s+', ' ', cleaned_caption).strip()
        
        return cleaned_caption

    def generate_with_retry(self, image_pil, max_retries=3):
        """Genera didascalia con retry"""
        for attempt in range(max_retries):
            try:
                inputs = self.processor(image_pil, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_length=25,
                        num_beams=5,
                        early_stopping=True,
                        do_sample=False,
                        repetition_penalty=1.3,
                        length_penalty=1.0,
                        temperature=0.8 if attempt > 0 else 1.0
                    )
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                cleaned = self.clean_caption(caption)
                if len(cleaned.split()) >= 2:
                    return cleaned
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Errore nella generazione: {e}")
        return "oggetto non identificato"

    def recognize(self, left_img_path, right_img_path, debug=False):
        bbox = pipeline_image_detector(left_img_path, right_img_path)
        if bbox is None:
            raise RuntimeError("Nessun oggetto rilevato.")
        x, y, w_box, h_box = bbox
        img = cv2.imread(left_img_path)
        cropped = img[y : y + h_box, x : x + w_box]
        image_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        caption = self.generate_with_retry(image_pil)
        return caption, bbox

    def caption_image(self, image_pil):
        """Metodo di compatibilità con la versione precedente"""
        return self.generate_with_retry(image_pil)


# Versione alternativa con modello più recente
class ObjectRecognizerV2:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Prova modelli più recenti in ordine di preferenza
        model_configs = [
            ("Salesforce/blip2-opt-2.7b", "Blip2Processor", "Blip2ForConditionalGeneration"),
            ("Salesforce/instructblip-vicuna-7b", "InstructBlipProcessor", "InstructBlipForConditionalGeneration"),
            ("Salesforce/blip-image-captioning-large", "BlipProcessor", "BlipForConditionalGeneration"),
            ("Salesforce/blip-image-captioning-base", "BlipProcessor", "BlipForConditionalGeneration")
        ]
        
        self.model_loaded = False
        for model_name, processor_class, model_class in model_configs:
            try:
                from transformers import AutoProcessor, AutoModelForSeq2SeqLM
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                self.model.eval()
                self.model_loaded = True
                print(f"Caricato modello: {model_name}")
                break
            except:
                continue
        
        if not self.model_loaded:
            raise RuntimeError("Impossibile caricare nessun modello")

    def recognize(self, left_img_path, right_img_path, debug=False):
        """Versione con modello più recente"""
        bbox = pipeline_image_detector(left_img_path, right_img_path)
        if bbox is None:
            raise RuntimeError("Nessun oggetto rilevato nelle immagini stereo.")

        x, y, w_box, h_box = bbox
        img = cv2.imread(left_img_path)
        cropped = img[y : y + h_box, x : x + w_box]

        if debug:
            cv2.imshow("Originale", img)
            cv2.imshow("Ritaglio", cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        image_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        inputs = self.processor(image_pil, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=30,
                num_beams=5,
                early_stopping=True,
                repetition_penalty=1.5,
                do_sample=False
            )
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Pulizia generica
        caption = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', caption, flags=re.IGNORECASE)
        caption = re.sub(r'\s+', ' ', caption).strip()
        
        return caption, bbox