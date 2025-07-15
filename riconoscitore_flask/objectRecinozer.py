from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2
import numpy as np
from find_box_pipeline import pipeline_image_detector,pipeline_image_detector_stamp

class ObjectRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = (
            BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            .to(self.device)
        )
        self.model.eval()

    def recognize(self, left_img_path, right_img_path, debug=False):
        """
        Riconosce l'oggetto in coppia stereo e restituisce una tupla (label, bbox).
        """
        # Calcola bounding box con la pipeline stereo
        pipeline_image_detector_stamp(left_img_path, right_img_path)
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

        # Converti ritaglio in PIL e genera didascalia
        image_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        label = self.caption_image(image_pil)

        # Ritorna etichetta e box
        return label, bbox

    def caption_image(self, image_pil):
        inputs = self.processor(image_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
