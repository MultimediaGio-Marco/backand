import cv2
import numpy as np
from deapMap import relative_depth_map,colorize_depth_map  # la tua funzione esistente
from pathlib import Path
class DeapMapProcessor:
    def __init__(self, left_path: str | Path, right_path: str | Path):
        self.left_path = left_path
        self.right_path = right_path

    def _relative_depth_map(self) -> np.ndarray:
        """
        Calcola una mappa di profondità relativa (valori alti = oggetti vicini),
        normalizzata tra 0 e 1.
        """
        return relative_depth_map(self.left_path, self.right_path)

    def _colorize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Converte una mappa di profondità normalizzata (float 0-1) in immagine colorata BGR uint8.
        """
        return colorize_depth_map(depth_map)
    
    def _extract_mask_depth_map(self,depth_map):
        # Trova la profondità dominante (modalità dell'istogramma)
        hist = cv2.calcHist([depth_map], [0], None, [256], [0,256])
        dominant_depth = np.argmax(hist)

        # Maschera sulla profondità dominante ±10
        lower = np.clip(dominant_depth - 10, 0, 255)
        upper = np.clip(dominant_depth + 10, 0, 255)
        mask = cv2.inRange(depth_map, int(lower), int(upper))

        mask = 255 - mask
    
        # Pulizia morfologica
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=5)
        #cv2.imshow("Median", mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big, iterations=2)

        return mask
    
    def process(self):
        depth_map = self._relative_depth_map()
        depth_color = self._colorize_depth_map(depth_map)
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        mask = self._extract_mask_depth_map(depth_uint8)
        return depth_map, depth_color, mask

def main():
    nomeImage = "-L_auoHUVsy3AxDvE-Lg_left"
    nomeImage = nomeImage.replace("_left", "")
    left_path = f"/home/giovanni/Uni/multimedia/progetto/Holopix50k/val/left/{nomeImage}_left.jpg"
    right_path = f"/home/giovanni/Uni/multimedia/progetto/Holopix50k/val/right/{nomeImage}_right.jpg"

    processor = DeapMapProcessor(left_path, right_path)
    depth_map, depth_color, mask = processor.process()

    left_img = cv2.imread(left_path)
    out = left_img.copy()

    cv2.imshow("Depth Map Colorata", depth_color)
    cv2.imshow("Mask", mask)
    cv2.imshow("Boxes on Left Image", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
