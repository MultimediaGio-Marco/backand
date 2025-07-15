import cv2
import numpy as np

def grabcut_bounding_box(img, iter_count=5):
    """
    Applica GrabCut all'immagine e ne estrae il bounding box del foreground.
    Restituisce la maschera binaria e il rettangolo (x, y, w, h).
    """
    h, w = img.shape[:2]
    # Rettangolo iniziale ampio per GrabCut (copre quasi tutta l'immagine)
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)

    # Maschera definitiva: 255 per foreground, 0 per background
    final_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')

    # Trova i contorni del foreground e ne estrae il bounding box maggiore
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(c)
    else:
        # Fallback: rettangolo iniziale
        x, y, w_box, h_box = rect

    return final_mask, (x, y, w_box, h_box)
