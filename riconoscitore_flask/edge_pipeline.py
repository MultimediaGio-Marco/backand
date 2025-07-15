import cv2
import numpy as np

def preprocess(img):
    """Converti in scala di grigi e migliora il contrasto."""
    cv2.GaussianBlur(img, (5, 5), 0, img)
    Y,Cb,Cr = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
    Y = cv2.equalizeHist(Y)
    return Y

def compute_edges(gray):
    """Calcola le mappe di edge: Canny, Sobel (magnitudine), Laplaciano."""
    canny = cv2.Canny(gray, 50, 150)

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = cv2.convertScaleAbs(lap)

    return canny, mag, lap

def fuse_stereo_maps(maps_left, maps_right):
    """Fusione media pixel-per-pixel delle mappe stereo."""
    fused = [(l.astype(np.float32) + r.astype(np.float32)) / 2 for l, r in zip(maps_left, maps_right)]
    return [cv2.convertScaleAbs(m) for m in fused]

def final_fusion(fused_maps):
    """Media finale tra tutte le mappe fuse."""
    stacked = np.stack(fused_maps, axis=0).astype(np.float32)
    mean_map = np.mean(stacked, axis=0)
    return cv2.convertScaleAbs(mean_map)

def stereo_edge_enhanced(left_img, right_img):
    """
    Pipeline stereo edge fusion.
    Ritorna:
        - final_edges: mappa combinata
        - fused_canny, fused_mag, fused_laplacian: mappe intermedie fuse
        - (facoltativo) raw_left_maps, raw_right_maps: se servono
    """
    gray_left = preprocess(left_img)
    gray_right = preprocess(right_img)

    maps_left = compute_edges(gray_left)
    maps_right = compute_edges(gray_right)

    fused_maps = fuse_stereo_maps(maps_left, maps_right)
    fused_canny, fused_mag, fused_laplacian = fused_maps

    final_edges = final_fusion(fused_maps)

    return {
        "final": final_edges,
        "fused_canny": fused_canny,
        "fused_magnitude": fused_mag,
        "fused_laplacian": fused_laplacian,
        "maps_left": maps_left,
        "maps_right": maps_right
    }
    
def postprocess_edges(edge_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(edge_img, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened
