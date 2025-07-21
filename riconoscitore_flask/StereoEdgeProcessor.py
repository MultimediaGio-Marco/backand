import cv2
import numpy as np
from pathlib import Path

class StereoEdgeProcessor:
    def __init__(self, left_path: str | Path, right_path: str | Path):
        self.left_path = left_path
        self.right_path = right_path

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Converti in scala di grigi e migliora il contrasto."""
        cv2.GaussianBlur(img, (5, 5), 0, img)
        Y, Cb, Cr = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
        Y = cv2.equalizeHist(Y)
        return Y

    def _compute_edges(self, gray: np.ndarray):
        """Calcola le mappe di edge: Canny, Sobel (magnitudine), Laplaciano."""
        canny = cv2.Canny(gray, 50, 150)

        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobelx, sobely)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap = cv2.convertScaleAbs(lap)

        return canny, mag, lap

    def _fuse_stereo_maps(self, maps_left, maps_right):
        """Fusione media pixel-per-pixel delle mappe stereo."""
        fused = [(l.astype(np.float32) + r.astype(np.float32)) / 2 for l, r in zip(maps_left, maps_right)]
        return [cv2.convertScaleAbs(m) for m in fused]

    def _final_fusion(self, fused_maps):
        """Media finale tra tutte le mappe fuse."""
        stacked = np.stack(fused_maps, axis=0).astype(np.float32)
        mean_map = np.mean(stacked, axis=0)
        return cv2.convertScaleAbs(mean_map)

    def _postprocess_edges(self, edge_img: np.ndarray) -> np.ndarray:
        """Morfologia per pulizia bordi."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(edge_img, cv2.MORPH_OPEN, kernel, iterations=1)
        return opened

    def process(self):
        """
        Applica la pipeline edge stereo fusion.
        Ritorna:
            - final: fusione totale dei bordi
            - fused_canny, fused_magnitude, fused_laplacian
            - raw grayscale maps (opzionale)
        """
        left_img = cv2.imread(str(self.left_path))
        right_img = cv2.imread(str(self.right_path))

        gray_left = self._preprocess(left_img)
        gray_right = self._preprocess(right_img)

        maps_left = self._compute_edges(gray_left)
        maps_right = self._compute_edges(gray_right)

        fused_maps = self._fuse_stereo_maps(maps_left, maps_right)
        fused_canny, fused_magnitude, fused_laplacian = fused_maps

        final_edges = self._final_fusion(fused_maps)
        #final_edges = self._postprocess_edges(final_edges)

        return {
            "final": final_edges,
            "fused_canny": fused_canny,
            "fused_magnitude": fused_magnitude,
            "fused_laplacian": fused_laplacian,
            "maps_left": maps_left,
            "maps_right": maps_right
        }
