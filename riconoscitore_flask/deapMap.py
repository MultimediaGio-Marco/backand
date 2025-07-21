"""
Stereo-based relative depth map (pure OpenCV).
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Stereo matcher parameters
# ------------------------------------------------------------------
STEREO_PARAMS = dict(
    minDisparity=0,
    numDisparities=128,
    blockSize=5,
    P1=8 * 1 * 5 ** 2,
    P2=32 * 1 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
)


# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------
def relative_depth_map(left_path: str | Path, right_path: str | Path) -> np.ndarray:
    """
    Calcola una mappa di profondità relativa (valori alti = oggetti vicini),
    normalizzata tra 0 e 1.
    """
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

    stereo = cv2.StereoSGBM_create(**STEREO_PARAMS)
    disp = stereo.compute(left, right).astype(np.float32) / 16.0

    valid = disp > 0
    rel = np.zeros_like(disp)
    rel[valid] = np.log1p(disp[valid])

    if np.any(valid):
        v = rel[valid]
        rel[valid] = (v - v.min()) / (v.max() - v.min() + 1e-6)

    return rel

def colorize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """
    Converte una mappa di profondità normalizzata (float 0-1) in immagine colorata BGR uint8.
    
    Args:
        depth_map: np.ndarray, valori float tra 0 e 1
    
    Returns:
        img_color: np.ndarray, immagine BGR uint8 colorata con COLORMAP_INFERNO
    """
    # Scala [0..1] → [0..255] uint8
    depth_uint8 = (depth_map * 255).astype(np.uint8)

    # Applica colormap inferno di OpenCV
    img_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    return img_color


# ------------------------------------------------------------------
# Quick visualisation helper
# ------------------------------------------------------------------
def display_relative_depth_map(depth:np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.imshow(depth, cmap="inferno")
    plt.title("Relative depth map")
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.show()