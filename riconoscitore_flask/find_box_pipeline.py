import cv2
from edge_pipeline import stereo_edge_enhanced
import os
from box_utils import *
from deapMap import relative_depth_map

print("Working directory:", os.getcwd())
nomeImage = "-LahPsJhCZTWwgvaAMB4_left"
#nomeImage = nomeImage.replace("_left", "")
#left_path = f"./Holopix50k/val/left/{nomeImage}_left.jpg"
#right_path = f"./Holopix50k/val/right/{nomeImage}_right.jpg"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pipeline_image_detector(left_path, right_path):
    # Carica immagini stereo
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    # Calcola edge fusi da stereo
    results = stereo_edge_enhanced(left, right)
    edge_map = results["final"]

    # Estrai bounding boxes dagli edge
    bboxes = extract_bounding_boxes_from_edges(edge_map, min_area=100, threshold=50)

    # Clusterizzazione con DBSCAN per unire box vicine
    merged_bboxes, labels = cluster_boxes_dbscan(bboxes, eps=55, min_samples=1)

    # Rimozione box troppo grandi e contenute
    img_area = left.shape[0] * left.shape[1]
    notobig = remove_bigest_boxes(merged_bboxes, img_area, 0.9)
    noinside = remove_contained_boxes(notobig)

    # Calcolo mappa di profondità
    depth_map = relative_depth_map(left_path, right_path)

    # Generazione maschera dagli edge per la selezione
    edge_mask = (edge_map > 50).astype('uint8') * 255

    # Selezione della migliore box sulla base di profondità e edge
    best_box = pick_best_box(noinside, edge_mask, depth_map)

    # Ritorna la box se trovata
    return best_box if best_box else None

