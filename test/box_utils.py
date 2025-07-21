import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import numpy as np

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area

def cluster_boxes_dbscan_with_iou(boxes, eps=50, min_samples=1, iou_threshold=0.3):
    if not boxes:
        return [], []

    centroids = np.array([[x + w / 2, y + h / 2] for (x, y, w, h) in boxes])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
    labels = clustering.labels_

    merged_boxes = []

    for label in np.unique(labels):
        cluster_boxes = [boxes[i] for i in range(len(boxes)) if labels[i] == label]
        
        # Unisce solo le box con IoU ≥ soglia
        used = [False] * len(cluster_boxes)

        for i in range(len(cluster_boxes)):
            if used[i]:
                continue

            x1, y1, w1, h1 = cluster_boxes[i]
            group = [cluster_boxes[i]]
            used[i] = True

            for j in range(i + 1, len(cluster_boxes)):
                if not used[j]:
                    iou = compute_iou(cluster_boxes[i], cluster_boxes[j])
                    if iou >= iou_threshold:
                        group.append(cluster_boxes[j])
                        used[j] = True

            # Fonde il gruppo
            xs = [x for x, y, w, h in group]
            ys = [y for x, y, w, h in group]
            x2s = [x + w for x, y, w, h in group]
            y2s = [y + h for x, y, w, h in group]

            x_min = min(xs)
            y_min = min(ys)
            x_max = max(x2s)
            y_max = max(y2s)

            merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return merged_boxes, labels

def cluster_boxes_dbscan(boxes, eps=50, min_samples=1):
    """
    Raggruppa le bounding box vicine usando DBSCAN e le fonde.
    - boxes: lista di (x, y, w, h)
    - eps: distanza massima tra centroidi per considerarli vicini
    - min_samples: minimo numero di vicini per formare un cluster (usa 1)

    Ritorna:
    - merged_boxes: lista di (x, y, w, h) una per cluster
    - labels: etichetta assegnata ad ogni box originale
    """
    if not boxes:
        return [], []

    # Calcola i centroidi
    centroids = np.array([[x + w / 2, y + h / 2] for (x, y, w, h) in boxes])

    # Applica DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
    labels = clustering.labels_

    merged_boxes = []

    for label in np.unique(labels):
        cluster_boxes = [boxes[i] for i in range(len(boxes)) if labels[i] == label]
        xs = [x for x, y, w, h in cluster_boxes]
        ys = [y for x, y, w, h in cluster_boxes]
        x2s = [x + w for x, y, w, h in cluster_boxes]
        y2s = [y + h for x, y, w, h in cluster_boxes]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(x2s)
        y_max = max(y2s)

        merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return merged_boxes, labels

def extract_bounding_boxes_from_edges(edge_map, min_area=100, threshold=50):
    """
    Estrae bounding box da una mappa di edge binarizzata.

    Parametri:
    - edge_map: immagine di edge (uint8) su cui lavorare
    - min_area: area minima per filtrare rumore
    - threshold: valore di soglia per binarizzare la mappa

    Ritorna:
    - bboxes: lista di tuple (x, y, w, h)
    """
    # Binarizza la mappa
    _, binary = cv2.threshold(edge_map, threshold, 255, cv2.THRESH_BINARY)

    # Trova i contorni
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calcola bounding box con filtro sull'area
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area >= min_area:
            bboxes.append((x, y, w, h))

    return bboxes

def remove_contained_boxes(boxes):
    """
    Rimuove le bounding box che sono completamente contenute in un'altra box.

    Parametri:
    - boxes: lista di (x, y, w, h)

    Ritorna:
    - filtered_boxes: sottoinsieme di boxes senza quelle contenute
    """
    filtered = []

    for i, (x1, y1, w1, h1) in enumerate(boxes):
        x1_max, y1_max = x1 + w1, y1 + h1
        contained = False

        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i == j:
                continue
            x2_max, y2_max = x2 + w2, y2 + h2

            # Controlla se box i è contenuta nella box j
            if x1 >= x2 and y1 >= y2 and x1_max <= x2_max and y1_max <= y2_max:
                contained = True
                break

        if not contained:
            filtered.append((x1, y1, w1, h1))

    return filtered

def remove_bigest_boxes(boxes,img_area, max_area_ratio=0.8):
    """
    Rimuove le bounding box più grandi di una certa percentuale dell'area dell'immagine.

    Parametri:
    - boxes: lista di (x, y, w, h)
    - max_area_ratio: rapporto massimo dell'area della box rispetto all'immagine

    Ritorna:
    - filtered_boxes: sottoinsieme di boxes senza quelle troppo grandi
    """
    if not boxes:
        return []

    filtered = []
    for (x, y, w, h) in boxes:
        area = w * h
        if area / img_area < max_area_ratio:
            filtered.append((x, y, w, h))

    return filtered

def pick_best_box(boxes, depth, edge_mask):
    """
    Seleziona la box migliore in base a:
    - area (favorisce box grandi)
    - profondità media (oggetti più vicini)
    - densità di edge (oggetti ben definiti)
    - prossimità al centro

    Parametri:
    - boxes: lista di (x, y, w, h)
    - depth: mappa di profondità (array 2D float32)
    - edge_mask: mappa di edge binaria (uint8)

    Ritorna:
    - la box migliore (x, y, w, h), o None se lista vuota
    """
    if not boxes:
        return None

    H, W = edge_mask.shape
    center_x, center_y = W // 2, H // 2
    img_diag = np.sqrt(H ** 2 + W ** 2)

    best_score = -np.inf
    best_box = None

    for (x, y, w, h) in boxes:
        area = w * h
        cx, cy = x + w // 2, y + h // 2
        dist_center = np.linalg.norm([cx - center_x, cy - center_y])

        roi_edge = edge_mask[y:y + h, x:x + w]
        edge_density = np.sum(roi_edge > 0) / (area + 1e-6)

        roi_depth = depth[y:y + h, x:x + w]
        valid_depth = roi_depth[roi_depth > 0]
        mean_depth = np.mean(valid_depth) if valid_depth.size > 0 else 0

        score = (
            0.35 * (area / (H * W)) +
            0.35 * mean_depth +
            0.15 * edge_density +
            0.15 * (1 - dist_center / img_diag)
        )

        if score > best_score:
            best_score = score
            best_box = (x, y, w, h)

    return best_box

def merge_contact_boxes(boxes, tolerance=5):
    """
    Unisce bounding box che si sovrappongono o che sono molto vicine (a contatto).

    Parametri:
    - boxes: lista di (x, y, w, h)
    - tolerance: distanza massima per considerare due box adiacenti

    Ritorna:
    - merged_boxes: lista di box unificate
    """
    import itertools

    def boxes_touch(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        # Coordinate dei bordi
        left1, right1, top1, bottom1 = x1, x1 + w1, y1, y1 + h1
        left2, right2, top2, bottom2 = x2, x2 + w2, y2, y2 + h2

        # Estendi i bordi con la tolleranza
        left1 -= tolerance
        right1 += tolerance
        top1 -= tolerance
        bottom1 += tolerance

        # Controlla sovrapposizione con margine
        return not (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1)

    boxes = boxes[:]
    merged = []

    while boxes:
        base = boxes.pop(0)
        group = [base]

        i = 0
        while i < len(boxes):
            if any(boxes_touch(base, b) for b in group):
                group.append(boxes.pop(i))
                i = 0  # ricomincia con la nuova lista
            else:
                i += 1

        # Fonde tutte le box del gruppo
        xs = [x for x, y, w, h in group]
        ys = [y for x, y, w, h in group]
        x2s = [x + w for x, y, w, h in group]
        y2s = [y + h for x, y, w, h in group]

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(x2s), max(y2s)

        merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return merged

def merge_boxes_connected_components(boxes, image_shape):
    """
    Unisce le box usando componenti connesse in una maschera binaria.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    num_labels, labels = cv2.connectedComponents(mask)

    merged_boxes = []
    for label_id in range(1, num_labels):  # salta sfondo (0)
        component_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            merged_boxes.append((x, y, w, h))

    return merged_boxes

#ciao

def bbox_distance(b1, b2):
    """
    Calcola una distanza tra due bounding box.
    - Se si sovrappongono: distanza = 0
    - Altrimenti: distanza euclidea tra bordi più vicini
    """
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Check sovrapposizione
    overlap_x = max(0, min(x1_max, x2_max) - max(x1, x2))
    overlap_y = max(0, min(y1_max, y2_max) - max(y1, y2))
    if overlap_x > 0 and overlap_y > 0:
        return 0  # si sovrappongono

    # Distanza minima tra bordi
    dx = max(x1 - x2_max, x2 - x1_max, 0)
    dy = max(y1 - y2_max, y2 - y1_max, 0)
    return np.sqrt(dx**2 + dy**2)

def cluster_boxes_agglomerative(boxes, max_distance=30):
    """
    Raggruppa le box usando Agglomerative Clustering su distanze box-box.
    - max_distance: distanza massima per unire cluster
    """
    if not boxes:
        return [], []

    # Costruisci matrice di distanze pairwise
    D = squareform(pdist(np.array(boxes), metric=lambda u, v: bbox_distance(tuple(u), tuple(v))))

    # Clustering
    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed',
                                         linkage='complete', distance_threshold=max_distance)
    labels = clustering.fit_predict(D)

    # Merge per cluster
    merged_boxes = []
    for label in np.unique(labels):
        cluster = [boxes[i] for i in range(len(boxes)) if labels[i] == label]

        xs = [x for x, y, w, h in cluster]
        ys = [y for x, y, w, h in cluster]
        x2s = [x + w for x, y, w, h in cluster]
        y2s = [y + h for x, y, w, h in cluster]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(x2s)
        y_max = max(y2s)

        merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return merged_boxes, labels





