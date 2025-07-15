import cv2
from edge_pipeline import stereo_edge_enhanced,postprocess_edges
import os
from box_utils import *
from deapMap import display_relative_depth_map, relative_depth_map
import matplotlib.pyplot as plt

print("Working directory:", os.getcwd())
nomeImage = "-LahPsJhCZTWwgvaAMB4_left"
nomeImage = nomeImage.replace("_left", "")
left_path = f"./Holopix50k/val/left/{nomeImage}_left.jpg"
right_path = f"./Holopix50k/val/right/{nomeImage}_right.jpg"

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



def pipeline_image_detector_stamp(left_path, right_path):
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    results = stereo_edge_enhanced(left, right)

    cv2.imshow("Final Combined", results["final"])
    cv2.imshow("Fused Canny", results["fused_canny"])
    cv2.imshow("Fused Magnitude", results["fused_magnitude"])
    cv2.imshow("Fused Laplacian", results["fused_laplacian"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Estrai bounding box
    edge_map = results["final"]

    bboxes = extract_bounding_boxes_from_edges(edge_map, min_area=100, threshold=50)

    # Disegna le box originali (verde)
    output_img = left.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Bounding Boxes Originali (Verdi)", output_img)
    cv2.waitKey(0)

    # Clusterizza le bounding box con DBSCAN
    merged_bboxes, labels = cluster_boxes_dbscan(bboxes, eps=55, min_samples=1)
    #merged_bboxes, labels = cluster_boxes_agglomerative(bboxes, max_distance=60)
    #merged_bboxes, labels = cluster_boxes_dbscan_with_iou(bboxes, eps=1000, min_samples=1, iou_threshold=0.3)
    #merged_bboxes=merge_contact_boxes(merged_bboxes,0)
    #merged_bboxes = merge_boxes_connected_components(bboxes, left.shape)
    output_img = left.copy()
    for (x, y, w, h) in merged_bboxes:
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    cv2.imshow("Bounding Boxes Clusterizzate (Rosse)", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_area = left.shape[0] * left.shape[1]
    notobig=remove_bigest_boxes(merged_bboxes, img_area, 0.9)
    noinside = remove_contained_boxes(notobig)
    output_img = left.copy()
    for (x, y, w, h) in noinside:
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    cv2.imshow("filtered Boxes (blue)", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    depth_map=relative_depth_map(left_path, right_path)
    display_relative_depth_map(depth_map)
    edge_mask = (edge_map > 50).astype('uint8') * 255  # stesso threshold usato prima
    best_box = pick_best_box(noinside, edge_mask, depth_map)

    # Disegna best box in giallo
    if best_box:
        x, y, w, h = best_box
        output_img = left.copy()
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow("Best Box (Gialla)", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nessuna box selezionata come migliore.")
        
def pipeline_image_detector_save(left_path, right_path):
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    # Creazione cartelle output
    output_base = "./backup/test 6/output"
    dirs = {
        "original": os.path.join(output_base, "original"),
        "edges": os.path.join(output_base, "edges"),
        "bboxes_original": os.path.join(output_base, "bboxes_original"),
        "bboxes_clustered": os.path.join(output_base, "bboxes_clustered"),
        "bboxes_filtered": os.path.join(output_base, "bboxes_filtered"),
        "best_box": os.path.join(output_base, "best_box"),
    }
    for d in dirs.values():
        ensure_dir(d)

    # Salvo immagini originali
    cv2.imwrite(os.path.join(dirs["original"], f"{nomeImage}_left.jpg"), left)
    cv2.imwrite(os.path.join(dirs["original"], f"{nomeImage}_right.jpg"), right)

    results = stereo_edge_enhanced(left, right)

    # Salvo immagini edge
    cv2.imwrite(os.path.join(dirs["edges"], f"{nomeImage}_final.jpg"), results["final"])
    cv2.imwrite(os.path.join(dirs["edges"], f"{nomeImage}_fused_canny.jpg"), results["fused_canny"])
    cv2.imwrite(os.path.join(dirs["edges"], f"{nomeImage}_fused_magnitude.jpg"), results["fused_magnitude"])
    cv2.imwrite(os.path.join(dirs["edges"], f"{nomeImage}_fused_laplacian.jpg"), results["fused_laplacian"])

    cv2.imshow("Final Combined", results["final"])
    cv2.imshow("Fused Canny", results["fused_canny"])
    cv2.imshow("Fused Magnitude", results["fused_magnitude"])
    cv2.imshow("Fused Laplacian", results["fused_laplacian"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Estrai bounding box
    edge_map = results["final"]

    bboxes = extract_bounding_boxes_from_edges(edge_map, min_area=100, threshold=50)

    # Disegna le box originali (verde)
    output_img = left.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(dirs["bboxes_original"], f"{nomeImage}_bboxes_original.jpg"), output_img)
    cv2.imshow("Bounding Boxes Originali (Verdi)", output_img)
    cv2.waitKey(0)

    # Clusterizza le bounding box con DBSCAN
    merged_bboxes, labels = cluster_boxes_dbscan(bboxes, eps=60, min_samples=1)
    output_img = left.copy()
    for (x, y, w, h) in merged_bboxes:
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(dirs["bboxes_clustered"], f"{nomeImage}_bboxes_clustered.jpg"), output_img)
    cv2.imshow("Bounding Boxes Clusterizzate (Rosse)", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_area = left.shape[0] * left.shape[1]
    notobig = remove_bigest_boxes(merged_bboxes, img_area, 0.9)
    noinside = remove_contained_boxes(notobig)
    output_img = left.copy()
    for (x, y, w, h) in noinside:
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imwrite(os.path.join(dirs["bboxes_filtered"], f"{nomeImage}_bboxes_filtered.jpg"), output_img)
    cv2.imshow("filtered Boxes (blue)", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    depth_map = relative_depth_map(left_path, right_path)
    # Salvo la mappa di profondità in una cartella a parte
    depth_dir = os.path.join(output_base, "depth_maps")
    ensure_dir(depth_dir)
    display_relative_depth_map(depth_map)
    depth_path = os.path.join(depth_dir, f"{nomeImage}_depth_map.png")
    # Salva con colormap inferno usando matplotlib
    plt.imsave(depth_path, depth_map, cmap='inferno')

    edge_mask = (edge_map > 50).astype('uint8') * 255  # stesso threshold usato prima
    best_box = pick_best_box(noinside, edge_mask, depth_map)

    # Disegna best box in giallo
    if best_box:
        x, y, w, h = best_box
        output_img = left.copy()
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imwrite(os.path.join(dirs["best_box"], f"{nomeImage}_best_box.jpg"), output_img)
        cv2.imshow("Best Box (Gialla)", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nessuna box selezionata come migliore.")
        
pipeline_image_detector_stamp(left_path, right_path)
