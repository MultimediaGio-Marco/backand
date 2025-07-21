import cv2
import numpy as np
from pathlib import Path
import concurrent.futures
from ProccessingDeapMap import DeapMapProcessor
from StereoEdgeProcessor import StereoEdgeProcessor

def extract_boxes_from_mask(mask, min_area=300):
    """Estrae bounding box da una maschera binaria."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]
    return boxes

def process_all(left_path: Path, right_path: Path):
    """
    Esegue in parallelo il calcolo della mappa di profondità e dei bordi.
    Ritorna: depth_map, depth_color, depth_mask, edge_mask (binaria), boxes
    """
    depth_processor = DeapMapProcessor(left_path, right_path)
    edge_processor = StereoEdgeProcessor(left_path, right_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_depth = executor.submit(depth_processor.process)
        future_edge = executor.submit(edge_processor.process)

        depth_map, depth_color, depth_mask = future_depth.result()
        edges_result = future_edge.result()

    edge_mask = edges_result["final"]
    _, edge_mask_bin = cv2.threshold(edge_mask, 50, 255, cv2.THRESH_BINARY)
    #edge_mask_neg= 255 - edge_mask_bin
    combined_mask = cv2.bitwise_and(depth_mask,edge_mask)
    boxes = extract_boxes_from_mask(combined_mask)

    return depth_map, depth_color, depth_mask, edge_mask_bin, combined_mask, boxes,edge_mask

def main():
    nomeImage = "-La6idHh58M0HdCICX3J_left"
    nomeImage = nomeImage.replace("_left", "")
    left_path = f"/home/giovanni/Uni/multimedia/progetto/Holopix50k/val/left/{nomeImage}_left.jpg"
    right_path = f"/home/giovanni/Uni/multimedia/progetto/Holopix50k/val/right/{nomeImage}_right.jpg"

    # Solo visualizzazione, assume che process_all abbia fatto il calcolo
    depth_map, depth_color, depth_mask, edge_mask_bin, combined_mask, boxes,edge_mask = process_all(left_path, right_path)

    left_img = cv2.imread(str(left_path))
    out_img = left_img.copy()

    for (x, y, w, h) in boxes:
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow("edge_mask", edge_mask)
    cv2.imshow("Depth Mask", depth_mask)
    cv2.imshow("Edge Mask Binary", edge_mask_bin)
    cv2.imshow("Combined Mask", combined_mask)
    cv2.imshow("Detected Boxes", out_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
