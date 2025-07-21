import cv2
import numpy as np
from pathlib import Path
import concurrent.futures
import os
from ProccessingDeapMap import DeapMapProcessor
from StereoEdgeProcessor import StereoEdgeProcessor
from box_utils import (
    extract_bounding_boxes_from_edges,
    cluster_boxes_dbscan,
    remove_bigest_boxes,
    remove_contained_boxes,
    pick_best_box
)

class StereoObjectDetector:
    def __init__(self, left_path: Path, right_path: Path, output_dir: str = "output"):
        self.left_path = Path(left_path)
        self.right_path = Path(right_path)
        self.left_img = cv2.imread(str(self.left_path))
        self.right_img = cv2.imread(str(self.right_path))
        
        # Crea directory di output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Estrae il nome base dell'immagine per i file di output
        self.base_name = self.left_path.stem.replace("_left", "")

        self.depth_processor = DeapMapProcessor(self.left_path, self.right_path)
        self.edge_processor = StereoEdgeProcessor(self.left_path, self.right_path)

    def _save_image(self, image, suffix, colormap=None):
        """Salva un'immagine con il nome base + suffisso"""
        filename = f"{self.base_name}_{suffix}.jpg"
        filepath = self.output_dir / filename
        
        if colormap is not None:
            # Applica colormap se specificato (per depth map)
            image_colored = cv2.applyColorMap(image, colormap)
            cv2.imwrite(str(filepath), image_colored)
        else:
            cv2.imwrite(str(filepath), image)
        
        print(f"Salvata: {filepath}")

    def _save_original_images(self):
        """Salva le immagini originali left e right"""
        self._save_image(self.left_img, "original_left")
        self._save_image(self.right_img, "original_right")

    def _extract_boxes_from_mask(self, mask, min_area=300):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]
        return boxes

    def process_all(self):
        # Salva le immagini originali
        self._save_original_images()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_depth = executor.submit(self.depth_processor.process)
            future_edge = executor.submit(self.edge_processor.process)

            depth_map, depth_color, depth_mask = future_depth.result()
            edges_result = future_edge.result()

        # Salva i risultati della depth map
        self._save_image(depth_map, "depth_map_raw")
        self._save_image(depth_color, "depth_color")
        self._save_image(depth_mask, "depth_mask")

        # Salva i risultati degli edge (inclusi quelli secondari)
        edge_mask = edges_result["final"]
        self._save_image(edge_mask, "edge_final")
        self._save_image(edges_result["fused_canny"], "fused_canny")
        self._save_image(edges_result["fused_magnitude"], "fused_magnitude")
        self._save_image(edges_result["fused_laplacian"], "fused_laplacian")

        edge_mask_bin = None
        if edge_mask is not None:
            _, edge_mask_bin = cv2.threshold(edge_mask, 50, 255, cv2.THRESH_BINARY)
            self._save_image(edge_mask_bin, "edge_mask_binary")

        # Combina le maschere e salva
        combined_mask = cv2.bitwise_and(depth_mask, edge_mask)
        self._save_image(combined_mask, "combined_mask")
        
        boxes = self._extract_boxes_from_mask(combined_mask)

        return {
            "depth_map": depth_map,
            "depth_color": depth_color,
            "depth_mask": depth_mask,
            "edge_mask": edge_mask,
            "edge_mask_bin": edge_mask_bin,
            "combined_mask": combined_mask,
            "raw_boxes": boxes,
            "edges_result": edges_result
        }

    def refine_boxes(self, edge_map, boxes):
        merged_bboxes, _ = cluster_boxes_dbscan(boxes, eps=55, min_samples=1)
        img_area = self.left_img.shape[0] * self.left_img.shape[1]
        notobig = remove_bigest_boxes(merged_bboxes, img_area, 0.9)
        noinside = remove_contained_boxes(notobig)

        depth_map = self.depth_processor._relative_depth_map()
        edge_mask = (edge_map > 50).astype('uint8') * 255

        best_box = pick_best_box(noinside, edge_mask, depth_map)
        return best_box, noinside

    def _save_final_results(self, result):
        """Salva le immagini finali con le bounding boxes"""
        out_img = self.left_img.copy()
        
        # Disegna tutti i box raffinati in verde
        for (x, y, w, h) in result["refined_boxes"]:
            cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Disegna il miglior box in rosso
        if result["best_box"]:
            x, y, w, h = result["best_box"]
            cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

        self._save_image(out_img, "final_detection")

        # Salva anche un'immagine solo con il best box
        if result["best_box"]:
            best_box_img = self.left_img.copy()
            x, y, w, h = result["best_box"]
            cv2.rectangle(best_box_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            self._save_image(best_box_img, "best_box_only")

    def run(self):
        result = self.process_all()
        best_box, refined_boxes = self.refine_boxes(result["edge_mask"], result["raw_boxes"])
        result.update({
            "refined_boxes": refined_boxes,
            "best_box": best_box
        })
        
        # Salva i risultati finali
        self._save_final_results(result)
        
        return result


def main():
    nome_image = "-L__uMAz3k25WltzLheY_left"
    nome_image = nome_image.replace("_left", "")
    left_path = f"/home/giovanni/Uni/multimedia/progetto/Holopix50k/val/left/{nome_image}_left.jpg"
    right_path = f"/home/giovanni/Uni/multimedia/progetto/Holopix50k/val/right/{nome_image}_right.jpg"

    # Specifica la directory di output
    output_directory = "detection_results"
    detector = StereoObjectDetector(left_path, right_path, output_directory)
    result = detector.run()

    print(f"\nElaborazione completata!")
    print(f"Immagini salvate in: {detector.output_dir}")
    print(f"Numero di box raffinati: {len(result['refined_boxes'])}")
    print(f"Best box trovato: {'Sì' if result['best_box'] else 'No'}")

    # Opzionale: mostra ancora le immagini principali
    cv2.imshow("Depth Mask", result["depth_mask"])
    cv2.imshow("Edge Mask", result["edge_mask"])
    cv2.imshow("Combined Mask", result["combined_mask"])
    
    # Immagine finale
    out_img = cv2.imread(str(left_path)).copy()
    for (x, y, w, h) in result["refined_boxes"]:
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if result["best_box"]:
        x, y, w, h = result["best_box"]
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("Detected Boxes", out_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()