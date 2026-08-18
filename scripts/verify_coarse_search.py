import os
import sys
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import load_or_generate_dataset
from src.utils.visualizer import save_image
from src.utils.report_generator import ReportGenerator
from src.coarse_search.gspe import GlobalSearchProposalEngine

def main():
    report = ReportGenerator("GSPE", save_dir="outputs/reports")
    report.add_parameters({"top_k": 5, "method": "NCC"})
    
    try:
        ref_img, search_img = load_or_generate_dataset()
        
        engine = GlobalSearchProposalEngine(top_k=5)
        result = engine.run({'reference': ref_img, 'search': search_img})
        
        boxes = result['boxes']
        heatmap = result['heatmap']
        stats = result['stats']
        
        # Save visualizations
        save_image(heatmap, "outputs/debug/candidates", "ncc_heatmap.png")
        
        # Draw boxes on search image
        search_color = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(boxes):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.rectangle(search_color, (x, y), (x+w, y+h), color, 2)
            cv2.putText(search_color, f"Top-{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        save_image(search_color, "outputs/debug/candidates", "top_k_candidates.png")
        
        report.add_execution_stats(stats['runtime_ms'], stats['memory_kb'], stats['heatmap_dims'])
        report.add_images([
            "../debug/candidates/ncc_heatmap.png",
            "../debug/candidates/top_k_candidates.png"
        ])
        
        obs = f"Generated {len(boxes)} candidate bounding boxes. Top-1 score: {result['scores'][0]:.3f}"
        report.add_status(True, obs)
        
    except Exception as e:
        report.add_status(False, f"Exception occurred: {str(e)}")
        raise
    finally:
        report.save()

if __name__ == "__main__":
    main()
