import os
import sys
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import load_or_generate_dataset
from src.utils.visualizer import save_image
from src.utils.report_generator import ReportGenerator
from src.feature_matching.gfee import GeometricFeatureExtractionEngine

def main():
    report = ReportGenerator("GFEE", save_dir="outputs/reports")
    report.add_parameters({"extractor": "SIFT", "ratio_thresh": 0.75})
    
    try:
        ref_img, search_img = load_or_generate_dataset()
        
        # Crop a candidate to simulate GSPE output
        h, w = ref_img.shape
        cand_img = search_img[300:300+h, 300:300+w].copy() # Roughly near true location
        
        engine = GeometricFeatureExtractionEngine(ratio_thresh=0.75)
        result = engine.run({'reference': ref_img, 'candidate': cand_img})
        
        kp1, kp2, matches, stats = result['kp1'], result['kp2'], result['good_matches'], result['stats']
        
        # Visualize matches
        match_img = cv2.drawMatches(ref_img, kp1, cand_img, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        save_image(match_img, "outputs/debug/matches", "sift_matches.png")
        
        report.add_execution_stats(stats['runtime_ms'], stats['memory_kb'], "N/A")
        report.add_images(["../debug/matches/sift_matches.png"])
        
        obs = f"Found {stats['num_kp1']} ref keypoints, {stats['num_kp2']} cand keypoints, {stats['num_good_matches']} good matches."
        report.add_status(True, obs)
        
    except Exception as e:
        report.add_status(False, f"Exception occurred: {str(e)}")
        raise
    finally:
        report.save()

if __name__ == "__main__":
    main()
