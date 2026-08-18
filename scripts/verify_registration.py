import os
import sys
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import load_or_generate_dataset
from src.utils.visualizer import save_overlay
from src.utils.report_generator import ReportGenerator
from src.feature_matching.gfee import GeometricFeatureExtractionEngine
from src.registration.srae import SpatialRegistrationAlignmentEngine

def main():
    report = ReportGenerator("SRAE", save_dir="outputs/reports")
    report.add_parameters({"ransac_thresh": 5.0})
    
    try:
        ref_img, search_img = load_or_generate_dataset()
        
        # Simulate candidate crop with artificial shift
        h, w = ref_img.shape
        cand_img = search_img[300+10:300+h+10, 300+15:300+w+15].copy() 
        
        # Extract features
        gfee = GeometricFeatureExtractionEngine()
        res_gfee = gfee.run({'reference': ref_img, 'candidate': cand_img})
        
        # Registration
        engine = SpatialRegistrationAlignmentEngine(ransac_thresh=5.0)
        result = engine.run({
            'reference': ref_img, 
            'candidate': cand_img, 
            'kp1': res_gfee['kp1'], 
            'kp2': res_gfee['kp2'], 
            'matches': res_gfee['good_matches']
        })
        
        aligned_img = result['aligned_candidate']
        stats = result['stats']
        
        # Visualizations
        save_overlay(ref_img, cand_img, "outputs/debug/registration", "before_alignment_diff.png")
        save_overlay(ref_img, aligned_img, "outputs/debug/registration", "after_alignment_diff.png")
        
        report.add_execution_stats(stats['runtime_ms'], stats['memory_kb'], str(aligned_img.shape))
        report.add_images([
            "../debug/registration/before_alignment_diff.png",
            "../debug/registration/after_alignment_diff.png"
        ])
        
        obs = f"Inliers: {stats['inliers']} / {stats['total_matches']} matches.\nTransformation: {result['affine_matrix']}"
        report.add_status(True, obs)
        
    except Exception as e:
        report.add_status(False, f"Exception occurred: {str(e)}")
        raise
    finally:
        report.save()

if __name__ == "__main__":
    main()
