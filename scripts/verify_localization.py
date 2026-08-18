import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.report_generator import ReportGenerator
from src.localization.localization import ClassicalLocalization
import numpy as np

def main():
    report = ReportGenerator("LOCALIZATION", save_dir="outputs/reports")
    report.add_parameters({"method": "Classical Affine Residual"})
    
    try:
        # Mock inputs for localization
        # Simulated affine matrix representing a shift of tx=2.5, ty=-1.2
        affine_matrix = np.array([[1.0, 0.0, 2.5], [0.0, 1.0, -1.2]])
        candidate_x = 300
        candidate_y = 300
        inliers = 45
        
        engine = ClassicalLocalization()
        result = engine.run({
            'affine_matrix': affine_matrix, 
            'candidate_x': candidate_x, 
            'candidate_y': candidate_y, 
            'inliers': inliers
        })
        
        dx, dy, conf, stats = result['dx'], result['dy'], result['confidence'], result['stats']
        
        report.add_execution_stats(stats['runtime_ms'], stats['memory_kb'], "N/A")
        
        obs = f"Recovered Coordinate: X={dx:.2f}, Y={dy:.2f}\nConfidence: {conf:.2f}"
        report.add_status(True, obs)
        
    except Exception as e:
        report.add_status(False, f"Exception occurred: {str(e)}")
        raise
    finally:
        report.save()

if __name__ == "__main__":
    main()
