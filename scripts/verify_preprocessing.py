import os
import sys

# Ensure correct import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import load_or_generate_dataset
from src.utils.visualizer import save_image, plot_histogram
from src.utils.report_generator import ReportGenerator
from src.preprocessing.ice import ImageConditioningEngine

def main():
    report = ReportGenerator("ICE", save_dir="outputs/reports")
    report.add_parameters({"blur_ksize": "(5, 5)", "clahe_clip": 2.0, "clahe_grid": "(8, 8)"})
    
    try:
        # Load inputs
        ref_img, search_img = load_or_generate_dataset()
        
        # Save raw inputs for debug
        save_image(ref_img, "outputs/debug/preprocessing", "raw_reference.png")
        save_image(search_img, "outputs/debug/preprocessing", "raw_search.png")
        plot_histogram(search_img, "outputs/debug/preprocessing", "hist_raw_search.png", "Raw Search Histogram")
        
        # Instantiate and run ICE
        engine = ImageConditioningEngine()
        result = engine.run({'reference': ref_img, 'search': search_img})
        
        ref_cond = result['reference_cond']
        search_cond = result['search_cond']
        stats = result['stats']
        
        # Save processed outputs
        save_image(ref_cond, "outputs/debug/preprocessing", "cond_reference.png")
        save_image(search_cond, "outputs/debug/preprocessing", "cond_search.png")
        plot_histogram(search_cond, "outputs/debug/preprocessing", "hist_cond_search.png", "Conditioned Search Histogram")
        
        report.add_execution_stats(stats['runtime_ms'], stats['memory_kb'], stats['search_dims'])
        report.add_images([
            "../debug/preprocessing/raw_search.png",
            "../debug/preprocessing/cond_search.png",
            "../debug/preprocessing/hist_raw_search.png",
            "../debug/preprocessing/hist_cond_search.png"
        ])
        report.add_status(True, "ICE executed successfully. Image contrast improved via CLAHE and noise reduced.")
        
    except Exception as e:
        report.add_status(False, f"Exception occurred: {str(e)}")
        raise
    finally:
        report.save()

if __name__ == "__main__":
    main()
