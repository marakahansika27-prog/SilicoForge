import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import load_or_generate_dataset
from src.integration.pipeline import HybridNavigationPipeline
from src.integration.visualization import PipelineVisualizer
from src.integration.performance import PerformanceEvaluationEngine

def main():
    print("========================================")
    print("DRIFT-SENSE V2 PIPELINE")
    print("========================================")
    
    # 1. Load Dataset
    ref_img, search_img = load_or_generate_dataset()
    print("Reference Loaded")
    
    # Use the mathematically correct ground truth corresponding to data_loader.py offset (330+150=480, 345+150=495)
    import numpy as np
    ground_truth = np.array([480.0, 495.0]) # True geometric center
    
    # 2. Run Pipeline
    pipeline = HybridNavigationPipeline()
    state = pipeline.run(ref_img, search_img)
    
    # Check execution
    modules = ["ICE", "GSPE", "GFEE", "SRAE", "Localization", "AI Refinement", "Decision Fusion", "Evaluation"]
    overall_pass = True
    
    # FIX: Enhanced debug trace for integration errors
    if 'error' in state:
        print(f"\n[CRITICAL FAILURE] Pipeline crashed with error:\n{state['error']}\n")
        overall_pass = False
    else:
        for mod in modules:
            status = "PASS" if state['modules'].get(mod, False) else "FAIL"
            print(f"{mod:<15} {status}")
            if status == "FAIL":
                overall_pass = False
            
    print("========================================")
    
    if overall_pass:
        # 3. Evaluate Performance
        perf = PerformanceEvaluationEngine(success_threshold=10.0)
        state['ground_truth'] = ground_truth
        metrics = perf.run(state)
        
        print(f"Classical Error : {metrics['classical_error']:.4f} px")
        print(f"AI Error        : {metrics['ai_error']:.4f} px")
        print(f"Improvement %   : {metrics['improvement_pct']:.2f}%")
        print(f"Confidence      : {metrics['confidence']:.4f}")
        print(f"Runtime         : {metrics['runtime']:.4f} s")
        print(f"Final Decision  : {metrics['decision']}")
        print("========================================")
        print("PASS" if metrics['success'] else "FAIL")
        print("========================================")
        
        # 4. Save Visualizations
        visualizer = PipelineVisualizer("outputs/pipeline")
        visualizer.run(state)
        
        # 5. Generate PIPELINE_REPORT.md
        os.makedirs("outputs/reports", exist_ok=True)
        with open("outputs/reports/PIPELINE_REPORT.md", "w") as f:
            f.write("# Drift-Sense V2 Pipeline Report\n\n")
            f.write("## Purpose\nEnd-to-end integration and performance tracking.\n\n")
            f.write("## Inputs\n- Reference Image\n- Search Image\n\n")
            f.write("## Intermediate Outputs Saved\n")
            f.write("- reference.png\n- candidate.png\n- coarse_match.png\n- feature_matches.png\n")
            f.write("- registered.png\n- attention_map.png\n- heatmap.png\n- difference_map.png\n- final_prediction.png\n\n")
            f.write("## Coordinates\n")
            f.write(f"- Classical: {state['classical_coord']}\n")
            f.write(f"- AI: {state['ai_coord']}\n")
            f.write(f"- Final: {state['final_coord']}\n")
            f.write(f"- Ground Truth: {ground_truth}\n\n")
            f.write("## Performance\n")
            f.write(f"- Classical Error: {metrics['classical_error']:.4f}\n")
            f.write(f"- AI Error: {metrics['ai_error']:.4f}\n")
            f.write(f"- Improvement: {metrics['improvement_pct']:.2f}%\n")
            f.write(f"- Runtime: {metrics['runtime']:.4f}s\n")
            f.write(f"- Confidence: {metrics['confidence']:.4f}\n")
            f.write(f"- Decision Taken: {metrics['decision']}\n\n")
            f.write(f"## Status\n**{'PASS' if metrics['success'] else 'FAIL'}**\n")
    else:
        print("Pipeline execution failed.")
        print("========================================")

if __name__ == "__main__":
    main()
