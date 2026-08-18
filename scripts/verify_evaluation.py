import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.report_generator import ReportGenerator
from src.evaluation.evaluation import EvaluationEngine

def main():
    report = ReportGenerator("EVALUATION", save_dir="outputs/reports")
    report.add_parameters({"metrics": "Pixel Error, Mean, Median, Success Rate"})
    
    try:
        engine = EvaluationEngine()
        # Mock some results
        engine.add_result(gt_x=330.0, gt_y=345.0, pred_x=330.5, pred_y=344.8, conf=0.92, time_ms=120.0)
        engine.add_result(gt_x=400.0, gt_y=400.0, pred_x=401.2, pred_y=399.5, conf=0.88, time_ms=115.0)
        engine.add_result(gt_x=100.0, gt_y=100.0, pred_x=100.1, pred_y=100.0, conf=0.98, time_ms=130.0)
        
        metrics = engine.run()
        
        report.add_execution_stats(metrics['avg_runtime_ms'], 0, "N/A")
        
        obs = "\n".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
        report.add_status(True, f"Evaluation completed successfully.\n{obs}")
        
    except Exception as e:
        report.add_status(False, f"Exception occurred: {str(e)}")
        raise
    finally:
        report.save()

if __name__ == "__main__":
    main()
