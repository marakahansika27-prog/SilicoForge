import numpy as np
import time
import os
import csv
from typing import List, Dict

class EvaluationEngine:
    """
    Computes Pixel Error, Top-K Accuracy, and logs results to CSV.
    """
    def __init__(self, output_dir: str = "outputs/evaluation"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        
    def add_result(self, gt_x: float, gt_y: float, pred_x: float, pred_y: float, conf: float, time_ms: float):
        err_x = abs(gt_x - pred_x)
        err_y = abs(gt_y - pred_y)
        pixel_error = np.sqrt(err_x**2 + err_y**2)
        
        self.results.append({
            'gt_x': gt_x, 'gt_y': gt_y,
            'pred_x': pred_x, 'pred_y': pred_y,
            'pixel_error': pixel_error,
            'confidence': conf,
            'runtime_ms': time_ms
        })

    def run(self) -> dict:
        """
        Computes final evaluation metrics based on added results.
        returns: dict of metrics
        """
        if not self.results:
            return {}
            
        errors = [r['pixel_error'] for r in self.results]
        runtimes = [r['runtime_ms'] for r in self.results]
        
        metrics = {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'max_error': float(np.max(errors)),
            'success_rate_2px': float(np.mean([1 if e < 2.0 else 0 for e in errors]) * 100),
            'avg_runtime_ms': float(np.mean(runtimes))
        }
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, "evaluation_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
            
        # Write summary report
        summary_path = os.path.join(self.output_dir, "summary_report.txt")
        with open(summary_path, 'w') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.3f}\n")
                
        return metrics
