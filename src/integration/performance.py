import numpy as np

class PerformanceEvaluationEngine:
    """
    Computes performance metrics for the complete Drift-Sense V2 pipeline.
    Compares Classical vs AI vs Final coordinates against Ground Truth.
    """
    def __init__(self, success_threshold=10.0):
        self.success_threshold = success_threshold
        
    def run(self, inputs):
        gt = np.array(inputs['ground_truth'], dtype=np.float32)
        classical = np.array(inputs['classical_coord'], dtype=np.float32)
        ai = np.array(inputs.get('ai_coord', classical), dtype=np.float32)
        final = np.array(inputs['final_coord'], dtype=np.float32)
        
        classical_err = float(np.linalg.norm(gt - classical))
        ai_err = float(np.linalg.norm(gt - ai))
        final_err = float(np.linalg.norm(gt - final))
        
        # Improvement relative to classical baseline
        improvement_pct = ((classical_err - final_err) / (classical_err + 1e-8)) * 100.0
        
        success = final_err <= self.success_threshold
        
        return {
            'classical_error': classical_err,
            'ai_error': ai_err,
            'final_error': final_err,
            'improvement_pct': improvement_pct,
            'runtime': inputs.get('runtime', 0.0),
            'confidence': inputs.get('confidence', 0.0),
            'decision': inputs.get('decision', 'UNKNOWN'),
            'success': success
        }
