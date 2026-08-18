import torch
import numpy as np

class SNRNInference:
    """Inference Engine for SNRN. Evaluates classical vs AI errors."""
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def run(self, batch):
        with torch.no_grad():
            preds = self.model(batch['reference_patch'], batch['candidate_patch'])
            
            gt_coord = batch['ground_truth_coordinate'].numpy()
            class_coord = batch['classical_coordinate'].numpy()
            pred_delta = preds['residual'].numpy()
            
            ai_coord = class_coord + pred_delta
            
            # Calculate errors
            classical_err = np.linalg.norm(gt_coord - class_coord, axis=1)
            ai_err = np.linalg.norm(gt_coord - ai_coord, axis=1)
            
            improvement_pct = ((classical_err - ai_err) / (classical_err + 1e-8)) * 100
            
            return {
                'classical_error': classical_err,
                'ai_error': ai_err,
                'improvement_pct': improvement_pct,
                'predicted_delta': pred_delta,
                'confidence': preds['confidence'].numpy()
            }
