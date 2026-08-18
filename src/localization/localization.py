import cv2
import numpy as np
from src.utils.logger import Profiler

class ClassicalLocalization:
    """
    Classical Localization Engine
    Computes the final navigation coordinate using the Affine Transform matrix.
    """
    def __init__(self):
        self.stats = {}

    def run(self, inputs: dict) -> dict:
        """
        inputs: {'affine_matrix': ndarray, 'candidate_x': int, 'candidate_y': int, 'inliers': int}
        returns: {'dx': float, 'dy': float, 'confidence': float, 'stats': dict}
        """
        matrix = inputs['affine_matrix']
        cx = inputs['candidate_x']
        cy = inputs['candidate_y']
        inliers = inputs['inliers']
        
        with Profiler("Localization_Run") as p:
            if matrix is not None:
                # Affine matrix is [ [a, b, tx], [c, d, ty] ]
                # The translation component (residual drift)
                tx, ty = matrix[0, 2], matrix[1, 2]
                
                # Total drift = coarse drift + fine residual drift
                dx = float(cx + tx)
                dy = float(cy + ty)
                
                # Simple classical confidence: min(1.0, inliers / 50)
                confidence = float(np.clip(inliers / 50.0, 0, 1.0))
            else:
                dx, dy, confidence = 0.0, 0.0, 0.0

        self.stats['runtime_ms'] = p.elapsed_ms
        self.stats['memory_kb'] = p.mem_diff_kb
        
        return {
            'dx': dx,
            'dy': dy,
            'confidence': confidence,
            'stats': self.stats
        }
