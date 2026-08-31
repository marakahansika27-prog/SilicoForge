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
        inputs: {'affine_matrix': ndarray, 'candidate_x': float, 'candidate_y': float, 'inliers': int}
        returns: {'dx': float, 'dy': float, 'confidence': float, 'stats': dict}
        """
        matrix = inputs['affine_matrix']
        cx = inputs['candidate_x']  # Top-left X of the candidate crop in search space
        cy = inputs['candidate_y']  # Top-left Y of the candidate crop in search space
        inliers = inputs['inliers']
        
        # We assume the target is at the center of the reference image
        ref_center_x = inputs.get('ref_center_x', 500.0)
        ref_center_y = inputs.get('ref_center_y', 500.0)
        
        with Profiler("Localization_Run") as p:
            if matrix is not None:
                # SRAE computes Candidate -> Reference matrix.
                # We need Reference -> Candidate.
                inv_matrix = cv2.invertAffineTransform(matrix)
                
                # Project the reference target center into the candidate crop space
                ref_pt = np.array([ref_center_x, ref_center_y, 1.0], dtype=np.float32)
                cand_local_x = inv_matrix[0, 0] * ref_pt[0] + inv_matrix[0, 1] * ref_pt[1] + inv_matrix[0, 2]
                cand_local_y = inv_matrix[1, 0] * ref_pt[0] + inv_matrix[1, 1] * ref_pt[1] + inv_matrix[1, 2]
                
                # The final coordinate in the search space is the crop's top-left + the local coordinate
                dx = float(cx + cand_local_x)
                dy = float(cy + cand_local_y)
                
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
