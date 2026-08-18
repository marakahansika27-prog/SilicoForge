import cv2
import numpy as np
from src.utils.logger import Profiler

class SpatialRegistrationAlignmentEngine:
    """
    Spatial Registration & Alignment Engine (SRAE)
    Implements RANSAC affine transformation estimation and warping.
    """
    def __init__(self, ransac_thresh: float = 5.0):
        self.ransac_thresh = ransac_thresh
        self.stats = {}

    def run(self, inputs: dict) -> dict:
        """
        inputs: {'reference': ndarray, 'candidate': ndarray, 'kp1': list, 'kp2': list, 'matches': list}
        returns: {'affine_matrix': ndarray, 'aligned_candidate': ndarray, 'inlier_mask': ndarray, 'stats': dict}
        """
        ref_img = inputs['reference']
        cand_img = inputs['candidate']
        kp1 = inputs['kp1']
        kp2 = inputs['kp2']
        matches = inputs['matches']
        
        with Profiler("SRAE_Run") as p:
            affine_matrix = None
            aligned_candidate = cand_img.copy()
            inlier_mask = None
            inlier_count = 0
            
            if len(matches) >= 3: # Need at least 3 points for Affine
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # We map candidate to reference (dst to src) to warp candidate
                matrix, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh)
                
                if matrix is not None:
                    affine_matrix = matrix
                    inlier_mask = inliers.ravel().tolist()
                    inlier_count = np.sum(inlier_mask)
                    h, w = ref_img.shape
                    aligned_candidate = cv2.warpAffine(cand_img, affine_matrix, (w, h))

        self.stats['runtime_ms'] = p.elapsed_ms
        self.stats['memory_kb'] = p.mem_diff_kb
        self.stats['inliers'] = int(inlier_count)
        self.stats['total_matches'] = len(matches)
        
        return {
            'affine_matrix': affine_matrix,
            'aligned_candidate': aligned_candidate,
            'inlier_mask': inlier_mask,
            'stats': self.stats
        }
