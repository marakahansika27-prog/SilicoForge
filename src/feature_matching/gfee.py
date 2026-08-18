import cv2
import numpy as np
from src.utils.logger import Profiler

class GeometricFeatureExtractionEngine:
    """
    Geometric Feature Extraction Engine (GFEE)
    Implements SIFT + BFMatcher with Lowe's Ratio Test.
    """
    def __init__(self, ratio_thresh: float = 0.75):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.ratio_thresh = ratio_thresh
        self.stats = {}

    def run(self, inputs: dict) -> dict:
        """
        inputs: {'reference': ndarray, 'candidate': ndarray}
        returns: {'kp1': list, 'kp2': list, 'good_matches': list, 'stats': dict}
        """
        img1 = inputs['reference']
        img2 = inputs['candidate']
        
        with Profiler("GFEE_Run") as p:
            kp1, des1 = self.sift.detectAndCompute(img1, None)
            kp2, des2 = self.sift.detectAndCompute(img2, None)
            
            good_matches = []
            if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
                matches = self.bf.knnMatch(des1, des2, k=2)
                for m, n in matches:
                    if m.distance < self.ratio_thresh * n.distance:
                        good_matches.append(m)

        self.stats['runtime_ms'] = p.elapsed_ms
        self.stats['memory_kb'] = p.mem_diff_kb
        self.stats['num_kp1'] = len(kp1)
        self.stats['num_kp2'] = len(kp2)
        self.stats['num_good_matches'] = len(good_matches)
        
        return {
            'kp1': kp1,
            'kp2': kp2,
            'good_matches': good_matches,
            'stats': self.stats
        }
