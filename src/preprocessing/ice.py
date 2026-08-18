import cv2
import numpy as np
from src.utils.logger import Profiler

class ImageConditioningEngine:
    """
    Image Conditioning Engine (ICE)
    Implements Gaussian Blur, CLAHE, and Intensity Normalization.
    """
    def __init__(self, blur_ksize: tuple = (5, 5), clahe_clip: float = 2.0, clahe_grid: tuple = (8, 8)):
        self.blur_ksize = blur_ksize
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        self.stats = {}

    def _process_single(self, img: np.ndarray, name: str) -> np.ndarray:
        # Gaussian Blur
        blurred = cv2.GaussianBlur(img, self.blur_ksize, 0)
        # CLAHE
        clahe_img = self.clahe.apply(blurred)
        # Intensity Normalization (Min-Max to 0-255)
        normalized = cv2.normalize(clahe_img, None, 0, 255, cv2.NORM_MINMAX)
        
        self.stats[f'{name}_dims'] = str(normalized.shape)
        return normalized

    def run(self, inputs: dict) -> dict:
        """
        inputs: {'reference': ndarray, 'search': ndarray}
        returns: {'reference_cond': ndarray, 'search_cond': ndarray, 'stats': dict}
        """
        with Profiler("ICE_Run") as p:
            ref_cond = self._process_single(inputs['reference'], 'reference')
            search_cond = self._process_single(inputs['search'], 'search')
            
        self.stats['runtime_ms'] = p.elapsed_ms
        self.stats['memory_kb'] = p.mem_diff_kb
        
        return {
            'reference_cond': ref_cond,
            'search_cond': search_cond,
            'stats': self.stats
        }
