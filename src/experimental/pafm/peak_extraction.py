import cv2
import numpy as np

def extract_diagnostic_peaks(res_surface, num_peaks=20, radius=50):
    """
    Extracts the strongest spatially distinct local maxima from a response surface.
    Returns: list of (cx, cy, score)
    """
    peaks = []
    res = res_surface.copy()
    h, w = res.shape
    
    r = int(radius)
    for _ in range(num_peaks):
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < -1.0: # If all useful peaks are exhausted
            break
            
        cx, cy = float(max_loc[0]), float(max_loc[1])
        peaks.append((cx, cy, float(max_val)))
        
        # Suppress neighborhood to ensure spatially distinct candidates
        ix, iy = max_loc
        y0, y1 = max(0, iy - r), min(h, iy + r + 1)
        x0, x1 = max(0, ix - r), min(w, ix + r + 1)
        res[y0:y1, x0:x1] = -999.0
        
    return peaks
