import cv2
import numpy as np

def extract_patch(img, cx, cy, w, h):
    ix, iy = int(round(cx)), int(round(cy))
    img_h, img_w = img.shape
    pad_t = max(0, h//2 - iy)
    pad_b = max(0, iy + h//2 - img_h)
    pad_l = max(0, w//2 - ix)
    pad_r = max(0, ix + w//2 - img_w)
    if pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0:
        padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    else:
        padded = img
    nx, ny = ix + pad_l, iy + pad_t
    return padded[ny - h//2 : ny + h//2, nx - w//2 : nx + w//2]

def get_gradient(img):
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)

def get_edges(img):
    return cv2.Canny(img, 50, 150)

def score_structural(search_patch, ref_patch):
    sg = get_gradient(search_patch)
    rg = get_gradient(ref_patch)
    gncc = cv2.matchTemplate(sg, rg, cv2.TM_CCOEFF_NORMED)[0, 0]
    
    se = get_edges(search_patch)
    re = get_edges(ref_patch)
    if np.max(se) == 0 or np.max(re) == 0:
        encc = 0.0
    else:
        encc = cv2.matchTemplate(se, re, cv2.TM_CCOEFF_NORMED)[0, 0]
        
    mean_val = np.mean(search_patch)
    max_val = np.max(search_patch)
    contrast = float(max_val - mean_val)
    
    return float(gncc), float(encc), contrast

def select_candidate(clusters, search_img, ref_img_100, classification):
    """
    Stage A: Generation / Lattice NMS (done)
    Stage B: Candidate discrimination
    """
    for c in clusters:
        cx, cy = c['cluster_centroid']
        h, w = ref_img_100.shape
        sp = extract_patch(search_img, cx, cy, w, h)
        g, e, con = score_structural(sp, ref_img_100)
        c['structural_scores'] = {
            'gradient': g,
            'edge': e,
            'contrast': con,
            'hybrid_struct': 0.5 * g + 0.5 * e
        }
        
    # Selection Logic
    if classification == "CLEAR":
        # Trust correlation and family support
        best = max(clusters, key=lambda c: c['max_score'])
        best['selection_reason'] = "CLEAR_MARGIN"
        return best
        
    # If ambiguous, use structural evidence
    best = max(clusters, key=lambda c: c['structural_scores']['hybrid_struct'])
    best['selection_reason'] = "STRUCTURAL_TIEBREAKER"
    return best
