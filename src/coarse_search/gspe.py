import cv2
import numpy as np
from typing import List, Tuple, Dict
from src.utils.logger import Profiler

class GlobalSearchProposalEngine:
    """
    Global Search & Proposal Engine (GSPE)
    Uses Normalized Cross Correlation to extract Top-K candidate regions.
    """
    def __init__(self, top_k: int = 5, nms_radius: int = None, 
                 scale_hypotheses: List[float] = None, 
                 rotation_hypotheses: List[float] = None):
        self.top_k = top_k
        self.nms_radius = nms_radius
        
        # Geometry-aware coarse search configuration
        self.scale_hypotheses = scale_hypotheses if scale_hypotheses is not None else [10.0]
        self.rotation_hypotheses = rotation_hypotheses if rotation_hypotheses is not None else [0.0]
        
        self.stats = {}

    def run(self, inputs: dict) -> dict:
        """
        inputs: {'reference': ndarray, 'search': ndarray}
        returns: {'boxes': List[Tuple], 'scores': List[float], 'stats': dict, 'heatmap': ndarray}
        """
        ref_img = inputs['reference']
        search_img = inputs['search']
        
        with Profiler("GSPE_Run") as p:
            h_ref, w_ref = ref_img.shape
            search_h, search_w = search_img.shape
            
            search_blurred = cv2.GaussianBlur(search_img, (31, 31), 15)
            
            best_res = None
            best_res_raw = None
            best_res_lowfreq = None
            best_scale_map = None
            best_rot_map = None
            best_w_map = None
            best_h_map = None
            
            # Geometry-aware sweep
            for scale in self.scale_hypotheses:
                for rot in self.rotation_hypotheses:
                    
                    # Compute transformed template size in search image space
                    # The scaling factor maps reference pixels to search pixels.
                    # effective_scale = scale.
                    # A scale of 10.0 means 1 search pixel = 10 reference pixels.
                    new_w = int(round(w_ref / scale))
                    new_h = int(round(h_ref / scale))
                    
                    # 1. Rotate the reference image first
                    M_rot = cv2.getRotationMatrix2D((w_ref / 2.0, h_ref / 2.0), rot, 1.0)
                    
                    # To prevent cropping corners, we compute bounding box of rotated image
                    cos_val = np.abs(M_rot[0, 0])
                    sin_val = np.abs(M_rot[0, 1])
                    bound_w = int((h_ref * sin_val) + (w_ref * cos_val))
                    bound_h = int((h_ref * cos_val) + (w_ref * sin_val))
                    
                    # Adjust transform to center
                    M_rot[0, 2] += (bound_w / 2) - (w_ref / 2)
                    M_rot[1, 2] += (bound_h / 2) - (h_ref / 2)
                    
                    ref_rotated = cv2.warpAffine(ref_img, M_rot, (bound_w, bound_h), 
                                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                                 
                    # INSCRIBED CROP: Remove all border artifacts caused by rotation
                    theta = np.deg2rad(np.abs(rot))
                    crop_w = int(w_ref / (np.sin(theta) + np.cos(theta)))
                    crop_h = int(h_ref / (np.sin(theta) + np.cos(theta)))
                    
                    cx_rot = bound_w // 2
                    cy_rot = bound_h // 2
                    y1 = max(0, cy_rot - crop_h // 2)
                    y2 = y1 + crop_h
                    x1 = max(0, cx_rot - crop_w // 2)
                    x2 = x1 + crop_w
                    
                    ref_rotated_cropped = ref_rotated[y1:y2, x1:x2]
                    
                    # 2. Scale it down to search space
                    scaled_bound_w = int(round(crop_w / scale))
                    scaled_bound_h = int(round(crop_h / scale))
                    
                    if scaled_bound_w > search_w or scaled_bound_h > search_h:
                        raise ValueError(f"Template size ({scaled_bound_w}x{scaled_bound_h}) exceeds search image size ({search_w}x{search_h}). "
                                         f"Check scale hypothesis ({scale}).")
                                         
                    ref_scaled = cv2.resize(ref_rotated_cropped, (scaled_bound_w, scaled_bound_h), interpolation=cv2.INTER_AREA)
                    
                    # 3. Raw NCC
                    res_raw = cv2.matchTemplate(search_img, ref_scaled, cv2.TM_CCOEFF_NORMED)
                    
                    # 4. Low-Frequency NCC
                    ref_blurred = cv2.GaussianBlur(ref_scaled, (31, 31), 15)
                    res_lowfreq = cv2.matchTemplate(search_blurred, ref_blurred, cv2.TM_CCOEFF_NORMED)
                    
                    # 5. Hybrid Score
                    res = 0.5 * res_raw + 0.5 * res_lowfreq
                    
                    # Since template size varies, the output response map size varies slightly.
                    # We pad the response map so it has the same size (search_h, search_w) 
                    # by offsetting it by the template center, so the response value exactly 
                    # corresponds to the CENTER of the matched template in the search image.
                    pad_top = scaled_bound_h // 2
                    pad_bottom = search_h - res.shape[0] - pad_top
                    pad_left = scaled_bound_w // 2
                    pad_right = search_w - res.shape[1] - pad_left
                    
                    assert pad_top >= 0 and pad_bottom >= 0 and pad_left >= 0 and pad_right >= 0, \
                        f"Negative padding calculated! top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}. " \
                        f"Search Size: {search_w}x{search_h}. Response Size: {res.shape[1]}x{res.shape[0]}. " \
                        f"Template Size: {scaled_bound_w}x{scaled_bound_h}."
                    
                    res_centered = cv2.copyMakeBorder(res, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=-1.0)
                    res_raw_centered = cv2.copyMakeBorder(res_raw, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=-1.0)
                    res_lowfreq_centered = cv2.copyMakeBorder(res_lowfreq, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=-1.0)
                    
                    # Initialize cumulative maps if this is the first iteration
                    if best_res is None:
                        best_res = np.full_like(res_centered, -1.0)
                        best_res_raw = np.full_like(res_centered, -1.0)
                        best_res_lowfreq = np.full_like(res_centered, -1.0)
                        best_scale_map = np.zeros_like(res_centered)
                        best_rot_map = np.zeros_like(res_centered)
                        best_w_map = np.zeros_like(res_centered, dtype=np.int32)
                        best_h_map = np.zeros_like(res_centered, dtype=np.int32)
                        
                    # Update maps where the new hybrid score is better
                    mask = res_centered > best_res
                    best_res[mask] = res_centered[mask]
                    best_res_raw[mask] = res_raw_centered[mask]
                    best_res_lowfreq[mask] = res_lowfreq_centered[mask]
                    best_scale_map[mask] = scale
                    best_rot_map[mask] = rot
                    best_w_map[mask] = scaled_bound_w
                    best_h_map[mask] = scaled_bound_h
                    
            res = best_res
            res_raw = best_res_raw
            res_lowfreq = best_res_lowfreq
            
            # Diagnostic Logging
            _, max_val_raw, _, _ = cv2.minMaxLoc(res_raw)
            _, max_val_low, _, _ = cv2.minMaxLoc(res_lowfreq)
            _, max_val_hyb, _, max_loc_hyb = cv2.minMaxLoc(res)
            
            best_overall_scale = best_scale_map[max_loc_hyb[1], max_loc_hyb[0]]
            best_overall_rot = best_rot_map[max_loc_hyb[1], max_loc_hyb[0]]
            
            print("\n--- GSPE HYBRID DIAGNOSTIC ---")
            print(f"Hypotheses Evaluated    : {len(self.scale_hypotheses) * len(self.rotation_hypotheses)}")
            print(f"Raw NCC Top-1 Score     : {max_val_raw:.4f}")
            print(f"LowFreq NCC Top-1 Score : {max_val_low:.4f}")
            print(f"Hybrid Top-1 Score      : {max_val_hyb:.4f}")
            print(f"Hybrid Top-1 Center     : {max_loc_hyb}")
            print(f"Top-1 Geometry          : Scale {best_overall_scale:.3f}, Rot {best_overall_rot:.3f}")
            
            # Add to stats for telemetry
            self.stats['hybrid_top1_score'] = float(max_val_hyb)
            self.stats['raw_top1_score'] = float(max_val_raw)
            self.stats['lowfreq_top1_score'] = float(max_val_low)
            self.stats['best_scale'] = float(best_overall_scale)
            self.stats['best_rot'] = float(best_overall_rot)
            
            # Find Top-K geographically distinct peaks via NMS
            boxes = []
            scores = []
            
            # Use a copy of res for NMS
            res_nms = res.copy()
            # Use a safe 10px radius to avoid suppressing distinct periodic candidates
            nms_radius = self.nms_radius if self.nms_radius is not None else 10
            
            # Count raw peaks (above mean) for diagnostic purposes
            raw_peaks = np.sum(res > np.mean(res) + np.std(res))
            self.stats['raw_peaks'] = int(raw_peaks)
            
            class SubpixelFloat(float):
                def __new__(cls, value):
                    return super().__new__(cls, value)
                def __index__(self):
                    return int(self)
                def __add__(self, other):
                    return SubpixelFloat(super().__add__(other))
                def __radd__(self, other):
                    return SubpixelFloat(super().__radd__(other))
            
            for _ in range(self.top_k):
                _, max_val, _, max_loc = cv2.minMaxLoc(res_nms)
                if max_val < 0: # Correlation too low
                    break
                    
                x, y = max_loc
                
                # --- SUBPIXEL INTERPOLATION ---
                delta_x = 0.0
                delta_y = 0.0
                
                if x > 0 and x < res.shape[1] - 1 and y > 0 and y < res.shape[0] - 1:
                    fx_m1 = float(res[y, x - 1])
                    fx_0  = float(res[y, x])
                    fx_p1 = float(res[y, x + 1])
                    
                    denom_x = fx_m1 - 2 * fx_0 + fx_p1
                    if abs(denom_x) > 1e-6:
                        dx = 0.5 * (fx_m1 - fx_p1) / denom_x
                        if not (np.isnan(dx) or np.isinf(dx)) and abs(dx) <= 0.5:
                            delta_x = dx
                            
                    fy_m1 = float(res[y - 1, x])
                    fy_0  = float(res[y, x])
                    fy_p1 = float(res[y + 1, x])
                    
                    denom_y = fy_m1 - 2 * fy_0 + fy_p1
                    if abs(denom_y) > 1e-6:
                        dy = 0.5 * (fy_m1 - fy_p1) / denom_y
                        if not (np.isnan(dy) or np.isinf(dy)) and abs(dy) <= 0.5:
                            delta_y = dy
                
                sub_x = SubpixelFloat(x + delta_x)
                sub_y = SubpixelFloat(y + delta_y)
                
                print(f"\n--- GSPE SUBPIXEL DIAGNOSTIC ---")
                print(f"Integer Peak : ({float(x):.4f}, {float(y):.4f})")
                print(f"Delta        : ({delta_x:+.4f}, {delta_y:+.4f})")
                print(f"Subpixel    : ({float(sub_x):.4f}, {float(sub_y):.4f})")
                
                scores.append(float(max_val))
                w_cand = best_w_map[y, x]
                h_cand = best_h_map[y, x]
                scale_cand = best_scale_map[y, x]
                rot_cand = best_rot_map[y, x]
                
                # We return the top-left coordinate as per convention (sub_x - w/2, sub_y - h/2)
                # But since the sub_x, sub_y are now explicitly the CENTER coordinate (due to centering border),
                # the top-left representation is:
                tl_x = sub_x - (w_cand / 2.0)
                tl_y = sub_y - (h_cand / 2.0)
                
                boxes.append((tl_x, tl_y, int(w_cand), int(h_cand), scale_cand, rot_cand))
                
                # Suppress this region
                nms_r = int(nms_radius)
                y1, y2 = max(0, y - nms_r), min(res_nms.shape[0], y + nms_r)
                x1, x2 = max(0, x - nms_r), min(res_nms.shape[1], x + nms_r)
                res_nms[y1:y2, x1:x2] = -1.0
                
            print(f"Top-{self.top_k} Hybrid Candidates Extracted:")
            for rank, (b, s) in enumerate(zip(boxes, scores)):
                # b is (tl_x, tl_y, w, h, scale, rot)
                center_x = b[0] + b[2] / 2.0
                center_y = b[1] + b[3] / 2.0
                print(f"  Rank {rank+1}: score={s:.4f}, center=({float(center_x):.4f}, {float(center_y):.4f}), geom=(Scale {b[4]:.3f}, Rot {b[5]:.3f})")
                
            # Normalize heatmap for visualization
            heatmap = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        self.stats['runtime_ms'] = p.elapsed_ms
        self.stats['memory_kb'] = p.mem_diff_kb
        self.stats['heatmap_dims'] = str(heatmap.shape)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'heatmap': heatmap,
            'stats': self.stats,
            'res_raw': res_raw,
            'res_lowfreq': res_lowfreq,
            'res_hybrid': res,
            'best_scale_map': best_scale_map,
            'best_rot_map': best_rot_map,
            'best_w_map': best_w_map,
            'best_h_map': best_h_map
        }
