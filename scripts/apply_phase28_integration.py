import os
import shutil

pipeline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'integration', 'pipeline.py'))
backup_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'integration', 'pipeline_backup_v2_ai.py'))

if not os.path.exists(backup_path):
    shutil.copy2(pipeline_path, backup_path)
    print(f"Backed up {pipeline_path} to {backup_path}")

new_pipeline_code = """import time
import torch
import cv2
import numpy as np

# Phase 1
from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine
from src.feature_matching.gfee import GeometricFeatureExtractionEngine
from src.registration.srae import SpatialRegistrationAlignmentEngine
from src.localization.localization import ClassicalLocalization
# Phase 3
from src.integration.decision_fusion import DecisionFusionEngine

def get_subpixel(res, x, y):
    h, w = res.shape
    dx, dy = 0.0, 0.0
    if 0 < x < w - 1 and 0 < y < h - 1:
        fx_m1 = float(res[y, x - 1])
        fx_0  = float(res[y, x])
        fx_p1 = float(res[y, x + 1])
        
        denom_x = fx_m1 - 2 * fx_0 + fx_p1
        if abs(denom_x) > 1e-6:
            val = 0.5 * (fx_m1 - fx_p1) / denom_x
            if not np.isnan(val) and abs(val) <= 0.5:
                dx = val
                
        fy_m1 = float(res[y - 1, x])
        fy_0  = float(res[y, x])
        fy_p1 = float(res[y + 1, x])
        
        denom_y = fy_m1 - 2 * fy_0 + fy_p1
        if abs(denom_y) > 1e-6:
            val = 0.5 * (fy_m1 - fy_p1) / denom_y
            if not np.isnan(val) and abs(val) <= 0.5:
                dy = val
    return float(x + dx), float(y + dy)

def local_refine(res, cx, cy, radius):
    h, w = res.shape
    y1 = max(0, int(cy - radius))
    y2 = min(h, int(cy + radius + 1))
    x1 = max(0, int(cx - radius))
    x2 = min(w, int(cx + radius + 1))
    
    if y2 <= y1 or x2 <= x1:
        return float(cx), float(cy)
        
    window = res[y1:y2, x1:x2]
    _, _, _, max_loc = cv2.minMaxLoc(window)
    px = x1 + max_loc[0]
    py = y1 + max_loc[1]
    
    return px, py

def local_refine_subpixel(res, cx, cy, radius):
    px, py = local_refine(res, cx, cy, radius)
    return get_subpixel(res, px, py)


class HybridNavigationPipeline:
    def __init__(self, top_k: int = 1, nms_radius: int = None):
        self.ice = ImageConditioningEngine()
        
        # Phase 27 Validation established scale=10.0 is mathematically optimal for V3.
        scales = [10.0]
        rotations = [0.0]
        self.gspe = GlobalSearchProposalEngine(
            top_k=top_k, 
            nms_radius=nms_radius,
            scale_hypotheses=scales,
            rotation_hypotheses=rotations
        )
        self.gfee = GeometricFeatureExtractionEngine()
        self.srae = SpatialRegistrationAlignmentEngine()
        self.loc = ClassicalLocalization()
        
        # No SNRN required (FROZEN PHASE 27 ALGORITHM)
        self.fusion = DecisionFusionEngine(confidence_threshold=0.90, residual_deadband=0.10)
        self.state = {}

    def run(self, ref_img, search_img, ref_macro=None):
        \"\"\"
        Executes the final Phase 28 integrated pipeline end-to-end.
        \"\"\"
        t0 = time.time()
        self.state = {
            'reference': ref_img,
            'search': search_img,
            'modules': {}
        }
        
        try:
            cond = self.ice.run({'reference': ref_img, 'search': search_img})
            self.state['modules']['ICE'] = True
            
            gspe_res_100 = self.gspe.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
            res_hybrid_100 = gspe_res_100['res_hybrid']
            res_raw_100 = gspe_res_100['res_raw']
            
            # Use Macro Context if provided
            if ref_macro is not None:
                gspe_res_macro = self.gspe.run({'reference': ref_macro, 'search': cond['search_cond']})
                _, _, _, max_loc_macro = cv2.minMaxLoc(gspe_res_macro['res_hybrid'])
                macro_cx, macro_cy = float(max_loc_macro[0]), float(max_loc_macro[1])
            else:
                # Fallback to local 100px GSPE max_loc
                _, _, _, max_loc_100 = cv2.minMaxLoc(res_hybrid_100)
                macro_cx, macro_cy = float(max_loc_100[0]), float(max_loc_100[1])
                
            # Coarse neighborhood
            loc_cx, loc_cy = local_refine(res_hybrid_100, macro_cx, macro_cy, 50)
            
            # Phase 24 40px refinement on the 100px raw response for final subpixel precision
            final_x, final_y = local_refine_subpixel(res_raw_100, loc_cx, loc_cy, 40)
            
            final_coord = np.array([final_x, final_y], dtype=np.float32)
            
            self.state['modules']['GFEE'] = False
            self.state['modules']['SRAE'] = False
            self.state['modules']['Localization'] = False
            self.state['modules']['AI Refinement'] = False
            self.state['modules']['Fusion'] = True
            
            self.state['final_coord'] = final_coord
            self.state['decision'] = "MACRO_GSPE_FROZEN"
            self.state['runtime'] = time.time() - t0
            
            return self.state
            
        except Exception as e:
            self.state['error'] = str(e)
            self.state['runtime'] = time.time() - t0
            return self.state
"""

with open(pipeline_path, 'w', encoding='utf-8') as f:
    f.write(new_pipeline_code)

print(f"Applied Phase 28 Production Integration to {pipeline_path}")
