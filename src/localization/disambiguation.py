import numpy as np
import cv2

class BoundaryMaskedReranker:
    def __init__(self, hp_kernel=11):
        self.hp_kernel = hp_kernel

    def run(self, reference, search, gspe_candidates):
        if not gspe_candidates:
            return []

        reranked_candidates = []
        scale_cache = {}

        for cand in gspe_candidates:
            x, y, w, h = cand['box']
            scale = cand.get('scale', 10.0)
            
            if scale not in scale_cache:
                scaled_w = int(round(reference.shape[1] / scale))
                scaled_h = int(round(reference.shape[0] / scale))
                ref_scaled = cv2.resize(reference, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA).astype(np.float32)
                
                # High-pass filter reference
                ref_blur = cv2.GaussianBlur(ref_scaled, (self.hp_kernel, self.hp_kernel), 0)
                ref_hp = ref_scaled - ref_blur
                
                # Normalize reference
                ref_hp = (ref_hp - np.mean(ref_hp)) / (np.std(ref_hp) + 1e-6)
                scale_cache[scale] = ref_hp
                
            ref_hp = scale_cache[scale]
            
            x_int, y_int = int(x), int(y)
            w_int, h_int = ref_hp.shape[1], ref_hp.shape[0]
            
            if x_int < 0 or y_int < 0 or x_int + w_int > search.shape[1] or y_int + h_int > search.shape[0]:
                cand['bmr_score'] = -1.0
                reranked_candidates.append(cand)
                continue
                
            cand_crop = search[y_int:y_int+h_int, x_int:x_int+w_int].astype(np.float32)
            
            # High-pass filter candidate
            cand_blur = cv2.GaussianBlur(cand_crop, (self.hp_kernel, self.hp_kernel), 0)
            cand_hp = cand_crop - cand_blur
            
            # Normalize candidate
            cand_hp = (cand_hp - np.mean(cand_hp)) / (np.std(cand_hp) + 1e-6)
            
            # ZNCC on High-Pass
            cov = np.mean(ref_hp * cand_hp)
            bmr_score = float(cov)
            
            cand['bmr_score'] = bmr_score
            reranked_candidates.append(cand)

        # Sort descending by BMR score
        reranked_candidates.sort(key=lambda c: c['bmr_score'], reverse=True)
        return reranked_candidates
