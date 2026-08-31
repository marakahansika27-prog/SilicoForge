import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from src.ai_refinement.config import SNRNConfig
from src.ai_refinement.augmentations import apply_industrial_augmentations
from src.utils.data_loader import load_or_generate_dataset
from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine
from src.feature_matching.gfee import GeometricFeatureExtractionEngine
from src.registration.srae import SpatialRegistrationAlignmentEngine
from src.localization.localization import ClassicalLocalization

class Phase1OutputSimDataset(Dataset):
    """
    Simulates fetching valid classical alignments from Phase 1.
    Uses deterministic geometric transformations to create a mathematically
    perfect self-supervised alignment task for the AI to learn from.
    Includes HARD NEGATIVE generation and GEOMETRIC SCALE FIX.
    
    COORDINATE CONTRACT:
    1. Generator base coords: Center (5000, 5000) on 10000x10000 base image.
    2. Search image coords: Scaled by 1/10. Origin top-left.
    3. Ground-truth coordinate (gt_coord): The CENTER of the true target in search image space (float).
    4. GSPE candidate coord (x, y): The TOP-LEFT of the candidate bounding box in search space (integer).
    5. classical_coord (C_0 or loc_res_w): The classical predicted CENTER of the target in search space (float).
    6. Candidate patch: Extracted around classical_coord, scaled by 1/10 (patch size 128x128).
    7. SNRN residual coordinate (target_delta): (gt_coord - classical_coord), bounded to < 5px.
    8. Final output coordinate: classical_coord + predicted_residual.
    """
    def __init__(self, num_samples=10, apply_aug=True):
        self.num_samples = num_samples
        self.apply_aug = apply_aug
        self._cache = {}
        
        # Load available dataset pairs
        ref_img, search_img = load_or_generate_dataset()
        self.available_pairs = [(ref_img, search_img)]

    def __len__(self):
        return self.num_samples

    def _generate_sample(self, idx):
        max_attempts = 20
        base_seed = 42
        
        for attempt in range(max_attempts):
            seed = base_seed + idx * 10000 + attempt
            rs = np.random.RandomState(seed)
            
            source_id = (idx * 73) % len(self.available_pairs)
            ref_img, search_img = self.available_pairs[source_id]
            
            # 1. Run Phase 1 Coarse Search to get candidates
            ice = ImageConditioningEngine()
            cond = ice.run({'reference': ref_img, 'search': search_img})
            
            gspe = GlobalSearchProposalEngine(
                top_k=5, 
                nms_radius=10,
                scale_hypotheses=[10.0]
            )
            gspe_res = gspe.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
            
            if not gspe_res['boxes']:
                continue
                
            # Determine if this sample will be a POSITIVE (50%) or HARD NEGATIVE (50%)
            is_positive = rs.rand() > 0.5
            
            if is_positive or len(gspe_res['boxes']) == 1:
                cand_idx = 0 # Top-1 is usually the true candidate (for training we assume it is)
            else:
                # Pick a random periodic false candidate (ranks 1 to 4)
                cand_idx = rs.randint(1, min(5, len(gspe_res['boxes'])))
                
            x, y, w, h, _, _ = gspe_res['boxes'][cand_idx]
            cand_crop = cond['search_cond'][int(y):int(y+h), int(x):int(x+w)]
            
            center_x = float(x) + float(w) / 2.0
            center_y = float(y) + float(h) / 2.0
            
            # 2. Compute Base Ground Truth (C_0) by aligning the original crop
            # Only do this if it's a positive candidate, otherwise we don't have a true classical correspondence
            gfee = GeometricFeatureExtractionEngine()
            srae = SpatialRegistrationAlignmentEngine()
            loc = ClassicalLocalization()
            
            # 3. Apply Deterministic Synthetic Transformation to create mathematical residual
            tx = rs.uniform(-3.0, 3.0)
            ty = rs.uniform(-3.0, 3.0)
            angle = rs.uniform(-2.0, 2.0)
            scale_aug = rs.uniform(0.98, 1.02)
            
            M_warp = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale_aug)
            M_warp[0, 2] += tx
            M_warp[1, 2] += ty
            
            warped_cand_crop = cv2.warpAffine(cand_crop, M_warp, (int(w), int(h)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            noise = rs.normal(0, 5.0, warped_cand_crop.shape).astype(np.float32)
            warped_cand_crop = np.clip(warped_cand_crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            target_dx, target_dy = 0.0, 0.0
            classical_coord = np.array([center_x, center_y], dtype=np.float32)
            gt_coord = np.array([-1.0, -1.0], dtype=np.float32)
            
            if is_positive:
                gfee_res_0 = gfee.run({'reference': cond['reference_cond'], 'candidate': cand_crop})
                srae_res_0 = srae.run({
                    'reference': cond['reference_cond'],
                    'candidate': cand_crop,
                    'kp1': gfee_res_0['kp1'],
                    'kp2': gfee_res_0['kp2'],
                    'matches': gfee_res_0['good_matches']
                })
                
                if srae_res_0['stats']['inliers'] < 3:
                    continue
                    
                mat_0 = srae_res_0['affine_matrix'].copy()
                mat_0[0, 2] = -mat_0[0, 2]
                mat_0[1, 2] = -mat_0[1, 2]
                
                loc_res_0 = loc.run({
                    'affine_matrix': mat_0,
                    'candidate_x': center_x,
                    'candidate_y': center_y,
                    'inliers': srae_res_0['stats']['inliers']
                })
                C_0 = np.array([loc_res_0['dx'], loc_res_0['dy']], dtype=np.float32)
                
                # 4. Predict Classical Coordinate on the Warped Image
                gfee_res_w = gfee.run({'reference': cond['reference_cond'], 'candidate': warped_cand_crop})
                srae_res_w = srae.run({
                    'reference': cond['reference_cond'],
                    'candidate': warped_cand_crop,
                    'kp1': gfee_res_w['kp1'],
                    'kp2': gfee_res_w['kp2'],
                    'matches': gfee_res_w['good_matches']
                })
                
                if srae_res_w['stats']['inliers'] < 3:
                    continue
                    
                mat_w = srae_res_w['affine_matrix'].copy()
                mat_w[0, 2] = -mat_w[0, 2]
                mat_w[1, 2] = -mat_w[1, 2]
                
                loc_res_w = loc.run({
                    'affine_matrix': mat_w,
                    'candidate_x': center_x,
                    'candidate_y': center_y,
                    'inliers': srae_res_w['stats']['inliers']
                })
                classical_coord = np.array([loc_res_w['dx'], loc_res_w['dy']], dtype=np.float32)
                
                # 5. Compute Exact Mathematical Target
                obj_rel = np.array([C_0[0] - x, C_0[1] - y, 1.0], dtype=np.float32)
                new_obj_rel = M_warp.dot(obj_rel)
                gt_coord = np.array([x + new_obj_rel[0], y + new_obj_rel[1]], dtype=np.float32)
                
                target_delta = gt_coord - classical_coord
                target_dx, target_dy = target_delta[0], target_delta[1]
                
                residual_mag = float(np.sqrt(target_dx**2 + target_dy**2))
                
                max_error = getattr(SNRNConfig, 'MAX_CLASSICAL_ERROR', 5.0)
                if residual_mag > max_error:
                    continue
            
            # 6. GEOMETRIC SCALE FIX
            # Downsample reference image by 10.0x to match candidate coordinate space
            ref_full = cond['reference_cond']
            scale_ratio = 10.0 # From GSPE 
            w_ref, h_ref = ref_full.shape[1], ref_full.shape[0]
            scaled_w = int(round(w_ref / scale_ratio))
            scaled_h = int(round(h_ref / scale_ratio))
            ref_scaled = cv2.resize(ref_full, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            
            def center_crop(img, size):
                h_img, w_img = img.shape
                ch, cw = size, size
                y_start = max(0, h_img//2 - ch//2)
                x_start = max(0, w_img//2 - cw//2)
                crop = img[y_start:y_start+ch, x_start:x_start+cw]
                if crop.shape[0] < ch or crop.shape[1] < cw:
                    crop = cv2.copyMakeBorder(crop, 0, ch-crop.shape[0], 0, cw-crop.shape[1], cv2.BORDER_CONSTANT, value=0)
                return crop
                
            ref_patch = center_crop(ref_scaled, SNRNConfig.PATCH_SIZE).astype(np.float32) / 255.0
            
            # Sub-pixel extraction of cand_patch centered at classical_coord
            local_class_x = classical_coord[0] - x
            local_class_y = classical_coord[1] - y
            
            M_extract = np.array([
                [1.0, 0.0, (SNRNConfig.PATCH_SIZE / 2.0) - local_class_x],
                [0.0, 1.0, (SNRNConfig.PATCH_SIZE / 2.0) - local_class_y]
            ], dtype=np.float32)
            
            cand_patch_raw = cv2.warpAffine(
                warped_cand_crop, M_extract, 
                (SNRNConfig.PATCH_SIZE, SNRNConfig.PATCH_SIZE), 
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
            cand_patch = cand_patch_raw.astype(np.float32) / 255.0
            
            # 7. Generate Heatmap directly from True Residual
            heatmap = np.zeros((SNRNConfig.PATCH_SIZE, SNRNConfig.PATCH_SIZE), dtype=np.float32)
            x_grid, y_grid = np.meshgrid(np.arange(SNRNConfig.PATCH_SIZE), np.arange(SNRNConfig.PATCH_SIZE))
            
            target_center_x = (SNRNConfig.PATCH_SIZE / 2.0) + target_dx
            target_center_y = (SNRNConfig.PATCH_SIZE / 2.0) + target_dy
            
            gaussian = np.exp(-((x_grid - target_center_x)**2 + (y_grid - target_center_y)**2) / (2 * SNRNConfig.HEATMAP_SIGMA**2))
            heatmap = gaussian / np.sum(gaussian)
            
            # 8. Generate Confidence Label from True Residual and Match Type
            if is_positive:
                conf_label = 1.0
            else:
                conf_label = 0.0 # Hard negative
            
            target_delta_tensor = torch.tensor([target_dx, target_dy], dtype=torch.float32)
            target_heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)
            confidence_label_tensor = torch.tensor([conf_label], dtype=torch.float32)
            
            return (ref_patch, cand_patch, classical_coord, gt_coord, 
                    target_delta_tensor, target_heatmap_tensor, confidence_label_tensor, idx)
                    
        raise RuntimeError(f"Failed to generate a valid sample for index {idx} after {max_attempts} attempts.")

    def __getitem__(self, idx):
        if idx not in self._cache:
            sample_tuple = self._generate_sample(idx)
            assert sample_tuple[-1] == idx, f"Sample index mismatch! Expected {idx}, got {sample_tuple[-1]}"
            self._cache[idx] = sample_tuple
            
        (ref_patch, cand_patch, classical_coord, gt_coord, 
         target_delta, target_heatmap, confidence_label, _) = self._cache[idx]
        
        ref_tensor = torch.from_numpy(ref_patch).unsqueeze(0)
        cand_tensor = torch.from_numpy(cand_patch).unsqueeze(0)
        
        if self.apply_aug:
            ref_tensor, cand_tensor = apply_industrial_augmentations(ref_tensor, cand_tensor)
            
        sample = {
            'reference_patch': ref_tensor,
            'candidate_patch': cand_tensor,
            'classical_coordinate': torch.from_numpy(classical_coord),
            'ground_truth_coordinate': torch.from_numpy(gt_coord),
            'target_delta': target_delta,
            'target_heatmap': target_heatmap,
            'confidence_label': confidence_label
        }
        return sample
