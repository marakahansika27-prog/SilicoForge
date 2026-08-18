import os
import cv2
import numpy as np
from dataset.metadata import CaseMetadata

class HackathonDatasetGenerator:
    def __init__(self, seed: int):
        self.seed = seed
        self.layout_rng = np.random.RandomState(seed)
        self.ref_noise_rng = np.random.RandomState(seed + 1)
        self.search_noise_rng = np.random.RandomState(seed + 2)
        self.aug_rng = np.random.RandomState(seed + 3)

    def generate_case(self, case_id: str, architecture: str, difficulty: str, version: str = "v1", spatial_region: str = None, ref_size: int = 1000):
        # 1. Base Geometry
        base_size = 10000
        base_img = np.zeros((base_size, base_size), dtype=np.float32)
        
        # Determine randomized placement of the main active region
        # Ensures that the border is variable
        start_x = self.layout_rng.randint(1500, 2500)
        start_y = self.layout_rng.randint(1500, 2500)
        end_x = start_x + 6000
        end_y = start_y + 6000

        # Architecture-specific drawing
        if architecture == "DRAM":
            pitch_x, pitch_y = 300, 300
            feature_w, feature_h = 120, 120 # Circle diameter
            for i in range(150, base_size, pitch_x):
                if not (start_x <= i < end_x): continue
                for j in range(150, base_size, pitch_y):
                    if not (start_y <= j < end_y): continue
                    cv2.circle(base_img, (i, j), 60, 180, -1)
                    cv2.rectangle(base_img, (i-45, j-45), (i+45, j+45), 100, 6)
        elif architecture == "FinFET":
            pitch_x, pitch_y = 200, 600
            feature_w, feature_h = 30, 15
            for i in range(150, base_size, pitch_x):
                if not (start_x <= i < end_x): continue
                cv2.line(base_img, (i, start_y), (i, end_y), 180, feature_w)
            for j in range(150, base_size, pitch_y):
                if not (start_y <= j < end_y): continue
                cv2.line(base_img, (start_x, j), (end_x, j), 120, feature_h)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Edge region structure
        cv2.rectangle(base_img, (start_x-200, start_y-200), (end_x+200, end_y+200), 50, 40)

        # 2. SEM-style Edge Brightening
        edge_strength = 1.5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        gradient = cv2.morphologyEx(base_img, cv2.MORPH_GRADIENT, kernel)
        base_img = np.clip(base_img + gradient * edge_strength, 0, 255).astype(np.float32)

        # 3. Augmentation Parameters
        if difficulty == "easy":
            ref_noise_level = 20
            search_noise_level = 20
            blur = 0
            rot = self.aug_rng.uniform(-1.0, 1.0)
            scale = self.aug_rng.uniform(0.98, 1.02)
        elif difficulty == "moderate":
            ref_noise_level = 20
            search_noise_level = 50
            blur = 3
            rot = self.aug_rng.uniform(-3.0, 3.0)
            scale = self.aug_rng.uniform(0.95, 1.05)
        elif difficulty == "hard":
            ref_noise_level = 20
            search_noise_level = 90
            blur = 5
            rot = self.aug_rng.uniform(-5.0, 5.0)
            scale = self.aug_rng.uniform(0.90, 1.10)
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        # 4. Search Image Generation (Downscale 10x)
        # 10000x10000 -> 1000x1000
        search_float = cv2.resize(base_img, (1000, 1000), interpolation=cv2.INTER_AREA)
        
        # Apply Blur to search
        if blur > 0:
            search_float = cv2.GaussianBlur(search_float, (blur, blur), 0)

        # 5. Reference Image Extraction (Native Crop + Augmentation)
        valid_reference = False
        resample_attempts = 0
        
        while not valid_reference:
            resample_attempts += 1
            if resample_attempts > 50:
                raise RuntimeError(f"Failed to find valid reference after 50 attempts for {case_id}")
                
            if version == "v1":
                # Choose a random crop center inside the active region (padding safely by 1000)
                base_cx = self.layout_rng.randint(start_x + 1000, end_x - 1000)
                base_cy = self.layout_rng.randint(start_y + 1000, end_y - 1000)
                context_type = None
            elif version == "v2":
                # Sample from a macro boundary deterministically
                context_types = ['left_edge', 'right_edge', 'top_edge', 'bottom_edge',
                                 'top_left_corner', 'top_right_corner', 'bottom_left_corner', 'bottom_right_corner']
                context_type = self.layout_rng.choice(context_types)
                
                # Jitter helps avoid the exact same boundary framing every time
                jitter_x = self.layout_rng.randint(-150, 150)
                jitter_y = self.layout_rng.randint(-150, 150)
                
                if context_type == 'left_edge':
                    base_cx = start_x + jitter_x
                    base_cy = self.layout_rng.randint(start_y + 500, end_y - 500)
                elif context_type == 'right_edge':
                    base_cx = end_x + jitter_x
                    base_cy = self.layout_rng.randint(start_y + 500, end_y - 500)
                elif context_type == 'top_edge':
                    base_cx = self.layout_rng.randint(start_x + 500, end_x - 500)
                    base_cy = start_y + jitter_y
                elif context_type == 'bottom_edge':
                    base_cx = self.layout_rng.randint(start_x + 500, end_x - 500)
                    base_cy = end_y + jitter_y
                elif context_type == 'top_left_corner':
                    base_cx = start_x + jitter_x
                    base_cy = start_y + jitter_y
                elif context_type == 'top_right_corner':
                    base_cx = end_x + jitter_x
                    base_cy = start_y + jitter_y
                elif context_type == 'bottom_left_corner':
                    base_cx = start_x + jitter_x
                    base_cy = end_y + jitter_y
                elif context_type == 'bottom_right_corner':
                    base_cx = end_x + jitter_x
                    base_cy = end_y + jitter_y
            elif version == "v3":
                safe_margin = 1000
                if spatial_region == "center":
                    cx = (start_x + end_x) // 2
                    cy = (start_y + end_y) // 2
                    base_cx = cx + self.layout_rng.randint(-150, 150)
                    base_cy = cy + self.layout_rng.randint(-150, 150)
                elif spatial_region == "interior":
                    base_cx = self.layout_rng.randint(start_x + 1500, end_x - 1500)
                    base_cy = self.layout_rng.randint(start_y + 1500, end_y - 1500)
                elif spatial_region == "left_boundary":
                    base_cx = start_x + self.layout_rng.randint(-100, 100)
                    base_cy = self.layout_rng.randint(start_y + 1500, end_y - 1500)
                elif spatial_region == "right_boundary":
                    base_cx = end_x + self.layout_rng.randint(-100, 100)
                    base_cy = self.layout_rng.randint(start_y + 1500, end_y - 1500)
                elif spatial_region == "top_boundary":
                    base_cx = self.layout_rng.randint(start_x + 1500, end_x - 1500)
                    base_cy = start_y + self.layout_rng.randint(-100, 100)
                elif spatial_region == "bottom_boundary":
                    base_cx = self.layout_rng.randint(start_x + 1500, end_x - 1500)
                    base_cy = end_y + self.layout_rng.randint(-100, 100)
                elif spatial_region == "top_left_corner":
                    base_cx = start_x + self.layout_rng.randint(-100, 100)
                    base_cy = start_y + self.layout_rng.randint(-100, 100)
                elif spatial_region == "top_right_corner":
                    base_cx = end_x + self.layout_rng.randint(-100, 100)
                    base_cy = start_y + self.layout_rng.randint(-100, 100)
                elif spatial_region == "bottom_left_corner":
                    base_cx = start_x + self.layout_rng.randint(-100, 100)
                    base_cy = end_y + self.layout_rng.randint(-100, 100)
                elif spatial_region == "bottom_right_corner":
                    base_cx = end_x + self.layout_rng.randint(-100, 100)
                    base_cy = end_y + self.layout_rng.randint(-100, 100)
                elif spatial_region == "random":
                    base_cx = self.layout_rng.randint(safe_margin, base_size - safe_margin)
                    base_cy = self.layout_rng.randint(safe_margin, base_size - safe_margin)
                else:
                    raise ValueError(f"Unknown spatial region: {spatial_region}")
            
            # Compute M mapping from ref_size x ref_size reference image to the 10000x10000 base image
            # cv2.getRotationMatrix2D applies rotation around center, and scales.
            M = cv2.getRotationMatrix2D((ref_size/2.0, ref_size/2.0), rot, scale)
            M[0, 2] += (base_cx - ref_size/2.0)
            M[1, 2] += (base_cy - ref_size/2.0)
            
            # Extract Reference
            ref_float = cv2.warpAffine(base_img, M, (ref_size, ref_size), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Image Content Diagnostics (Before Noise)
            variance = float(np.var(ref_float))
            grad_x = cv2.Sobel(ref_float, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(ref_float, cv2.CV_64F, 0, 1, ksize=3)
            grad_energy = float(np.mean(grad_x**2 + grad_y**2))
            
            edges = cv2.Canny(ref_float.astype(np.uint8), 50, 150)
            edge_density = float(np.count_nonzero(edges) / edges.size)
            
            # Periodicity Score: Cross-correlation with a shifted self to measure periodicity
            # A highly periodic region will have a strong peak at the pitch displacement.
            # Using structural pitch mapped to reference coordinates:
            # pitch in base is pitch_x, mapped to reference it's approx pitch_x
            if version == "v2":
                shift_x = int(pitch_x * scale)
                if shift_x < ref_float.shape[1]:
                    shifted = np.roll(ref_float, -shift_x, axis=1)
                    # Mask out the wrap-around
                    valid_mask = np.ones_like(ref_float)
                    valid_mask[:, -shift_x:] = 0
                    if np.sum(valid_mask) > 0:
                        norm_ref = ref_float - np.mean(ref_float)
                        norm_shifted = shifted - np.mean(shifted)
                        std_ref = np.std(ref_float)
                        std_shifted = np.std(shifted[valid_mask == 1])
                        if std_ref > 1e-5 and std_shifted > 1e-5:
                            periodicity_score = np.mean(norm_ref[valid_mask == 1] * norm_shifted[valid_mask == 1]) / (std_ref * std_shifted)
                        else:
                            periodicity_score = 1.0 # Flat is essentially perfectly ambiguous
                    else:
                        periodicity_score = 1.0
                else:
                    periodicity_score = 1.0
            else:
                periodicity_score = 0.0
                
            macro_boundary_present = (version == "v2")
            
            # Validation logic
            if version == "v2":
                if variance < 100:
                    continue # Structurally empty / nearly uniform
                if grad_energy < 500:
                    continue # Lacking features
                if edge_density < 0.01:
                    continue # Lacking sharp boundaries
                if ref_size < 4000 and periodicity_score > 0.95:
                    continue # Too strongly periodic, missing macro transition
                elif ref_size >= 4000 and periodicity_score > 0.95:
                    # For large macros, periodicity is expected.
                    # Ensure it's not genuinely featureless (already checked by variance/edge_density).
                    # We accept it if it has sufficient structural information.
                    pass
            elif version == "v3":
                # For V3, only ensure it's not a completely empty black patch
                if variance < 10:
                    continue
                    
            valid_reference = True
        
        # 6. Apply Independent Noise
        ref_noise = self.ref_noise_rng.poisson(ref_float / 255.0 * ref_noise_level) / ref_noise_level * 255
        ref_img = np.clip(ref_float + ref_noise - 128, 0, 255).astype(np.uint8)
        
        search_noise = self.search_noise_rng.poisson(search_float / 255.0 * search_noise_level) / search_noise_level * 255
        search_img = np.clip(search_float + search_noise - 128, 0, 255).astype(np.uint8)

        # 7. Ground Truth Calculation
        # Map reference corners to search image coordinates
        pts_ref = np.array([
            [0, 0, 1],
            [ref_size, 0, 1],
            [ref_size, ref_size, 1],
            [0, ref_size, 1]
        ]).T # 3 x 4
        
        pts_base = M.dot(pts_ref) # 2 x 4
        # Map from base image (10000x10000) to search image (1000x1000)
        pts_search = pts_base / 10.0
        
        gt_x = float(base_cx / 10.0)
        gt_y = float(base_cy / 10.0)
        
        min_x, max_x = np.min(pts_search[0]), np.max(pts_search[0])
        min_y, max_y = np.min(pts_search[1]), np.max(pts_search[1])
        
        gt_bbox = {
            "x": float(min_x),
            "y": float(min_y),
            "width": float(max_x - min_x),
            "height": float(max_y - min_y)
        }

        # 8. Metadata
        metadata = CaseMetadata(
            case_id=case_id,
            architecture=architecture,
            difficulty=difficulty,
            seed=self.seed,
            search_width=1000,
            search_height=1000,
            reference_width=1000,
            reference_height=1000,
            nominal_scale=10.0,
            augmentation_scale=float(scale),
            effective_scale=float(10.0 * scale),
            rotation_degrees=float(rot),
            gt_x=gt_x,
            gt_y=gt_y,
            gt_bbox=gt_bbox,
            reference_noise_level=float(ref_noise_level),
            search_noise_level=float(search_noise_level),
            blur_kernel=int(blur),
            edge_brightening_strength=float(edge_strength),
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            feature_width=feature_w,
            feature_height=feature_h
        )
        
        if version == "v2":
            metadata.reference_context_type = context_type
            metadata.macro_boundary_present = macro_boundary_present
            metadata.periodicity_score = float(periodicity_score)
            metadata.edge_density = float(edge_density)
            metadata.gradient_energy = float(grad_energy)
            metadata.local_variance = float(variance)
            metadata.reference_sampling_rule = "v2_macro_boundary"
            metadata.reference_origin_x = int(base_cx)
            metadata.reference_origin_y = int(base_cy)
        elif version == "v3":
            metadata.spatial_region = spatial_region
            metadata.periodicity_score = float(periodicity_score)
            metadata.edge_density = float(edge_density)
            metadata.gradient_energy = float(grad_energy)
            metadata.local_variance = float(variance)
            metadata.reference_sampling_rule = "v3_spatial_benchmark"
            metadata.reference_origin_x = int(base_cx)
            metadata.reference_origin_y = int(base_cy)

        return ref_img, search_img, metadata
