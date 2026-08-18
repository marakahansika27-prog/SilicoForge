import cv2
import numpy as np

def generate_family_variants(ref_img, offsets_x=[-2, -1, 0, 1, 2], offsets_y=[-2, -1, 0, 1, 2]):
    """
    Generates a deterministic family of reference templates by applying controlled spatial offsets.
    The center variant (0,0) perfectly preserves the baseline template geometry.
    
    Offsets are in the coordinate system of the reference image.
    """
    variants = []
    h, w = ref_img.shape
    
    family_id = 0
    for dy in offsets_y:
        for dx in offsets_x:
            # Deterministic translation
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            
            # Use BORDER_REPLICATE to avoid edge artifacts from shifting
            shifted = cv2.warpAffine(ref_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            metadata = {
                "family_id": family_id,
                "offset_x": float(dx),
                "offset_y": float(dy),
                "source_w": w,
                "source_h": h,
                "generated_w": w,
                "generated_h": h
            }
            
            variants.append({
                "template": shifted,
                "metadata": metadata
            })
            family_id += 1
            
    return variants
