import os
import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("DataLoader")

def load_or_generate_dataset(data_dir: str = "dataset"):
    """
    Checks if real dataset exists. If not, generates synthetic DRAM-like images
    for unit testing fallback.
    """
    ref_path = os.path.join(data_dir, "reference.png")
    search_path = os.path.join(data_dir, "search.png")
    
    if os.path.exists(ref_path) and os.path.exists(search_path):
        logger.info("Real dataset found. Loading...")
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        return ref_img, search_img
        
    # Generate high-resolution base image (1 unit = 1 px)
    # The physical scene is 10240 x 10240 units
    base_size = 10240
    base_img = np.zeros((base_size, base_size), dtype=np.float32)
    
    # Draw structures at high resolution
    # Dram pitch = 300 units (Search pitch = 30 px)
    for i in range(150, base_size, 300):
        for j in range(150, base_size, 300):
            cv2.circle(base_img, (i, j), 60, 150, -1)
            cv2.rectangle(base_img, (i-45, j-45), (i+45, j+45), 80, 6)
            
    # SEM Edge Brightening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    gradient = cv2.morphologyEx(base_img, cv2.MORPH_GRADIENT, kernel)
    sem_img = np.clip(base_img + gradient * 1.5, 0, 255).astype(np.float32)
    
    # Extract Reference (Clean) - 900x900 px
    # Add rejection sampling to prevent empty references
    valid = False
    attempts = 0
    while not valid and attempts < 100:
        attempts += 1
        offset_x = np.random.randint(2000, 8000)
        offset_y = np.random.randint(2000, 8000)
        ref_float = sem_img[offset_y:offset_y+900, offset_x:offset_x+900].copy()
        
        # Validation: Ref must have meaningful structure
        # Check standard deviation (structural energy)
        if np.std(ref_float) > 10.0:
            valid = True
            
    lam_ref = 50.0
    ref_noisy = np.random.poisson(ref_float / 255.0 * lam_ref) / lam_ref * 255.0
    ref_img = np.clip(ref_noisy, 0, 255).astype(np.uint8)
    
    # Generate Search Image - 1024x1024 px (downsampled by 10x)
    # 10240x10240 -> 1024x1024
    search_float = cv2.resize(sem_img, (1024, 1024), interpolation=cv2.INTER_AREA)
    
    # Search degradation
    search_blur = cv2.GaussianBlur(search_float, (3, 3), 0.8)
    lam_search = 20.0
    search_noisy = np.random.poisson(search_blur / 255.0 * lam_search) / lam_search * 255.0
    search_img = np.clip(search_noisy, 0, 255).astype(np.uint8)
    
    # Ground Truth coordinate in Search Image
    # Reference was taken at (offset_x, offset_y) in base coordinates.
    # The center of the 900x900 reference is (offset_x + 450, offset_y + 450)
    gt_x = (offset_x + 450.0) / 10.0
    gt_y = (offset_y + 450.0) / 10.0
    
    cv2.imwrite(ref_path, ref_img)
    cv2.imwrite(search_path, search_img)
    
    return ref_img, search_img, gt_x, gt_y
