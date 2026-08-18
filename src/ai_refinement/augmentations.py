import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2

# Ablation-friendly configuration
AUGMENTATION_CONFIG = {
    "gaussian_noise": True,
    "poisson_noise": True,
    "gaussian_blur": True,
    "brightness": True,
    "contrast": True
}

def add_gaussian_noise(tensor, mean=0., std=0.08): # Stronger Gaussian Noise
    noise = torch.randn_like(tensor) * std + mean
    return torch.clamp(tensor + noise, 0., 1.)

def add_poisson_noise(tensor, lam=30.0): # Lower lambda = higher variance
    noisy = torch.poisson(tensor * lam) / lam
    return torch.clamp(noisy, 0., 1.)

def add_gaussian_blur(tensor, kernel_size=5, sigma=1.5):
    # tensor is (1, H, W)
    return TF.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

def adjust_brightness_contrast(tensor, brightness_factor, contrast_factor):
    t = TF.adjust_brightness(tensor, brightness_factor)
    t = TF.adjust_contrast(t, contrast_factor)
    return t

def apply_industrial_augmentations(ref_patch, cand_patch):
    """
    Applies independent industrial noise to Reference and Candidate patches.
    Does NOT augment the geometric labels (target delta).
    """
    # Photometric augmentations only
    
    # 1. Blur (Simulate defocus)
    if AUGMENTATION_CONFIG["gaussian_blur"]:
        if torch.rand(1).item() > 0.5:
            cand_patch = add_gaussian_blur(cand_patch, kernel_size=5, sigma=1.5)
        if torch.rand(1).item() > 0.8: # Reference is cleaner, less likely to be blurred
            ref_patch = add_gaussian_blur(ref_patch, kernel_size=3, sigma=0.5)

    # 2. Brightness & Contrast (Simulate dosage variations)
    if AUGMENTATION_CONFIG["brightness"] and AUGMENTATION_CONFIG["contrast"]:
        if torch.rand(1).item() > 0.5:
            cand_patch = adjust_brightness_contrast(
                cand_patch, 
                brightness_factor=np.random.uniform(0.8, 1.2),
                contrast_factor=np.random.uniform(0.8, 1.2)
            )
        if torch.rand(1).item() > 0.5:
            ref_patch = adjust_brightness_contrast(
                ref_patch,
                brightness_factor=np.random.uniform(0.9, 1.1),
                contrast_factor=np.random.uniform(0.9, 1.1)
            )

    # 3. Noise (Simulate sensor / electron Poisson noise)
    if AUGMENTATION_CONFIG["gaussian_noise"]:
        if torch.rand(1).item() > 0.5:
            cand_patch = add_gaussian_noise(cand_patch, std=0.08)
    if AUGMENTATION_CONFIG["poisson_noise"]:
        if torch.rand(1).item() > 0.5:
            # Candidate is noisier (lam=20) than Reference (lam=50)
            cand_patch = add_poisson_noise(cand_patch, lam=20.0)
        if torch.rand(1).item() > 0.5:
            ref_patch = add_poisson_noise(ref_patch, lam=50.0)
            
    return ref_patch, cand_patch
