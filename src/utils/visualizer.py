import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def save_image(img: np.ndarray, save_dir: str, filename: str, is_rgb: bool = False):
    """Saves an image to disk."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    if is_rgb:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img_bgr)
    else:
        cv2.imwrite(filepath, img)

def plot_histogram(img: np.ndarray, save_dir: str, filename: str, title: str = "Histogram"):
    """Plots and saves an image histogram."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title(title)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def save_overlay(img1: np.ndarray, img2: np.ndarray, save_dir: str, filename: str):
    """Saves an overlay (difference map) of two images."""
    os.makedirs(save_dir, exist_ok=True)
    diff = cv2.absdiff(img1, img2)
    overlay = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_dir, filename), overlay)
