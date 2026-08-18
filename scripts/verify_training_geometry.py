import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ai_refinement.dataset import Phase1OutputSimDataset

def main():
    print("========================================")
    print("TRAINING GEOMETRY VERIFICATION")
    print("========================================")
    
    # 1. Instantiate the dataset (without aug first to just see the raw geometric generation)
    ds = Phase1OutputSimDataset(num_samples=50, apply_aug=False)
    
    out_dir = os.path.join("outputs", "debug", "training_geometry")
    os.makedirs(out_dir, exist_ok=True)
    
    dx_list = []
    dy_list = []
    mag_list = []
    
    large_pos_dx = None
    large_neg_dx = None
    large_pos_dy = None
    large_neg_dy = None
    
    print("Inspecting 20 samples...\n")
    for i in range(20):
        try:
            sample = ds[i]
        except Exception as e:
            print(f"Sample {i} failed generation: {e}")
            continue
            
        ref = sample['reference_patch'].numpy()[0]
        cand = sample['candidate_patch'].numpy()[0]
        
        c_coord = sample['classical_coordinate'].numpy()
        gt_coord = sample['ground_truth_coordinate'].numpy()
        t_delta = sample['target_delta'].numpy()
        
        dx, dy = t_delta
        mag = np.sqrt(dx**2 + dy**2)
        
        dx_list.append(dx)
        dy_list.append(dy)
        mag_list.append(mag)
        
        print(f"========================================")
        print(f"SAMPLE {i}")
        print(f"========================================")
        print(f"Reference patch shape: {ref.shape}")
        print(f"Candidate patch shape: {cand.shape}")
        print(f"Classical coordinate: {c_coord}")
        print(f"Ground-truth coordinate: {gt_coord}")
        print(f"Target dx: {dx:.4f}")
        print(f"Target dy: {dy:.4f}")
        print(f"Target magnitude: {mag:.4f}\n")
        
        # Track extremes for side-by-side
        if large_pos_dx is None or dx > large_pos_dx['dx']:
            large_pos_dx = {'idx': i, 'dx': dx, 'sample': sample}
        if large_neg_dx is None or dx < large_neg_dx['dx']:
            large_neg_dx = {'idx': i, 'dx': dx, 'sample': sample}
        if large_pos_dy is None or dy > large_pos_dy['dy']:
            large_pos_dy = {'idx': i, 'dy': dy, 'sample': sample}
        if large_neg_dy is None or dy < large_neg_dy['dy']:
            large_neg_dy = {'idx': i, 'dy': dy, 'sample': sample}
            
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(ref, cmap='gray')
        axes[0].set_title("Reference Patch")
        
        axes[1].imshow(cand, cmap='gray')
        axes[1].set_title("Candidate Patch")
        
        # Diagnostic overlay
        axes[2].imshow(cand, cmap='gray', alpha=0.5)
        # Center of patch is 64, 64 (for 128x128 patch)
        center = (64, 64)
        
        # In a perfect world, classical coordinate is at the center of the patch.
        # But target_delta is what we add to classical to get GT.
        # If we visualize the center of candidate as classical...
        axes[2].scatter([center[0]], [center[1]], c='red', label='Classical Center', marker='x')
        axes[2].scatter([center[0] + dx], [center[1] + dy], c='green', label='GT Center', marker='+')
        axes[2].arrow(center[0], center[1], dx, dy, color='yellow', head_width=2, length_includes_head=True)
        
        axes[2].set_title(f"Diagnostic\nTarget dx:{dx:.2f} dy:{dy:.2f}")
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{i:02d}.png"))
        plt.close()

    dx_arr = np.array(dx_list)
    dy_arr = np.array(dy_list)
    mag_arr = np.array(mag_list)
    
    print("========================================")
    print("DATASET STATISTICS (N=20)")
    print("========================================")
    print(f"Mean dx: {np.mean(dx_arr):.4f} | Std dx: {np.std(dx_arr):.4f}")
    print(f"Mean dy: {np.mean(dy_arr):.4f} | Std dy: {np.std(dy_arr):.4f}")
    print(f"Mean magnitude: {np.mean(mag_arr):.4f} | Std magnitude: {np.std(mag_arr):.4f}\n")
    
    # Save extremes
    def save_extreme(s_dict, name):
        if not s_dict: return
        sample = s_dict['sample']
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(sample['reference_patch'].numpy()[0], cmap='gray')
        axes[0].set_title("Reference")
        axes[1].imshow(sample['candidate_patch'].numpy()[0], cmap='gray')
        axes[1].set_title(f"Candidate ({name})")
        plt.savefig(os.path.join(out_dir, f"extreme_{name}.png"))
        plt.close()
        
    save_extreme(large_pos_dx, "large_pos_dx")
    save_extreme(large_neg_dx, "large_neg_dx")
    save_extreme(large_pos_dy, "large_pos_dy")
    save_extreme(large_neg_dy, "large_neg_dy")
    
    print(f"Visualizations saved to {out_dir}")

if __name__ == "__main__":
    main()
