import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ai_refinement.dataset import Phase1OutputSimDataset

def main():
    print("========================================")
    print("NEW IDENTIFIABILITY TEST")
    print("========================================")
    
    dataset_size = 100
    print(f"Loading Phase 2 Dataset (N={dataset_size})...")
    ds = Phase1OutputSimDataset(num_samples=dataset_size, apply_aug=False) # Disable aug to isolate pure geometry
    
    samples = []
    for i in range(dataset_size):
        samples.append(ds[i])
        
    ref_patches = torch.stack([s['reference_patch'] for s in samples]).numpy() # (N, 1, 128, 128)
    cand_patches = torch.stack([s['candidate_patch'] for s in samples]).numpy()
    target_deltas = torch.stack([s['target_delta'] for s in samples]).numpy()
    
    t_dx = target_deltas[:, 0]
    t_dy = target_deltas[:, 1]
    
    print(f"Target dx: Mean = {t_dx.mean():.4f}, Std = {t_dx.std():.4f}")
    print(f"Target dy: Mean = {t_dy.mean():.4f}, Std = {t_dy.std():.4f}")
    
    cand_std_across = np.std(cand_patches, axis=0).mean()
    print(f"Candidate patch standard deviation across dataset: {cand_std_across:.6f}")
    
    pairwise_diffs = []
    
    print("Computing pairwise candidate differences...")
    for i in range(dataset_size):
        for j in range(i + 1, dataset_size):
            cand_diff = np.mean(np.abs(cand_patches[i] - cand_patches[j]))
            target_diff = np.linalg.norm(target_deltas[i] - target_deltas[j])
            pairwise_diffs.append((cand_diff, target_diff, i, j))
            
    pairwise_diffs.sort(key=lambda x: x[0])
    
    visual_threshold = 0.02
    target_threshold = 1.0
    
    print("\n10 Most Visually Similar Candidate Pairs:")
    identifiability_broken = False
    
    for rank in range(10):
        cand_diff, target_diff, i, j = pairwise_diffs[rank]
        print(f"Rank {rank+1}: Sample {i} vs Sample {j}")
        print(f"  Target Diff: {target_diff:.4f} px")
        print(f"  Candidate Diff: {cand_diff:.6f}")
        
        if cand_diff < visual_threshold and target_diff > target_threshold:
            identifiability_broken = True
            
    # Calculate Correlation
    img_diffs = cand_patches - ref_patches
    abs_diffs = np.abs(img_diffs[:, 0, :, :])
    
    x_indices = np.arange(128)
    y_indices = np.arange(128)
    X, Y = np.meshgrid(x_indices, y_indices)
    
    diff_x_center = np.zeros(dataset_size)
    diff_y_center = np.zeros(dataset_size)
    
    for i in range(dataset_size):
        sum_diff = np.sum(abs_diffs[i])
        if sum_diff > 1e-6:
            diff_x_center[i] = np.sum(abs_diffs[i] * X) / sum_diff
            diff_y_center[i] = np.sum(abs_diffs[i] * Y) / sum_diff
        else:
            diff_x_center[i] = 64.0
            diff_y_center[i] = 64.0
            
    rel_x_center = diff_x_center - 64.0
    rel_y_center = diff_y_center - 64.0
    
    def safe_pearsonr(a, b):
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return 0.0
        return pearsonr(a, b)[0]
        
    r_x = safe_pearsonr(rel_x_center, t_dx)
    r_y = safe_pearsonr(rel_y_center, t_dy)
    
    print(f"\nCorrelation (Image Diff X-Centroid ↔ Target dx): {r_x:.4f}")
    print(f"Correlation (Image Diff Y-Centroid ↔ Target dy): {r_y:.4f}")
    
    # VISUAL VERIFICATION
    os.makedirs("outputs/debug", exist_ok=True)
    
    num_vis = min(12, dataset_size)
    fig, axes = plt.subplots(num_vis, 3, figsize=(10, 4 * num_vis))
    fig.suptitle("Input vs Target Visually Identifiable Features", fontsize=16)
    
    for i in range(num_vis):
        ax_ref = axes[i, 0]
        ax_cand = axes[i, 1]
        ax_diff = axes[i, 2]
        
        ax_ref.imshow(ref_patches[i, 0], cmap='gray')
        ax_ref.set_title(f"Sample {i} Reference")
        ax_ref.axis('off')
        
        ax_cand.imshow(cand_patches[i, 0], cmap='gray')
        ax_cand.set_title(f"Sample {i} Candidate")
        ax_cand.axis('off')
        
        ax_diff.imshow(abs_diffs[i], cmap='hot')
        ax_diff.set_title(f"Abs Diff\ntarget_delta = ({t_dx[i]:.2f}, {t_dy[i]:.2f})")
        ax_diff.axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("outputs/debug/input_target_grid.png")
    plt.close()
    
    print("\n========================================")
    print("SUCCESS CRITERIA")
    print("========================================")
    if not identifiability_broken:
        print("[PASS] INPUT-TARGET IDENTIFIABILITY RESTORED")
    else:
        print("[FAIL] INPUT-TARGET IDENTIFIABILITY STILL BROKEN")

if __name__ == "__main__":
    main()
