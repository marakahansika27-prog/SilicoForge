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
    print("INPUT-TARGET IDENTIFIABILITY AUDIT")
    print("========================================")
    
    # 1. Collect Dataset Samples
    dataset_size = 1000
    print(f"Loading Phase 2 Dataset...")
    ds = Phase1OutputSimDataset(num_samples=dataset_size, apply_aug=True)
    
    num_audit = 100
    samples = []
    
    for i in range(num_audit):
        sample = ds[i]
        # sample contains 'reference_patch', 'candidate_patch', 'target_delta', 
        # 'classical_coordinate', 'ground_truth_coordinate', etc.
        samples.append(sample)
        
    print(f"Collected {num_audit} deterministic samples.")
    
    # Extract tensors for vectorization
    ref_patches = torch.stack([s['reference_patch'] for s in samples]).numpy() # (N, 1, 128, 128)
    cand_patches = torch.stack([s['candidate_patch'] for s in samples]).numpy()
    target_deltas = torch.stack([s['target_delta'] for s in samples]).numpy()
    class_coords = torch.stack([s['classical_coordinate'] for s in samples]).numpy()
    gt_coords = torch.stack([s['ground_truth_coordinate'] for s in samples]).numpy()
    
    # 2. Input Diversity
    print("\n========================================")
    print("STEP 2 - INPUT DIVERSITY")
    print("========================================")
    print(f"Reference Patch: Mean = {ref_patches.mean():.4f}, Std = {ref_patches.std():.4f}")
    print(f"Candidate Patch: Mean = {cand_patches.mean():.4f}, Std = {cand_patches.std():.4f}")
    
    pairwise_diffs = []
    
    # Compute pairwise differences
    for i in range(num_audit):
        for j in range(i + 1, num_audit):
            ref_diff = np.mean(np.abs(ref_patches[i] - ref_patches[j]))
            cand_diff = np.mean(np.abs(cand_patches[i] - cand_patches[j]))
            total_diff = ref_diff + cand_diff
            
            target_diff = np.linalg.norm(target_deltas[i] - target_deltas[j])
            pairwise_diffs.append((total_diff, target_diff, i, j, ref_diff, cand_diff))
            
    # Sort by total image difference ascending
    pairwise_diffs.sort(key=lambda x: x[0])
    
    print("\n10 Most Similar Sample Pairs (Image Difference):")
    identifiability_fail = False
    
    for rank in range(10):
        total_diff, target_diff, i, j, ref_diff, cand_diff = pairwise_diffs[rank]
        print(f"Rank {rank+1}: Sample {i} vs Sample {j}")
        print(f"  Target A: {target_deltas[i]}, Target B: {target_deltas[j]} (Diff: {target_diff:.4f} px)")
        print(f"  Ref Diff: {ref_diff:.6f}, Cand Diff: {cand_diff:.6f}")
        
        # Thresholds for identically looking images but totally different targets
        if total_diff < 0.01 and target_diff > 1.0:
            identifiability_fail = True
            
    if identifiability_fail:
        print("\n[FAIL] INPUT-TARGET IDENTIFIABILITY")
        print("Images are nearly identical but targets differ substantially.")
    else:
        print("\n[PASS] Input differences naturally correspond to target variations.")

    # 3. Target vs Image Difference
    print("\n========================================")
    print("STEP 3 - TARGET VS IMAGE DIFFERENCE")
    print("========================================")
    
    img_diffs = cand_patches - ref_patches # (N, 1, 128, 128)
    
    # Calculate difference energy and centroid
    diff_energy = np.mean(np.abs(img_diffs), axis=(1, 2, 3))
    
    # Centroid calculation for difference map
    # We take the absolute difference
    abs_diffs = np.abs(img_diffs[:, 0, :, :]) # (N, 128, 128)
    
    x_indices = np.arange(128)
    y_indices = np.arange(128)
    X, Y = np.meshgrid(x_indices, y_indices)
    
    diff_x_center = np.zeros(num_audit)
    diff_y_center = np.zeros(num_audit)
    
    for i in range(num_audit):
        sum_diff = np.sum(abs_diffs[i])
        if sum_diff > 1e-6:
            diff_x_center[i] = np.sum(abs_diffs[i] * X) / sum_diff
            diff_y_center[i] = np.sum(abs_diffs[i] * Y) / sum_diff
        else:
            diff_x_center[i] = 64.0
            diff_y_center[i] = 64.0
            
    target_dx = target_deltas[:, 0]
    target_dy = target_deltas[:, 1]
    target_mag = np.linalg.norm(target_deltas, axis=1)
    
    # Shift centroids to be relative to center (64)
    rel_x_center = diff_x_center - 64.0
    rel_y_center = diff_y_center - 64.0
    
    def safe_pearsonr(a, b):
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return 0.0
        return pearsonr(a, b)[0]
        
    r_x = safe_pearsonr(rel_x_center, target_dx)
    r_y = safe_pearsonr(rel_y_center, target_dy)
    r_mag = safe_pearsonr(diff_energy, target_mag)
    
    print(f"Correlation (Image Diff X-Centroid ↔ Target dx): {r_x:.4f}")
    print(f"Correlation (Image Diff Y-Centroid ↔ Target dy): {r_y:.4f}")
    print(f"Correlation (Image Diff Energy ↔ Target Magnitude): {r_mag:.4f}")
    
    # 4. Visualize Samples
    print("\n========================================")
    print("STEP 4 - VISUALIZE SAMPLES")
    print("========================================")
    
    os.makedirs("outputs/debug", exist_ok=True)
    
    fig, axes = plt.subplots(4, 6, figsize=(18, 12)) # 12 samples, 2 columns per sample (ref/cand)
    fig.suptitle("Input Tensors vs Target Residuals", fontsize=16)
    
    for i in range(12):
        row = i // 3
        col = (i % 3) * 2
        
        ax_ref = axes[row, col]
        ax_cand = axes[row, col+1]
        
        ax_ref.imshow(ref_patches[i, 0], cmap='gray')
        ax_ref.set_title(f"Sample {i} Ref")
        ax_ref.axis('off')
        
        ax_cand.imshow(cand_patches[i, 0], cmap='gray')
        ax_cand.set_title(f"Target: ({target_dx[i]:.2f}, {target_dy[i]:.2f})")
        ax_cand.axis('off')
        
    plt.tight_layout()
    plt.savefig("outputs/debug/input_target_grid.png")
    plt.close()
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Absolute Difference (Reference vs Candidate)", fontsize=16)
    
    for i in range(12):
        row = i // 4
        col = i % 4
        
        ax_diff = axes[row, col]
        ax_diff.imshow(abs_diffs[i], cmap='hot')
        ax_diff.set_title(f"Sample {i} Diff\nMag: {target_mag[i]:.2f}")
        ax_diff.axis('off')
        
    plt.tight_layout()
    plt.savefig("outputs/debug/input_difference_grid.png")
    plt.close()
    
    # 5. Check Target Construction
    print("\n========================================")
    print("STEP 5 - CHECK TARGET CONSTRUCTION")
    print("========================================")
    
    max_eq_error = 0.0
    for i in range(20):
        c_c = class_coords[i]
        gt_c = gt_coords[i]
        t_d = target_deltas[i]
        
        expected_td = gt_c - c_c
        eq_error = np.max(np.abs(expected_td - t_d))
        max_eq_error = max(max_eq_error, eq_error)
        
        if i < 5:
            print(f"Sample {i}:")
            print(f"  Class Coord: {c_c}")
            print(f"  GT Coord   : {gt_c}")
            print(f"  Target     : {t_d}")
            print(f"  Expected   : {expected_td}")
            
    print(f"\nMaximum equation error (gt - class == target): {max_eq_error:.6f}")
    
    # 6. Check Warp Visibility
    print("\n========================================")
    print("STEP 6 - CHECK WARP VISIBILITY")
    print("========================================")
    print("Synthetic warp parameters (tx, ty, angle) are intentionally strictly encapsulated")
    print("inside dataset._generate_sample() and are not explicitly cached.")
    print("Skipping exact numerical comparison. Visual correlation handled in Step 3.")
    
    # FINAL CLASSIFICATION
    print("\n========================================")
    print("FINAL CLASSIFICATION")
    print("========================================")
    
    if max_eq_error > 1e-4:
        print("CASE D:\nTarget construction equation is inconsistent.\n=> Dataset target generation bug.")
    elif identifiability_fail:
        print("CASE A:\nInputs are nearly identical while target residuals vary substantially.\n=> Dataset target is not identifiable from model inputs.")
    elif abs(r_x) < 0.2 and abs(r_y) < 0.2 and abs(r_mag) < 0.2:
        print("CASE B:\nInputs vary substantially but input variation has weak/no relationship to target residual.\n=> Synthetic warp / target construction is not producing a learnable visual signal.")
    elif abs(r_x) > 0.4 or abs(r_y) > 0.4:
        print("CASE C:\nInputs clearly encode the residual and correlate with target dx/dy.\n=> Dataset is learnable; investigate model optimization/loss next.")
    else:
        print("Classification unclear. Manual review required.")

if __name__ == "__main__":
    main()
