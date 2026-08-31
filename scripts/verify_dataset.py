import os
import sys
import argparse
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ai_refinement.dataset import Phase1OutputSimDataset
from src.ai_refinement.phase2_dataset import Phase2EvaluationDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/hackathon_v3",
                        help="Path to the real Phase 2 generated dataset.")
    return parser.parse_args()

def safe_pearsonr(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return pearsonr(a, b)[0]

def run_synthetic_test():
    print("\n========================================")
    print("A) SYNTHETIC PHASE 1 IDENTIFIABILITY TEST")
    print("========================================")
    
    dataset_size = 100
    ds = Phase1OutputSimDataset(num_samples=dataset_size, apply_aug=False)
    
    samples = [ds[i] for i in range(dataset_size)]
    ref_patches = torch.stack([s['reference_patch'] for s in samples]).numpy()
    cand_patches = torch.stack([s['candidate_patch'] for s in samples]).numpy()
    target_deltas = torch.stack([s['target_delta'] for s in samples]).numpy()
    
    t_dx = target_deltas[:, 0]
    t_dy = target_deltas[:, 1]
    
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
    
    r_x = safe_pearsonr(rel_x_center, t_dx)
    r_y = safe_pearsonr(rel_y_center, t_dy)
    
    print(f"Correlation (Image Diff X-Centroid <-> Target dx): {r_x:.4f}")
    print(f"Correlation (Image Diff Y-Centroid <-> Target dy): {r_y:.4f}")

def run_phase2_integrity_test(data_dir):
    print("\n========================================")
    print("B) REAL PHASE 2 DATASET INTEGRITY TEST")
    print("========================================")
    print(f"Data Dir: {data_dir}\n")
    
    ds = Phase2EvaluationDataset(dataset_dir=data_dir)
    print(f"Total cases: {len(ds)}")
    print(f"Positive cases: {ds.positive_count}")
    print(f"Negative cases: {ds.negative_count}")
    
    unique_gt = set()
    gt_xs, gt_ys = [], []
    scales, rots = [], []
    
    integrity_pass = True
    
    for i in range(len(ds)):
        sample = ds[i]
        ci = sample['case_info']
        ref_img = sample['reference_img']
        search_img = sample['search_img']
        
        # Verify files and dimensions
        if ref_img is None or search_img is None:
            print(f"FAIL: Missing image in {ci['case_dir']}")
            integrity_pass = False
            continue
            
        if ref_img.shape != (1000, 1000) or search_img.shape != (1000, 1000):
            print(f"FAIL: Invalid dims in {ci['case_dir']} (ref:{ref_img.shape}, search:{search_img.shape})")
            integrity_pass = False
            
        if ci['case_id'] not in ci['case_dir']:
            print(f"FAIL: Case ID {ci['case_id']} mismatches directory {ci['case_dir']}")
            integrity_pass = False
            
        scales.append(ci['phase2_scale'])
        rots.append(ci['rotation_degrees'])
        
        if ci['target_present']:
            if ci['gt_x'] < 0 or ci['gt_y'] < 0:
                print(f"FAIL: Positive case {ci['case_id']} has invalid GT ({ci['gt_x']}, {ci['gt_y']})")
                integrity_pass = False
            else:
                unique_gt.add((ci['gt_x'], ci['gt_y']))
                gt_xs.append(ci['gt_x'])
                gt_ys.append(ci['gt_y'])
        else:
            if ci['gt_x'] != -1.0 or ci['gt_y'] != -1.0:
                print(f"FAIL: Negative case {ci['case_id']} has invalid GT ({ci['gt_x']}, {ci['gt_y']})")
                integrity_pass = False
                
    if len(gt_xs) > 0:
        print(f"Unique GT coordinates: {len(unique_gt)}")
        print(f"GT X range: {min(gt_xs):.2f} to {max(gt_xs):.2f}")
        print(f"GT Y range: {min(gt_ys):.2f} to {max(gt_ys):.2f}")
    if len(scales) > 0:
        print(f"Scale range: {min(scales):.2f} to {max(scales):.2f}")
        print(f"Rotation range: {min(rots):.2f} to {max(rots):.2f}")
        
    print("\nINTEGRITY STATUS: " + ("PASS" if integrity_pass else "FAIL"))
    return integrity_pass

def main():
    args = get_args()
    run_synthetic_test()
    pass_phase2 = run_phase2_integrity_test(args.data_dir)
    
    if pass_phase2:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
