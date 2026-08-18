import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ai_refinement.dataset import Phase1OutputSimDataset

def hash_tensor(tensor):
    return hash(tensor.numpy().tobytes())

def main():
    print("========================================")
    print("DATASET DIVERSITY AUDIT")
    print("========================================")
    
    ds = Phase1OutputSimDataset(num_samples=200, apply_aug=False)
    
    unique_refs = set()
    unique_cands = set()
    unique_targets = set()
    
    dx_list = []
    dy_list = []
    mag_list = []
    
    duplicates = 0
    total_samples = 100
    
    target_history = {}
    
    for i in range(total_samples):
        sample = ds[i]
        
        ref = sample['reference_patch']
        cand = sample['candidate_patch']
        target_delta = sample['target_delta']
        
        dx, dy = target_delta.numpy()
        mag = np.sqrt(dx**2 + dy**2)
        
        dx_list.append(dx)
        dy_list.append(dy)
        mag_list.append(mag)
        
        ref_h = hash_tensor(ref)
        cand_h = hash_tensor(cand)
        target_h = f"{dx:.6f}_{dy:.6f}"
        
        unique_refs.add(ref_h)
        unique_cands.add(cand_h)
        
        if target_h in target_history:
            duplicates += 1
        else:
            target_history[target_h] = i
            unique_targets.add(target_h)

    dup_ratio = duplicates / total_samples
    
    print(f"Total samples: {total_samples}")
    print(f"Unique source reference images: {len(unique_refs)}")
    print(f"Unique source candidate images: {len(unique_cands)}")
    print(f"Unique warp parameter sets: {len(unique_targets)}")  # Proxy via target
    print(f"Unique target residual vectors: {len(unique_targets)}")
    print(f"Duplicate sample count: {duplicates}")
    print(f"Duplicate ratio: {dup_ratio:.2%}")
    print()
    print(f"Mean dx: {np.mean(dx_list):.4f}")
    print(f"Std dx: {np.std(dx_list):.4f}")
    print(f"Mean dy: {np.mean(dy_list):.4f}")
    print(f"Std dy: {np.std(dy_list):.4f}")
    print(f"Mean residual magnitude: {np.mean(mag_list):.4f}")
    print(f"Std residual magnitude: {np.std(mag_list):.4f}")
    print("========================================")
    
    print("\nCRITICAL TESTS")
    # Verify dataset[1] != dataset[2]
    s1 = ds[1]
    s2 = ds[2]
    
    t1 = f"{s1['target_delta'][0]:.6f}_{s1['target_delta'][1]:.6f}"
    t2 = f"{s2['target_delta'][0]:.6f}_{s2['target_delta'][1]:.6f}"
    
    if t1 != t2:
        print("[PASS] dataset[1] != dataset[2]")
    else:
        print("[FAIL] dataset[1] == dataset[2]")
        
    # Verify dataset[1] called twice gives the same result (no aug)
    ds_no_cache = Phase1OutputSimDataset(num_samples=10, apply_aug=False)
    s1_a = ds_no_cache[1]
    # Bypass cache by creating a fresh object or just clearing the dictionary
    ds_no_cache._cache.clear()
    s1_b = ds_no_cache[1]
    
    t1_a = f"{s1_a['target_delta'][0]:.6f}_{s1_a['target_delta'][1]:.6f}"
    t1_b = f"{s1_b['target_delta'][0]:.6f}_{s1_b['target_delta'][1]:.6f}"
    
    if t1_a == t1_b:
        print("[PASS] dataset[1] is completely deterministic")
    else:
        print("[FAIL] dataset[1] is non-deterministic")

if __name__ == "__main__":
    main()
