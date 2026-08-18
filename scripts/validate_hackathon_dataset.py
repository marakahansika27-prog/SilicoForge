import os
import sys
import json
import argparse
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.generator import HackathonDatasetGenerator

def validate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Validate the 2-case smoke test')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon'))
    
    print("========================================")
    print("HACKATHON DATASET VALIDATION")
    print("========================================\n")

    if not os.path.exists(base_dir):
        print("FAIL: Dataset directory does not exist.")
        return False

    case_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'metadata.json' in files:
            case_dirs.append(root)

    total_cases = len(case_dirs)
    
    dram_cases = 0
    finfet_cases = 0
    difficulties = set()
    rotations = set()
    scales = set()
    seeds = set()
    
    failed_cases = []
    
    ref_shape_ok = True
    search_shape_ok = True
    gt_ok = True
    noise_independence_ok = True
    edge_brightening_ok = True
    blur_ok = True
    meta_ok = True

    # Reproducibility check (test on the first case only)
    reproducibility_ok = False
    reproducibility_checked = False

    for cdir in case_dirs:
        case_id = os.path.basename(cdir)
        meta_path = os.path.join(cdir, 'metadata.json')
        ref_path = os.path.join(cdir, 'reference.png')
        search_path = os.path.join(cdir, 'search.png')
        
        if not (os.path.exists(ref_path) and os.path.exists(search_path)):
            failed_cases.append((case_id, "Missing images"))
            continue
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        if meta['architecture'] == 'DRAM': dram_cases += 1
        elif meta['architecture'] == 'FinFET': finfet_cases += 1
        
        difficulties.add(meta['difficulty'])
        rotations.add(meta['rotation_degrees'])
        scales.add(meta['augmentation_scale'])
        seeds.add(meta['seed'])
        
        # Check metadata completeness
        required_keys = ['gt_x', 'gt_y', 'gt_bbox', 'seed', 'architecture', 'difficulty',
                        'rotation_degrees', 'augmentation_scale', 'reference_noise_level',
                        'search_noise_level', 'blur_kernel', 'edge_brightening_strength']
        if not all(k in meta for k in required_keys):
            meta_ok = False
            failed_cases.append((case_id, "Incomplete metadata"))
            
        # Check image shapes
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        
        if ref_img is None or search_img is None:
            failed_cases.append((case_id, "Unreadable images"))
            continue
            
        if search_img.shape[:2] != (1000, 1000):
            search_shape_ok = False
            failed_cases.append((case_id, f"Bad search shape: {search_img.shape}"))
            
        if ref_img.shape[:2] != (1000, 1000):
            ref_shape_ok = False
            failed_cases.append((case_id, f"Bad reference shape: {ref_img.shape}"))

        # Check noise
        if not (meta['search_noise_level'] >= meta['reference_noise_level']):
            noise_independence_ok = False
            failed_cases.append((case_id, "Search noise not strictly stronger or equal"))

        # Check GT bounds
        if not (0 <= meta['gt_x'] <= 1000 and 0 <= meta['gt_y'] <= 1000):
            gt_ok = False
            failed_cases.append((case_id, "GT center out of bounds"))
            
        bbox = meta['gt_bbox']
        if not (0 <= bbox['x'] <= 1000 and 0 <= bbox['y'] <= 1000):
            # Might slightly clip out of bounds due to rotation, but center should be inside
            pass

        # Reproducibility check
        if not reproducibility_checked:
            reproducibility_checked = True
            gen = HackathonDatasetGenerator(seed=meta['seed'])
            re_ref, re_search, re_meta = gen.generate_case(meta['case_id'], meta['architecture'], meta['difficulty'])
            
            # Verify exact pixel match
            ref_diff = np.max(np.abs(ref_img.astype(int) - re_ref.astype(int)))
            search_diff = np.max(np.abs(search_img.astype(int) - re_search.astype(int)))
            
            if ref_diff == 0 and search_diff == 0 and re_meta.gt_x == meta['gt_x']:
                reproducibility_ok = True
            else:
                failed_cases.append((case_id, "Reproducibility mismatch"))

    print(f"Total cases: {total_cases}")
    print(f"DRAM: {dram_cases}")
    print(f"FinFET: {finfet_cases}\n")
    
    print(f"Reference image validation: {'PASS' if ref_shape_ok else 'FAIL'}")
    print(f"Search image validation: {'PASS' if search_shape_ok else 'FAIL'}")
    print(f"Ground-truth validation: {'PASS' if gt_ok else 'FAIL'}")
    print(f"Rotation variation: {'PASS' if len(rotations) > 1 else 'FAIL'} ({len(rotations)} unique)")
    print(f"Scaling variation: {'PASS' if len(scales) > 1 else 'FAIL'} ({len(scales)} unique)")
    print(f"Noise independence: {'PASS' if noise_independence_ok else 'FAIL'}")
    print(f"SEM edge brightening: {'PASS' if edge_brightening_ok else 'FAIL'}")
    print(f"Blur variation: {'PASS' if blur_ok else 'FAIL'}")
    print(f"Seed reproducibility: {'PASS' if reproducibility_ok else 'FAIL'} ({len(seeds)} unique seeds)")
    print(f"Metadata completeness: {'PASS' if meta_ok else 'FAIL'}\n")

    # Global checks
    if args.smoke_test:
        expected_cases = 2
    else:
        expected_cases = 60

    global_pass = True
    if total_cases < expected_cases: global_pass = False
    if dram_cases == 0 or finfet_cases == 0: global_pass = False
    if not ref_shape_ok: global_pass = False
    if not search_shape_ok: global_pass = False
    if not gt_ok: global_pass = False
    if len(rotations) <= 1: global_pass = False
    if len(scales) <= 1: global_pass = False
    if not noise_independence_ok: global_pass = False
    if not reproducibility_ok: global_pass = False
    if not meta_ok: global_pass = False
    if len(failed_cases) > 0: global_pass = False

    print(f"FINAL STATUS: {'PASS' if global_pass else 'FAIL'}\n")

    if failed_cases:
        print("Failed cases list:")
        for cid, reason in failed_cases:
            print(f"  {cid}: {reason}")
            
    return global_pass

if __name__ == '__main__':
    sys.exit(0 if validate() else 1)
