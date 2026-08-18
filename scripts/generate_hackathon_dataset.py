import os
import sys
import argparse
import csv
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.generator import HackathonDatasetGenerator

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Run a 2-case smoke test')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon'))
    ensure_dir(base_dir)

    manifest_path = os.path.join(base_dir, 'dataset_manifest.csv')
    manifest_headers = [
        'case_id', 'architecture', 'difficulty', 'seed', 'reference_path', 'search_path',
        'gt_x', 'gt_y', 'gt_bbox_x', 'gt_bbox_y', 'gt_bbox_width', 'gt_bbox_height',
        'rotation_degrees', 'scale_factor', 'blur_kernel', 'reference_noise_level',
        'search_noise_level', 'edge_brightening_strength'
    ]

    manifest_rows = []

    if args.smoke_test:
        print("Running smoke test mode (2 cases)...")
        cases = [
            ("case_0001", "DRAM", "moderate"),
            ("case_0002", "FinFET", "hard")
        ]
    else:
        print("Running full dataset generation (60 cases)...")
        cases = []
        for i in range(1, 31):
            diff = ["easy", "moderate", "hard"][i % 3]
            cases.append((f"case_{i:04d}", "DRAM", diff))
        for i in range(31, 61):
            diff = ["easy", "moderate", "hard"][i % 3]
            cases.append((f"case_{i:04d}", "FinFET", diff))

    base_seed = 42000

    for idx, (case_id, arch, diff) in enumerate(cases):
        print(f"Generating {case_id} ({arch}, {diff})...")
        seed = base_seed + idx
        
        gen = HackathonDatasetGenerator(seed=seed)
        ref_img, search_img, meta = gen.generate_case(case_id, arch, diff)
        
        arch_lower = arch.lower()
        case_dir = os.path.join(base_dir, arch_lower, case_id)
        ensure_dir(case_dir)
        
        ref_path = os.path.join(case_dir, "reference.png")
        search_path = os.path.join(case_dir, "search.png")
        meta_path = os.path.join(case_dir, "metadata.json")
        
        cv2.imwrite(ref_path, ref_img)
        cv2.imwrite(search_path, search_img)
        meta.save_json(meta_path)
        
        # Relative paths for manifest
        rel_ref_path = f"{arch_lower}/{case_id}/reference.png"
        rel_search_path = f"{arch_lower}/{case_id}/search.png"
        
        manifest_rows.append({
            'case_id': meta.case_id,
            'architecture': meta.architecture,
            'difficulty': meta.difficulty,
            'seed': meta.seed,
            'reference_path': rel_ref_path,
            'search_path': rel_search_path,
            'gt_x': meta.gt_x,
            'gt_y': meta.gt_y,
            'gt_bbox_x': meta.gt_bbox['x'],
            'gt_bbox_y': meta.gt_bbox['y'],
            'gt_bbox_width': meta.gt_bbox['width'],
            'gt_bbox_height': meta.gt_bbox['height'],
            'rotation_degrees': meta.rotation_degrees,
            'scale_factor': meta.augmentation_scale,
            'blur_kernel': meta.blur_kernel,
            'reference_noise_level': meta.reference_noise_level,
            'search_noise_level': meta.search_noise_level,
            'edge_brightening_strength': meta.edge_brightening_strength
        })

    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=manifest_headers)
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow(r)

    print(f"Generated {len(cases)} cases successfully.")
    print(f"Manifest written to: {manifest_path}")

if __name__ == '__main__':
    main()
