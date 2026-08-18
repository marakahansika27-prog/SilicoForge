import os
import sys
import argparse
import csv
import cv2

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dataset.generator import HackathonDatasetGenerator

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser(description="Drift-Sense V2 Standalone Dataset Generator")
    parser.add_argument('--architecture', required=True, choices=['DRAM', 'FinFET'], help="Architecture to generate")
    parser.add_argument('--num-pairs', type=int, required=True, help="Number of pairs to generate")
    parser.add_argument('--output-dir', required=True, help="Output directory")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    arch_lower = args.architecture.lower()
    arch_dir = os.path.join(args.output_dir, arch_lower)
    ensure_dir(arch_dir)

    base_seed = 50000

    print(f"Generating {args.num_pairs} {args.architecture} pairs to {args.output_dir}...")

    for i in range(args.num_pairs):
        case_id = f"pair_{i+1:04d}"
        seed = base_seed + i
        
        diff = ["moderate", "hard", "easy"][i % 3]
        
        gen = HackathonDatasetGenerator(seed=seed)
        ref_img, search_img, meta = gen.generate_case(case_id, args.architecture, diff)
        
        case_dir = os.path.join(arch_dir, case_id)
        ensure_dir(case_dir)
        
        ref_path = os.path.join(case_dir, "reference.png")
        search_path = os.path.join(case_dir, "search.png")
        meta_path = os.path.join(case_dir, "ground_truth.json")
        
        cv2.imwrite(ref_path, ref_img)
        cv2.imwrite(search_path, search_img)
        
        # Save explicit GT json
        meta.save_json(meta_path)
        
        print(f"Generated {case_id}: GT=({meta.gt_x:.2f}, {meta.gt_y:.2f})")

    print(f"\nSuccessfully generated {args.num_pairs} pairs.")

if __name__ == '__main__':
    main()
