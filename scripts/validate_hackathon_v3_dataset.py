import os
import sys
import json
import csv
import glob

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v3'))
    
    if not os.path.exists(base_dir):
        print(f"FAIL: Directory {base_dir} does not exist.")
        sys.exit(1)
        
    manifest_path = os.path.join(base_dir, 'dataset_manifest.csv')
    if not os.path.exists(manifest_path):
        print(f"FAIL: Manifest {manifest_path} does not exist.")
        sys.exit(1)
        
    cases = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
            
    # 1. Exactly 60 cases exist
    if len(cases) != 60:
        print(f"FAIL: Expected 60 cases, found {len(cases)}.")
        sys.exit(1)
        
    # Counts
    arch_counts = {'DRAM': 0, 'FinFET': 0}
    diff_counts = {'easy': 0, 'moderate': 0, 'hard': 0}
    region_counts = {
        'center': 0, 'interior': 0, 'left_boundary': 0, 'right_boundary': 0,
        'top_boundary': 0, 'bottom_boundary': 0, 'top_left_corner': 0, 'top_right_corner': 0,
        'bottom_left_corner': 0, 'bottom_right_corner': 0, 'random': 0
    }
    
    case_ids = set()
    seeds = set()
    
    for c in cases:
        case_id = c['case_id']
        arch = c['architecture']
        diff = c['difficulty']
        region = c['spatial_region']
        seed = c['seed']
        
        arch_counts[arch] += 1
        diff_counts[diff] += 1
        region_counts[region] += 1
        
        # 19. No duplicate case IDs
        if case_id in case_ids:
            print(f"FAIL: Duplicate case_id found: {case_id}")
            sys.exit(1)
        case_ids.add(case_id)
        
        # 20. Seeds are unique
        if seed in seeds:
            print(f"FAIL: Duplicate seed found: {seed}")
            sys.exit(1)
        seeds.add(seed)
        
        arch_lower = arch.lower()
        case_dir = os.path.join(base_dir, arch_lower, case_id)
        
        # 13, 14, 15. Every case has reference.png, search.png, metadata.json
        ref_path = os.path.join(case_dir, "reference.png")
        search_path = os.path.join(case_dir, "search.png")
        meta_path = os.path.join(case_dir, "metadata.json")
        
        for p in [ref_path, search_path, meta_path]:
            if not os.path.exists(p):
                print(f"FAIL: Missing file {p}")
                sys.exit(1)
                
        with open(meta_path, 'r', encoding='utf-8') as mf:
            meta = json.load(mf)
            
        # 22. No case is accidentally outside its declared spatial region
        if meta['spatial_region'] != region:
            print(f"FAIL: Metadata spatial_region mismatch for {case_id}")
            sys.exit(1)
            
        # 16. Image dimensions are valid (1000x1000 search, 1000x1000 ref)
        if meta['search_width'] != 1000 or meta['search_height'] != 1000:
            print(f"FAIL: Invalid search dimensions for {case_id}")
            sys.exit(1)
            
        if meta['reference_width'] != 1000 or meta['reference_height'] != 1000:
            print(f"FAIL: Invalid reference dimensions for {case_id}")
            sys.exit(1)
            
        # 17. GT coordinates lie inside the search image
        gt_x = meta['gt_x']
        gt_y = meta['gt_y']
        if not (0 <= gt_x <= 1000) or not (0 <= gt_y <= 1000):
            print(f"FAIL: GT coordinates outside bounds for {case_id}")
            sys.exit(1)
            
        # 18. GT bounding boxes are valid
        gt_bbox = meta['gt_bbox']
        if gt_bbox['width'] <= 0 or gt_bbox['height'] <= 0:
            print(f"FAIL: Invalid GT bounding box for {case_id}")
            sys.exit(1)

    # 2. Exactly 30 DRAM cases
    if arch_counts['DRAM'] != 30:
        print(f"FAIL: Expected 30 DRAM cases, got {arch_counts['DRAM']}")
        sys.exit(1)
        
    # 3. Exactly 30 FinFET cases
    if arch_counts['FinFET'] != 30:
        print(f"FAIL: Expected 30 FinFET cases, got {arch_counts['FinFET']}")
        sys.exit(1)
        
    # 4, 5, 6. Exactly 20 easy, moderate, hard
    for diff in ['easy', 'moderate', 'hard']:
        if diff_counts[diff] != 20:
            print(f"FAIL: Expected 20 {diff} cases, got {diff_counts[diff]}")
            sys.exit(1)
            
    # 7. Exactly 10 center cases
    if region_counts['center'] != 10:
        print(f"FAIL: Expected 10 center cases, got {region_counts['center']}")
        sys.exit(1)
        
    # 8. Exactly 15 interior cases
    if region_counts['interior'] != 15:
        print(f"FAIL: Expected 15 interior cases, got {region_counts['interior']}")
        sys.exit(1)
        
    # 9. Exactly 10 left/right boundary cases
    if region_counts['left_boundary'] + region_counts['right_boundary'] != 10:
        print(f"FAIL: Expected 10 L/R boundary cases, got {region_counts['left_boundary']} L and {region_counts['right_boundary']} R")
        sys.exit(1)
        
    # 10. Exactly 10 top/bottom boundary cases
    if region_counts['top_boundary'] + region_counts['bottom_boundary'] != 10:
        print(f"FAIL: Expected 10 T/B boundary cases, got {region_counts['top_boundary']} T and {region_counts['bottom_boundary']} B")
        sys.exit(1)
        
    # 11. Exactly 8 corner cases
    corner_count = sum([region_counts['top_left_corner'], region_counts['top_right_corner'], region_counts['bottom_left_corner'], region_counts['bottom_right_corner']])
    if corner_count != 8:
        print(f"FAIL: Expected 8 corner cases, got {corner_count}")
        sys.exit(1)
        
    # 12. Exactly 7 random cases
    if region_counts['random'] != 7:
        print(f"FAIL: Expected 7 random cases, got {region_counts['random']}")
        sys.exit(1)
        
    print("ALL VALIDATION CHECKS PASSED.")
    print("V3 dataset is fully compliant with requirements.")
    
if __name__ == '__main__':
    main()
