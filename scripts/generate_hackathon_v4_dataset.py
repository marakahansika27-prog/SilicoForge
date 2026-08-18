import os
import sys
import json
import csv
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.generator import HackathonDatasetGenerator

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    print("Generating Hackathon V4 Dataset...")
    
    v3_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v3'))
    v4_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v4'))
    ensure_dir(v4_base_dir)
    
    v3_manifest_path = os.path.join(v3_base_dir, 'dataset_manifest.csv')
    v4_manifest_path = os.path.join(v4_base_dir, 'dataset_manifest.csv')
    
    cases = []
    with open(v3_manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        v3_headers = reader.fieldnames
        for row in reader:
            cases.append(row)
            
    v4_headers = v3_headers.copy()
    v4_headers.extend([
        'global_context_path',
        'global_mapping_scale',
        'global_mapping_offset_x',
        'global_mapping_offset_y'
    ])
    
    manifest_rows = []
    
    dram_count = 0
    finfet_count = 0
    
    for idx, c in enumerate(cases):
        case_id = c['case_id']
        arch = c['architecture']
        diff = c['difficulty']
        seed = int(c['seed'])
        region = c['spatial_region']
        
        if arch == "DRAM": dram_count += 1
        if arch == "FinFET": finfet_count += 1
        
        print(f"[{idx+1}/60] Generating V4 {case_id} ({arch}, {diff}, {region})...")
        
        # 1. Generate underlying images deterministically
        gen = HackathonDatasetGenerator(seed=seed)
        ref_img, search_img, meta = gen.generate_case(case_id, arch, diff, version="v3", spatial_region=region)
        
        # In the existing V3 generator, search_img is already a 10x downscale of the full 10000x10000 base_img.
        # It contains the complete active array and its macroscopic boundaries.
        # We define global_context.png as exactly this global view.
        global_context_img = search_img.copy()
        
        arch_lower = arch.lower()
        case_dir = os.path.join(v4_base_dir, arch_lower, case_id)
        ensure_dir(case_dir)
        
        ref_path = os.path.join(case_dir, "reference.png")
        search_path = os.path.join(case_dir, "search.png")
        global_path = os.path.join(case_dir, "global_context.png")
        meta_path = os.path.join(case_dir, "metadata.json")
        
        cv2.imwrite(ref_path, ref_img)
        cv2.imwrite(search_path, search_img)
        cv2.imwrite(global_path, global_context_img)
        
        meta_dict = meta.to_dict()
        meta_dict['global_context_path'] = f"{arch_lower}/{case_id}/global_context.png"
        meta_dict['global_mapping_scale'] = 1.0
        meta_dict['global_mapping_offset_x'] = 0.0
        meta_dict['global_mapping_offset_y'] = 0.0
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_dict, f, indent=4)
            
        row = c.copy()
        # Update paths to V4
        row['reference_path'] = f"{arch_lower}/{case_id}/reference.png"
        row['search_path'] = f"{arch_lower}/{case_id}/search.png"
        row['global_context_path'] = f"{arch_lower}/{case_id}/global_context.png"
        row['global_mapping_scale'] = 1.0
        row['global_mapping_offset_x'] = 0.0
        row['global_mapping_offset_y'] = 0.0
        
        manifest_rows.append(row)
        
    with open(v4_manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=v4_headers)
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow(r)
            
    print("\n--- V4 DATASET VALIDATION ---")
    print(f"Total Cases: {len(manifest_rows)} (Expected: 60)")
    print(f"DRAM Cases: {dram_count} (Expected: 30)")
    print(f"FinFET Cases: {finfet_count} (Expected: 30)")
    
    validation_passed = True
    if len(manifest_rows) != 60: validation_passed = False
    if dram_count != 30: validation_passed = False
    if finfet_count != 30: validation_passed = False
    
    for row in manifest_rows:
        case_id = row['case_id']
        arch = row['architecture'].lower()
        c_dir = os.path.join(v4_base_dir, arch, case_id)
        
        if not os.path.exists(os.path.join(c_dir, 'reference.png')): validation_passed = False
        if not os.path.exists(os.path.join(c_dir, 'search.png')): validation_passed = False
        if not os.path.exists(os.path.join(c_dir, 'global_context.png')): validation_passed = False
        if not os.path.exists(os.path.join(c_dir, 'metadata.json')): validation_passed = False
        
        with open(os.path.join(c_dir, 'metadata.json'), 'r') as f:
            m = json.load(f)
            if 'global_mapping_scale' not in m: validation_passed = False
            
    print(f"Validation Passed: {validation_passed}")
    if validation_passed:
        print("V4 Generator completed successfully. Ready for py_compile.")
        
if __name__ == '__main__':
    main()
