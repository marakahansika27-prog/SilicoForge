import os
import sys
import json
import csv
import cv2
import random
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.generator import HackathonDatasetGenerator

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    print("Generating Hackathon V4 Dataset (200 Cases)...")
    
    v4_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v4'))
    
    # Clean incomplete dataset
    if os.path.exists(v4_base_dir):
        shutil.rmtree(v4_base_dir)
    ensure_dir(v4_base_dir)
    
    v4_manifest_path = os.path.join(v4_base_dir, 'dataset_manifest.csv')
    
    headers = [
        'case_id', 'architecture', 'difficulty', 'seed', 'spatial_region',
        'target_present', 'reference_path', 'search_path', 'global_context_path',
        'global_mapping_scale', 'global_mapping_offset_x', 'global_mapping_offset_y'
    ]
    
    manifest_rows = []
    
    quotas = {
        ('DRAM', False): 50,    # DRAM, PRESENT
        ('DRAM', True): 50,     # DRAM, ABSENT
        ('FinFET', False): 50,  # FinFET, PRESENT
        ('FinFET', True): 50    # FinFET, ABSENT
    }
    
    failed_attempts = 0
    retry_attempts = 0
    periodic_rejections = 0
    other_rejections = 0
    
    case_idx = 0
    
    for (arch, is_absent), quota in quotas.items():
        count = 0
        while count < quota:
            case_id = f"case_v4_{arch.lower()}_{'absent' if is_absent else 'present'}_{count:03d}"
            difficulty = random.choice(["easy", "moderate", "hard"])
            region = random.choice(["center", "interior", "left_boundary", "right_boundary", "top_boundary", "bottom_boundary", "top_left_corner", "top_right_corner", "bottom_left_corner", "bottom_right_corner", "random"])
            seed = random.randint(10000, 99999)
            
            print(f"[{case_idx+1}/200] Attempting {case_id} (Seed: {seed})...")
            
            gen = HackathonDatasetGenerator(seed=seed)
            
            try:
                ref_img, search_img, meta = gen.generate_case(
                    case_id=case_id, 
                    architecture=arch, 
                    difficulty=difficulty, 
                    version="v4", 
                    spatial_region=region,
                    is_absent=is_absent
                )
                
                # Success
                periodic_rejections += gen.stats.get('periodic_rejections', 0)
                other_rejections += gen.stats.get('other_rejections', 0)
                
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
                
                manifest_rows.append({
                    'case_id': case_id,
                    'architecture': arch,
                    'difficulty': difficulty,
                    'seed': seed,
                    'spatial_region': region,
                    'target_present': not is_absent,
                    'reference_path': f"{arch_lower}/{case_id}/reference.png",
                    'search_path': f"{arch_lower}/{case_id}/search.png",
                    'global_context_path': f"{arch_lower}/{case_id}/global_context.png",
                    'global_mapping_scale': 1.0,
                    'global_mapping_offset_x': 0.0,
                    'global_mapping_offset_y': 0.0
                })
                
                count += 1
                case_idx += 1
                
            except RuntimeError as e:
                # Failure
                print(f"  -> Failed to generate {case_id}: {e}. Retrying with new seed...")
                failed_attempts += 1
                retry_attempts += 1
                periodic_rejections += gen.stats.get('periodic_rejections', 0)
                other_rejections += gen.stats.get('other_rejections', 0)
                continue

    with open(v4_manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow(r)
            
    # Calculate unique stats
    dram_present = sum(1 for r in manifest_rows if r['architecture'] == 'DRAM' and r['target_present'])
    dram_absent = sum(1 for r in manifest_rows if r['architecture'] == 'DRAM' and not r['target_present'])
    finfet_present = sum(1 for r in manifest_rows if r['architecture'] == 'FinFET' and r['target_present'])
    finfet_absent = sum(1 for r in manifest_rows if r['architecture'] == 'FinFET' and not r['target_present'])
    
    unique_cases = len(set([r['case_id'] for r in manifest_rows]))
    
    # Check actual files on disk
    unique_directories = 0
    unique_metadata = 0
    unique_images = 0
    for arch in ['dram', 'finfet']:
        arch_dir = os.path.join(v4_base_dir, arch)
        if os.path.exists(arch_dir):
            for case_name in os.listdir(arch_dir):
                c_dir = os.path.join(arch_dir, case_name)
                if os.path.isdir(c_dir):
                    unique_directories += 1
                    if os.path.exists(os.path.join(c_dir, 'metadata.json')):
                        unique_metadata += 1
                    if os.path.exists(os.path.join(c_dir, 'reference.png')) and os.path.exists(os.path.join(c_dir, 'search.png')):
                        unique_images += 1
            
    print("\n============================================================")
    print("V4 GENERATION MANIFEST")
    print("============================================================")
    print(f"REQUESTED_CASES         : {sum(quotas.values())}")
    print(f"SUCCESSFUL_CASES        : {len(manifest_rows)}")
    print(f"FAILED_ATTEMPTS         : {failed_attempts}")
    print(f"RETRY_ATTEMPTS          : {retry_attempts}")
    print(f"PERIODIC_REJECTIONS     : {periodic_rejections}")
    print(f"OTHER_REJECTIONS        : {other_rejections}")
    print(f"DRAM PRESENT            : {dram_present}")
    print(f"DRAM ABSENT             : {dram_absent}")
    print(f"FINFET PRESENT          : {finfet_present}")
    print(f"FINFET ABSENT           : {finfet_absent}")
    print(f"UNIQUE_CASE_DIRECTORIES : {unique_directories}")
    print(f"UNIQUE_METADATA_FILES   : {unique_metadata}")
    print(f"UNIQUE_IMAGE_PAIRS      : {unique_images}")
    print("============================================================\n")

    # Assertions
    try:
        assert len(manifest_rows) == 200, f"Expected 200 rows, got {len(manifest_rows)}"
        assert dram_present == 50, f"Expected 50 DRAM present, got {dram_present}"
        assert dram_absent == 50, f"Expected 50 DRAM absent, got {dram_absent}"
        assert finfet_present == 50, f"Expected 50 FinFET present, got {finfet_present}"
        assert finfet_absent == 50, f"Expected 50 FinFET absent, got {finfet_absent}"
        assert unique_directories == 200, f"Expected 200 unique case directories, got {unique_directories}"
        assert unique_metadata == 200, f"Expected 200 metadata files, got {unique_metadata}"
        assert unique_images == 200, f"Expected 200 image pairs, got {unique_images}"
    except AssertionError as e:
        print(f"ASSERTION FAILED: {e}")
        sys.exit(1)
        
    print("GENERATION SUCCESSFUL.")

if __name__ == '__main__':
    main()
