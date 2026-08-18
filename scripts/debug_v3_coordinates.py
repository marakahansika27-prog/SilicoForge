import os
import sys
import json
import csv
import math
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.coarse_search.gspe import GlobalSearchProposalEngine
from dataset.generator import HackathonDatasetGenerator

def main():
    print("V3 COORDINATE DEBUG TRACE")
    print("-------------------------")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v3'))
    
    manifest_path = os.path.join(base_dir, 'dataset_manifest.csv')
    cases = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cases.append(row)
            
    gspe = GlobalSearchProposalEngine(top_k=5, nms_radius=10, scale_hypotheses=[10.0], rotation_hypotheses=[0.0])
    
    for idx, c in enumerate(cases[:5]):  # Just trace first 5 cases
        case_id = c['case_id']
        arch = c['architecture']
        diff = c['difficulty']
        seed = int(c['seed'])
        region = c['spatial_region']
        
        gt_x = float(c['gt_x'])
        gt_y = float(c['gt_y'])
        
        gen = HackathonDatasetGenerator(seed=seed)
        ref_img, search_img, meta = gen.generate_case(case_id, arch, diff, version="v3", spatial_region=region, ref_size=1000)
        
        state = gspe.run({'reference': ref_img, 'search': search_img})
        
        _, max_val, _, max_loc = cv2.minMaxLoc(state['res_hybrid'])
        gspe_x, gspe_y = float(max_loc[0]), float(max_loc[1])
        
        dx = gspe_x - gt_x
        dy = gspe_y - gt_y
        error_px = math.sqrt(dx**2 + dy**2)
        
        meta_path = os.path.join(base_dir, arch.lower(), case_id, "metadata.json")
        with open(meta_path, 'r') as mf:
            m = json.load(mf)
            pitch_x = float(m['pitch_x']) / 10.0
            pitch_y = float(m['pitch_y']) / 10.0
        
        dx_pitch = dx / pitch_x if pitch_x > 0 else 0
        dy_pitch = dy / pitch_y if pitch_y > 0 else 0
        
        print(f"\n### Case {case_id} ({region})")
        print(f"- GT Coordinate: ({gt_x:.4f}, {gt_y:.4f})")
        print(f"- GSPE Center: ({gspe_x:.4f}, {gspe_y:.4f})")
        print(f"- Vector Delta: dx={dx:+.4f} px, dy={dy:+.4f} px")
        print(f"- Absolute Error: {error_px:.4f} px")
        print(f"- Pitch Analysis: Pitch X = {pitch_x:.1f}px, Pitch Y = {pitch_y:.1f}px")
        print(f"  - `dx` represents exactly {dx_pitch:+.2f} periods.")
        print(f"  - `dy` represents exactly {dy_pitch:+.2f} periods.")
        
        if abs(round(dx_pitch) - dx_pitch) < 0.1 and abs(round(dy_pitch) - dy_pitch) < 0.1:
            print(f"  - CONCLUSION: GENUINE PERIODIC ALIAS. Error is exactly a multiple of structural pitch.")
        else:
            print(f"  - CONCLUSION: NON-PERIODIC OFFSET. Possible coordinate bug or boundary artifact.")

if __name__ == "__main__":
    main()
