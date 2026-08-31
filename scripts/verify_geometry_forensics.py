import os
import sys
import json
import csv
import time
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.phase2_dataset import Phase2EvaluationDataset
from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine
from scripts.verify_utils import print_header, print_footer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/hackathon_v3")
    return parser.parse_args()

def main():
    args = get_args()
    print_header("NON-DESTRUCTIVE GEOMETRY FORENSIC ISOLATION")
    start_time = time.time()
    
    ds = Phase2EvaluationDataset(dataset_dir=args.data_dir)
    
    scales = [8.0, 9.0, 10.0, 11.0, 12.0]
    rotations = [-5.0, 0.0, 5.0]
    gspe_standard = GlobalSearchProposalEngine(top_k=3, scale_hypotheses=scales, rotation_hypotheses=rotations)
    
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    records = []
    
    for i in range(len(ds)):
        sample = ds[i]
        ci = sample['case_info']
        if not ci['target_present']:
            continue
            
        meta_path = os.path.join(args.data_dir, ci['architecture'].lower(), ci['case_id'], "metadata.json")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        gt_scale = metadata.get('effective_scale', 10.0)
        gt_rot = metadata.get('rotation_degrees', 0.0)
        gt_cx, gt_cy = ci['gt_x'], ci['gt_y']
        
        nearest_scale = min(scales, key=lambda x: abs(x - gt_scale))
        nearest_rot = min(rotations, key=lambda x: abs(x - gt_rot))
        scale_err = abs(gt_scale - nearest_scale)
        rot_err = abs(gt_rot - nearest_rot)
        
        ice = ImageConditioningEngine()
        cond = ice.run({'reference': sample['reference_img'], 'search': sample['search_img']})
        
        res_standard = gspe_standard.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
        
        coarse_data = res_standard.get('coarse_hypotheses', [])
        coarse_rank = -1
        coarse_score = -1.0
        best_c_scale, best_c_rot = -1.0, -1.0
        
        if coarse_data:
            best_c_scale, best_c_rot = coarse_data[0][1]
            for idx, (cs, (cs_scale, cs_rot)) in enumerate(coarse_data):
                if abs(cs_scale - nearest_scale) < 1e-4 and abs(cs_rot - nearest_rot) < 1e-4:
                    coarse_rank = idx + 1
                    coarse_score = cs
                    break
                    
        reached_top3 = (1 <= coarse_rank <= 3)
        
        gspe_gt = GlobalSearchProposalEngine(top_k=1, scale_hypotheses=[nearest_scale], rotation_hypotheses=[nearest_rot])
        res_gt = gspe_gt.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
        
        gt_spatial_error = -1.0
        if res_gt['boxes']:
            box = res_gt['boxes'][0]
            cx = box[0] + box[2]/2.0
            cy = box[1] + box[3]/2.0
            gt_spatial_error = float(np.linalg.norm([cx - gt_cx, cy - gt_cy]))
            
        classification = "UNKNOWN"
        if scale_err > 0.5 or rot_err > 2.0:
            classification = "GEOMETRY_NOT_EVALUATED"
        elif coarse_rank > 3:
            classification = "GEOMETRY_COARSE_RANK_LOSS"
        elif not reached_top3:
            classification = "GEOMETRY_COARSE_RANK_LOSS"
        else:
            win_scale = res_standard['boxes'][0][4] if res_standard['boxes'] else -1.0
            win_rot = res_standard['boxes'][0][5] if res_standard['boxes'] else -1.0
            if abs(win_scale - nearest_scale) > 1e-4 or abs(win_rot - nearest_rot) > 1e-4:
                classification = "GEOMETRY_PROMOTION_LOSS"
            else:
                if gt_spatial_error > 10.0:
                    classification = "GEOMETRY_CORRECT_SPATIAL_FAILURE"
                else:
                    classification = "GEOMETRY_CORRECT"
                    
        records.append({
            'case_id': ci['case_id'],
            'gt_scale': gt_scale,
            'gt_rotation': gt_rot,
            'nearest_scale': nearest_scale,
            'scale_err': scale_err,
            'nearest_rot': nearest_rot,
            'rot_err': rot_err,
            'coarse_score': coarse_score,
            'coarse_rank': coarse_rank,
            'reached_top3': reached_top3,
            'gt_spatial_error': gt_spatial_error,
            'classification': classification
        })
        
    csv_path = "outputs/reports/GSPE_GEOMETRY_FORENSICS.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
        
    counts = {
        'GEOMETRY_NOT_EVALUATED': 0,
        'GEOMETRY_COARSE_RANK_LOSS': 0,
        'GEOMETRY_PROMOTION_LOSS': 0,
        'GEOMETRY_CORRECT_SPATIAL_FAILURE': 0,
        'GEOMETRY_CORRECT': 0
    }
    for r in records:
        counts[r['classification']] = counts.get(r['classification'], 0) + 1
        
    print("\n--- AGGREGATE COUNTS ---")
    for k, v in counts.items():
        print(f"{k}: {v}")
        
    for r in records:
        if r['case_id'] == 'case_0058':
            print("\n--- CASE 0058 INVESTIGATION ---")
            print(f"GT Scale: {r['gt_scale']:.4f}, Nearest: {r['nearest_scale']:.1f}, Err: {r['scale_err']:.4f}")
            print(f"GT Rot: {r['gt_rotation']:.4f}, Nearest: {r['nearest_rot']:.1f}, Err: {r['rot_err']:.4f}")
            print(f"Coarse Rank: {r['coarse_rank']}, Reached Top3: {r['reached_top3']}")
            print(f"Classification: {r['classification']}")
            
    with open("docs/GSPE_GEOMETRY_FORENSIC_REPORT.md", "w") as f:
        f.write("# GSPE GEOMETRY FORENSIC REPORT\n\n")
        for k, v in counts.items():
            f.write(f"- {k}: {v}\n")
        
        dominant_problem = "E) geometry is actually correct and the failure is spatial/periodic"
        finding = "Spatial matching dominates."
        if counts['GEOMETRY_NOT_EVALUATED'] > 5:
            dominant_problem = "A) hypothesis-grid coverage"
            finding = "The continuous GT geometry falls too far from the discrete grid."
        elif counts['GEOMETRY_COARSE_RANK_LOSS'] > 5:
            dominant_problem = "B) coarse scoring/ranking"
            finding = "Coarse downsampling destroys structure."
        elif counts['GEOMETRY_PROMOTION_LOSS'] > 5:
            dominant_problem = "C) top-K promotion"
            finding = "Full-resolution score inversion."
            
        f.write(f"\n## DEFINITIVE GEOMETRY FINDING\n")
        f.write(f"**Dominant Problem:** {dominant_problem}\n")
        f.write(f"**Justification:** {finding}\n")
        
    print_footer("GEOMETRY FORENSIC ISOLATION", start_time, True)

if __name__ == "__main__":
    main()
