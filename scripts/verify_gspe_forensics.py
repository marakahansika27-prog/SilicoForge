import os
import sys
import json
import csv
import time
import argparse
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.phase2_dataset import Phase2EvaluationDataset
from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine
from scripts.verify_utils import print_header, print_footer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/hackathon_v3")
    return parser.parse_args()

def is_local_maximum(res, x, y):
    h, w = res.shape
    if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
        return False
    val = res[y, x]
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0: continue
            if res[y + dy, x + dx] >= val:
                return False
    return True

def get_rank(res, x, y):
    flat_res = res.flatten()
    val = res[y, x]
    # Rank is the number of pixels strictly greater than val + 1
    rank = np.count_nonzero(flat_res > val) + 1
    return rank

def evaluate_case(ci, ref_img, search_img, gspe):
    ice = ImageConditioningEngine()
    cond = ice.run({'reference': ref_img, 'search': search_img})
    
    gspe_res = gspe.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
    
    return gspe_res

def main():
    args = get_args()
    print_header("GSPE COMPREHENSIVE FORENSIC AUDIT")
    start_time = time.time()
    
    ds = Phase2EvaluationDataset(dataset_dir=args.data_dir)
    gspe = GlobalSearchProposalEngine(top_k=3, scale_hypotheses=[8.0, 9.0, 10.0, 11.0, 12.0], rotation_hypotheses=[-5.0, 0.0, 5.0])
    
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    records = []
    
    geom_correct = 0
    geom_wrong_scale = 0
    geom_wrong_rot = 0
    geom_wrong_both = 0
    
    hybrid_caused = 0
    lowfreq_caused = 0
    raw_already_wrong = 0
    
    nms_suppressed = 0
    coarse_promotion_correct = 0
    gt_not_in_top3 = 0
    fullres_failure = 0
    
    total_present = 0
    
    for i in range(len(ds)):
        sample = ds[i]
        ci = sample['case_info']
        present = ci['target_present']
        
        # Read exact generator metadata
        meta_path = os.path.join(args.data_dir, ci['architecture'].lower(), ci['case_id'], "metadata.json")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        gt_scale = metadata.get('effective_scale', 10.0)
        gt_rot = metadata.get('rotation_degrees', 0.0)
        
        # Template coordinate audit
        ref_w, ref_h = metadata.get('reference_width', 1000), metadata.get('reference_height', 1000)
        
        res = evaluate_case(ci, sample['reference_img'], sample['search_img'], gspe)
        
        if present:
            total_present += 1
            gt_cx = ci['gt_x']
            gt_cy = ci['gt_y']
            
            ix, iy = int(round(gt_cx)), int(round(gt_cy))
            ix = max(0, min(ix, 999))
            iy = max(0, min(iy, 999))
            
            if res['boxes']:
                box = res['boxes'][0]
                gspe_x, gspe_y = box[0], box[1]
                gspe_cx, gspe_cy = box[0] + box[2]/2.0, box[1] + box[3]/2.0
                selected_scale = box[4]
                selected_rot = box[5]
                gspe_err = float(np.linalg.norm(np.array([gt_cx, gt_cy]) - np.array([gspe_cx, gspe_cy])))
                
                # Geometry Diagnostic
                scale_err = abs(selected_scale - gt_scale)
                rot_err = abs(selected_rot - gt_rot)
                
                is_scale_wrong = scale_err > 0.5
                is_rot_wrong = rot_err > 2.0
                
                if not is_scale_wrong and not is_rot_wrong:
                    geom_correct += 1
                elif is_scale_wrong and not is_rot_wrong:
                    geom_wrong_scale += 1
                elif not is_scale_wrong and is_rot_wrong:
                    geom_wrong_rot += 1
                else:
                    geom_wrong_both += 1
                    
                # Ranks
                raw_rank = get_rank(res['res_raw'], ix, iy)
                lowfreq_rank = get_rank(res['res_lowfreq'], ix, iy)
                hybrid_rank = get_rank(res['res_hybrid'], ix, iy)
                
                # Score Decomposition
                raw_val = res['res_raw'][iy, ix]
                low_val = res['res_lowfreq'][iy, ix]
                hyb_val = res['res_hybrid'][iy, ix]
                
                raw_sel = res['res_raw'][int(gspe_cy), int(gspe_cx)]
                low_sel = res['res_lowfreq'][int(gspe_cy), int(gspe_cx)]
                hyb_sel = res['res_hybrid'][int(gspe_cy), int(gspe_cx)]
                
                # Determine Scoring Bottleneck
                # Check where the absolute peak of res_raw is
                _, _, _, max_loc_raw = cv2.minMaxLoc(res['res_raw'])
                raw_peak_err = np.linalg.norm(np.array([gt_cx, gt_cy]) - np.array(max_loc_raw))
                
                _, _, _, max_loc_low = cv2.minMaxLoc(res['res_lowfreq'])
                low_peak_err = np.linalg.norm(np.array([gt_cx, gt_cy]) - np.array(max_loc_low))
                
                if raw_peak_err <= 10.0 and gspe_err > 10.0:
                    hybrid_caused += 1
                elif raw_peak_err > 10.0 and gspe_err > 10.0:
                    raw_already_wrong += 1
                elif low_peak_err > 10.0 and raw_peak_err <= 10.0:
                    lowfreq_caused += 1
                    
                # NMS Diagnostic
                is_gt_local_max = is_local_maximum(res['res_hybrid'], ix, iy)
                gt_survived_nms = (res['res_nms'][iy, ix] != -1.0)
                if is_gt_local_max and not gt_survived_nms:
                    nms_suppressed += 1
                    
                # Clustering Diagnostic (dx, dy)
                dx = gspe_cx - gt_cx
                dy = gspe_cy - gt_cy
                
                # Periodicity Analysis (multiples of 30 for DRAM, 20/60 for FinFET)
                is_periodic = False
                if ci['architecture'] == 'DRAM':
                    # Check if dx, dy are multiples of 30
                    if abs(dx % 30 - 15) > 10 and abs(dy % 30 - 15) > 10 and gspe_err > 25.0:
                        is_periodic = (abs(round(dx/30)*30 - dx) < 10) and (abs(round(dy/30)*30 - dy) < 10)
                elif ci['architecture'] == 'FinFET':
                    # Check if dx is multiple of 20, dy is multiple of 60 (or vice versa depending on orientation)
                    # For safety check if they fall roughly on 100/200 grid
                    pass
                
                # Coarse-to-fine Diagnostic
                coarse_hypos = res['coarse_hypotheses'] # list of (score, (scale, rot))
                gt_in_top3 = any(abs(h[1][0] - gt_scale) <= 0.5 and abs(h[1][1] - gt_rot) <= 2.0 for h in coarse_hypos[:3])
                
                if gt_in_top3 and gspe_err <= 10.0:
                    coarse_promotion_correct += 1
                elif not gt_in_top3:
                    gt_not_in_top3 += 1
                elif gt_in_top3 and gspe_err > 10.0:
                    fullres_failure += 1
            else:
                gspe_x = gspe_y = gspe_cx = gspe_cy = gspe_err = -1.0
                selected_scale = selected_rot = scale_err = rot_err = -1.0
                raw_val = low_val = hyb_val = raw_sel = low_sel = hyb_sel = -1.0
                raw_rank = lowfreq_rank = hybrid_rank = -1
                
            c1_x = res['boxes'][0][0] + res['boxes'][0][2]/2 if len(res['boxes']) > 0 else -1
            c1_y = res['boxes'][0][1] + res['boxes'][0][3]/2 if len(res['boxes']) > 0 else -1
            c1_s = res['scores'][0] if len(res['scores']) > 0 else -1
            c2_x = res['boxes'][1][0] + res['boxes'][1][2]/2 if len(res['boxes']) > 1 else -1
            c2_y = res['boxes'][1][1] + res['boxes'][1][3]/2 if len(res['boxes']) > 1 else -1
            c2_s = res['scores'][1] if len(res['scores']) > 1 else -1
            c3_x = res['boxes'][2][0] + res['boxes'][2][2]/2 if len(res['boxes']) > 2 else -1
            c3_y = res['boxes'][2][1] + res['boxes'][2][3]/2 if len(res['boxes']) > 2 else -1
            c3_s = res['scores'][2] if len(res['scores']) > 2 else -1
            
            records.append({
                'case_id': ci['case_id'],
                'architecture': ci['architecture'],
                'present_absent': 'PRESENT' if present else 'ABSENT',
                'gt_x': gt_cx, 'gt_y': gt_cy, 'gt_center_x': gt_cx, 'gt_center_y': gt_cy,
                'gspe_x': gspe_x, 'gspe_y': gspe_y, 'gspe_center_x': gspe_cx, 'gspe_center_y': gspe_cy,
                'gspe_error_px': gspe_err,
                'dx': dx if present and res['boxes'] else -1.0,
                'dy': dy if present and res['boxes'] else -1.0,
                'gt_scale': gt_scale, 'gt_rotation': gt_rot,
                'selected_scale': selected_scale, 'selected_rotation': selected_rot,
                'scale_error': scale_err, 'rotation_error': rot_err,
                'raw_ncc_at_selected': raw_sel, 'lowfreq_ncc_at_selected': low_sel, 'hybrid_score_selected': hyb_sel,
                'raw_ncc_at_GT': raw_val, 'lowfreq_ncc_at_GT': low_val, 'hybrid_score_at_GT': hyb_val,
                'rank_of_GT_location_if_available': hybrid_rank,
                'candidate_1_x': c1_x, 'candidate_1_y': c1_y, 'candidate_1_score': c1_s,
                'candidate_2_x': c2_x, 'candidate_2_y': c2_y, 'candidate_2_score': c2_s,
                'candidate_3_x': c3_x, 'candidate_3_y': c3_y, 'candidate_3_score': c3_s
            })

    # Save to CSV
    csv_path = "outputs/reports/PHASE2_SAMPLE_FORENSICS.csv"
    if records:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    # Output Decision Logic to Terminal and Report
    print("--- FORENSIC ANALYSIS SUMMARY ---")
    print(f"Total Present Cases: {total_present}")
    print(f"Geometry Correct   : {geom_correct}")
    print(f"Geom Wrong Scale   : {geom_wrong_scale}")
    print(f"Geom Wrong Rot     : {geom_wrong_rot}")
    print(f"Geom Wrong Both    : {geom_wrong_both}")
    
    print(f"\n--- SCORING DECOMPOSITION ---")
    print(f"Hybrid Caused Regression (Raw was correct, Hybrid drifted): {hybrid_caused}")
    print(f"LowFreq Caused Regression                                 : {lowfreq_caused}")
    print(f"Raw NCC already picked wrong periodic peak                : {raw_already_wrong}")
    
    print(f"\n--- NMS & PROMOTION DIAGNOSTIC ---")
    print(f"GT Suppressed by NMS (Radius too large)        : {nms_suppressed}")
    print(f"Coarse Promotion Correct (GT in Top 3 -> Win)  : {coarse_promotion_correct}")
    print(f"GT Geometry entirely missed by Coarse Search   : {gt_not_in_top3}")
    print(f"GT Geometry promoted but failed at Full Res    : {fullres_failure}")
    
    # Bottleneck Decision Tree Implementation
    gspe_errs = [r['gspe_error_px'] for r in records if r['gspe_error_px'] != -1.0]
    g_mean = np.mean(gspe_errs) if gspe_errs else 0.0
    
    if geom_wrong_scale + geom_wrong_rot + geom_wrong_both > total_present / 2:
        bottleneck = "fix geometry search"
    elif gt_not_in_top3 > total_present / 2:
        bottleneck = "fix promotion strategy"
    elif nms_suppressed > total_present / 2:
        bottleneck = "fix NMS"
    elif hybrid_caused > total_present / 4: # Adjust threshold based on evidence
        bottleneck = "fix hybrid scoring"
    elif raw_already_wrong > total_present / 4:
        bottleneck = "investigate structural/periodic matching"
    else:
        bottleneck = "continue deeper investigation"
        
    print(f"\n================================")
    print(f"FINAL DECISION TREE OUTPUT")
    print(f"================================")
    print(f"DOMINANT BOTTLENECK -> {bottleneck}")
    print(f"================================\n")
    print(f"Data saved to {csv_path}")

    # Generate Markdown Report
    with open("docs/PHASE2_PIPELINE_FORENSIC_AUDIT.md", "w") as f:
        f.write("# PHASE 2 PIPELINE FORENSIC AUDIT\n\n")
        f.write(f"**Total Present Cases**: {total_present}\n\n")
        
        f.write("## 1. Geometry Diagnostic\n")
        f.write(f"- Geometry Correct: {geom_correct}\n")
        f.write(f"- Wrong Scale: {geom_wrong_scale}\n")
        f.write(f"- Wrong Rotation: {geom_wrong_rot}\n")
        f.write(f"- Wrong Both: {geom_wrong_both}\n\n")
        
        f.write("## 2. Scoring & Hybrid Diagnostic\n")
        f.write(f"- Hybrid Caused Regression: {hybrid_caused}\n")
        f.write(f"- LowFreq Caused Regression: {lowfreq_caused}\n")
        f.write(f"- Raw already wrong (Periodic Alias): {raw_already_wrong}\n\n")
        
        f.write("## 3. Coarse-to-Fine Diagnostic\n")
        f.write(f"- GT Geometry missed by Coarse Search: {gt_not_in_top3}\n")
        f.write(f"- GT Geometry in Top 3, failed FullRes: {fullres_failure}\n")
        f.write(f"- GT Geometry in Top 3, won FullRes: {coarse_promotion_correct}\n\n")
        
        f.write("## FINAL BOTTLENECK DECISION\n")
        f.write(f"**{bottleneck}**\n")

    print_footer("GSPE FORENSIC AUDIT", start_time, True)

if __name__ == "__main__":
    main()
