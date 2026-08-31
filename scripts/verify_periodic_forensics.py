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

def get_rank(res, x, y):
    flat_res = res.flatten()
    val = res[y, x]
    rank = np.count_nonzero(flat_res > val) + 1
    return rank

def evaluate_case(ci, ref_img, search_img, gspe):
    ice = ImageConditioningEngine()
    cond = ice.run({'reference': ref_img, 'search': search_img})
    gspe_res = gspe.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
    return gspe_res, cond['search_cond']

def main():
    args = get_args()
    print_header("PHASE 2 - PERIODIC MATCHING FORENSIC INVESTIGATION")
    start_time = time.time()
    
    ds = Phase2EvaluationDataset(dataset_dir=args.data_dir)
    gspe = GlobalSearchProposalEngine(top_k=3, scale_hypotheses=[8.0, 9.0, 10.0, 11.0, 12.0], rotation_hypotheses=[-5.0, 0.0, 5.0])
    
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    records = []
    absent_records = []
    
    total_present = 0
    total_absent = 0
    
    # Classification counters
    periodic_alias_confirmed = 0
    periodic_alias_plausible = 0
    not_periodic = 0
    
    true_informational_ambiguity = 0
    sufficient_unique_info_ignored = 0
    score_function_destroys = 0
    context_starvation = 0
    context_does_not_solve = 0
    
    # Store all dx, dy for clustering analysis
    dx_list, dy_list = [], []
    
    for i in range(len(ds)):
        sample = ds[i]
        ci = sample['case_info']
        present = ci['target_present']
        
        meta_path = os.path.join(args.data_dir, ci['architecture'].lower(), ci['case_id'], "metadata.json")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        gt_scale = metadata.get('effective_scale', 10.0)
        gt_rot = metadata.get('rotation_degrees', 0.0)
        pitch_x = metadata.get('pitch_x', 300) / 10.0  # Search space pitch
        pitch_y = metadata.get('pitch_y', 300) / 10.0
        
        res, search_cond = evaluate_case(ci, sample['reference_img'], sample['search_img'], gspe)
        
        if not present:
            total_absent += 1
            max_score = res['scores'][0] if res['scores'] else -1.0
            absent_records.append({
                'case_id': ci['case_id'],
                'architecture': ci['architecture'],
                'max_score': max_score
            })
            continue
            
        total_present += 1
        gt_cx, gt_cy = ci['gt_x'], ci['gt_y']
        ix, iy = int(round(gt_cx)), int(round(gt_cy))
        ix = max(0, min(ix, 999))
        iy = max(0, min(iy, 999))
        
        if not res['boxes']:
            continue
            
        box = res['boxes'][0]
        gspe_x, gspe_y = box[0], box[1]
        gspe_cx, gspe_cy = box[0] + box[2]/2.0, box[1] + box[3]/2.0
        selected_scale = box[4]
        selected_rot = box[5]
        
        dx = gspe_cx - gt_cx
        dy = gspe_cy - gt_cy
        dist = float(np.linalg.norm([dx, dy]))
        
        dx_list.append(dx)
        dy_list.append(dy)
        
        # Periodic Alias Test
        mod_x = abs(dx) % pitch_x
        mod_y = abs(dy) % pitch_y
        dist_x_pitch = min(mod_x, pitch_x - mod_x)
        dist_y_pitch = min(mod_y, pitch_y - mod_y)
        nearest_pitch_err = float(np.linalg.norm([dist_x_pitch, dist_y_pitch]))
        
        alias_class = "NOT_PERIODIC"
        if dist > 50:
            if nearest_pitch_err < 5.0:
                alias_class = "PERIODIC_ALIAS_CONFIRMED"
                periodic_alias_confirmed += 1
            elif nearest_pitch_err < 15.0:
                alias_class = "PERIODIC_ALIAS_PLAUSIBLE"
                periodic_alias_plausible += 1
            else:
                not_periodic += 1
                
        # Raw vs LowFreq vs Hybrid
        raw_gt = res['res_raw'][iy, ix]
        low_gt = res['res_lowfreq'][iy, ix]
        hyb_gt = res['res_hybrid'][iy, ix]
        
        sy, sx = int(round(gspe_cy)), int(round(gspe_cx))
        sy = max(0, min(sy, 999))
        sx = max(0, min(sx, 999))
        
        raw_sel = res['res_raw'][sy, sx]
        low_sel = res['res_lowfreq'][sy, sx]
        hyb_sel = res['res_hybrid'][sy, sx]
        
        raw_diff = raw_sel - raw_gt
        low_diff = low_sel - low_gt
        hyb_diff = hyb_sel - hyb_gt
        
        # Rankings
        raw_rank = get_rank(res['res_raw'], ix, iy)
        low_rank = get_rank(res['res_lowfreq'], ix, iy)
        hyb_rank = get_rank(res['res_hybrid'], ix, iy)
        
        # Visual / Structural Match & Context Test
        gt_vs_sel_diff = -1.0
        ctx_results = {}
        if dist > 50:
            for ctx_mult in [1.0, 1.25, 1.5, 2.0]:
                cw = int(box[2] * ctx_mult)
                ch = int(box[3] * ctx_mult)
                
                gt_x1, gt_x2 = max(0, int(gt_cx - cw/2)), min(1000, int(gt_cx + cw/2))
                gt_y1, gt_y2 = max(0, int(gt_cy - ch/2)), min(1000, int(gt_cy + ch/2))
                
                sel_x1, sel_x2 = max(0, int(gspe_cx - cw/2)), min(1000, int(gspe_cx + cw/2))
                sel_y1, sel_y2 = max(0, int(gspe_cy - ch/2)), min(1000, int(gspe_cy + ch/2))
                
                # Check bounds
                if (gt_x2 - gt_x1) == (sel_x2 - sel_x1) and (gt_y2 - gt_y1) == (sel_y2 - sel_y1) and (gt_x2 - gt_x1) > 0:
                    patch_gt = search_cond[gt_y1:gt_y2, gt_x1:gt_x2]
                    patch_sel = search_cond[sel_y1:sel_y2, sel_x1:sel_x2]
                    
                    # Normalize patches
                    pg_mean, pg_std = np.mean(patch_gt), np.std(patch_gt)
                    ps_mean, ps_std = np.mean(patch_sel), np.std(patch_sel)
                    
                    if pg_std > 1e-5 and ps_std > 1e-5:
                        pg_norm = (patch_gt - pg_mean) / pg_std
                        ps_norm = (patch_sel - ps_mean) / ps_std
                        # Compute NCC between patches
                        ncc = np.mean(pg_norm * ps_norm)
                    else:
                        ncc = 0.0
                    
                    ctx_results[ctx_mult] = ncc
                    if ctx_mult == 1.0:
                        gt_vs_sel_diff = ncc
            
            if 1.0 in ctx_results and 2.0 in ctx_results:
                if ctx_results[1.0] > 0.95 and ctx_results[2.0] < 0.80:
                    context_starvation += 1
                    ambiguity_class = "CONTEXT_STARVATION"
                elif ctx_results[1.0] > 0.95 and ctx_results[2.0] > 0.95:
                    true_informational_ambiguity += 1
                    ambiguity_class = "TRUE_INFORMATIONAL_AMBIGUITY"
                else:
                    context_does_not_solve += 1
                    ambiguity_class = "CONTEXT_DOES_NOT_SOLVE"
                    
            if raw_rank == 1 and hyb_rank > 1:
                score_function_destroys += 1
                ambiguity_class = "SCORE_FUNCTION_DESTROYS_UNIQUE_INFORMATION"
                
        else:
            ambiguity_class = "CORRECT_LOCALIZATION"

        records.append({
            'case_id': ci['case_id'],
            'architecture': ci['architecture'],
            'gt_x': gt_cx, 'gt_y': gt_cy,
            'gspe_x': gspe_cx, 'gspe_y': gspe_cy,
            'error': dist,
            'dx': dx, 'dy': dy,
            'gt_scale': gt_scale, 'gt_rot': gt_rot,
            'selected_scale': selected_scale, 'selected_rot': selected_rot,
            'pitch_x': pitch_x, 'pitch_y': pitch_y,
            'nearest_pitch_err': nearest_pitch_err,
            'alias_class': alias_class,
            'ambiguity_class': ambiguity_class,
            'gt_vs_sel_ncc_1.0x': ctx_results.get(1.0, -1.0),
            'gt_vs_sel_ncc_1.5x': ctx_results.get(1.5, -1.0),
            'gt_vs_sel_ncc_2.0x': ctx_results.get(2.0, -1.0),
            'raw_diff': raw_diff, 'low_diff': low_diff, 'hyb_diff': hyb_diff,
            'raw_rank': raw_rank, 'low_rank': low_rank, 'hyb_rank': hyb_rank,
            'raw_gt': raw_gt, 'hyb_gt': hyb_gt
        })

    # Save CSV
    csv_path = "outputs/reports/GSPE_PERIODIC_CASE_ANALYSIS.csv"
    if records:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    # --- AGGREGATE STATISTICS ---
    print("\n============================================================")
    print("1. GSPE FORENSIC SUMMARY")
    print("============================================================")
    errs = [r['error'] for r in records if r['error'] >= 0]
    if errs:
        print(f"PRESENT COUNT : {len(errs)}")
        print(f"Mean Error    : {np.mean(errs):.2f} px")
        print(f"Median Error  : {np.median(errs):.2f} px")
        print(f"P90 Error     : {np.percentile(errs, 90):.2f} px")
        print(f"Max Error     : {np.max(errs):.2f} px")
        
        bins = [1, 5, 10, 25, 50, 100, 200]
        for b in bins:
            pct = 100.0 * sum(1 for e in errs if e < b) / len(errs)
            print(f"<{b:<3} px       : {pct:.1f}%")
        pct_high = 100.0 * sum(1 for e in errs if e >= 200) / len(errs)
        print(f">=200 px      : {pct_high:.1f}%")
        
    print(f"\nABSENT COUNT  : {len(absent_records)}")
    if absent_records:
        absent_scores = [r['max_score'] for r in absent_records]
        print(f"Highest Score : {np.max(absent_scores):.4f}")
        print(f"Mean Score    : {np.mean(absent_scores):.4f}")

    print("\n============================================================")
    print("2. ERROR VECTOR CLUSTERING (> 50 px)")
    print("============================================================")
    for r in records:
        if r['error'] > 50:
            print(f"{r['case_id']} ({r['architecture']}): GT=({r['gt_x']:.1f}, {r['gt_y']:.1f}) -> GSPE=({r['gspe_x']:.1f}, {r['gspe_y']:.1f})")
            print(f"   dx={r['dx']:+.1f}, dy={r['dy']:+.1f}, err={r['error']:.1f} px")
            print(f"   Scale: GT={r['gt_scale']:.1f} Sel={r['selected_scale']:.1f} | Rot: GT={r['gt_rot']:.1f} Sel={r['selected_rot']:.1f}")
            print(f"   Nearest Pitch Err: {r['nearest_pitch_err']:.1f} px (Pitch: {r['pitch_x']}x{r['pitch_y']})")
            print(f"   Alias Class: {r['alias_class']}")
            print(f"   Ambiguity  : {r['ambiguity_class']}")
            print(f"   Patch NCC (1.0x, 1.5x, 2.0x): {r['gt_vs_sel_ncc_1.0x']:.3f}, {r['gt_vs_sel_ncc_1.5x']:.3f}, {r['gt_vs_sel_ncc_2.0x']:.3f}")
            print(f"   GT Rank (Raw, Low, Hyb): {r['raw_rank']}, {r['low_rank']}, {r['hyb_rank']}")
            print(f"   Score Diff (Sel - GT): Raw={r['raw_diff']:+.4f}, Low={r['low_diff']:+.4f}, Hyb={r['hyb_diff']:+.4f}")
            print("-" * 60)

    # Output Markdown Report
    with open("docs/GSPE_PERIODIC_FORENSIC_REPORT.md", "w") as f:
        f.write("# GSPE PERIODIC MATCHING FORENSIC REPORT\n\n")
        f.write("## 1. Summary Statistics\n")
        f.write(f"- Mean Error: {np.mean(errs):.2f} px\n")
        f.write(f"- Median Error: {np.median(errs):.2f} px\n")
        f.write(f"- Max Error: {np.max(errs):.2f} px\n")
        
        f.write("\n## 2. Periodic Alias Test\n")
        f.write(f"- Confirmed Periodic Aliases (<5px pitch err): {periodic_alias_confirmed}\n")
        f.write(f"- Plausible Periodic Aliases (<15px pitch err): {periodic_alias_plausible}\n")
        f.write(f"- Not Periodic (>15px pitch err): {not_periodic}\n")
        
        f.write("\n## 3. Informational Ambiguity Test\n")
        f.write(f"- True Informational Ambiguity: {true_informational_ambiguity}\n")
        f.write(f"- Score Function Destroys Unique Info: {score_function_destroys}\n")
        f.write(f"- Context Starvation (2x context fixes it): {context_starvation}\n")
        f.write(f"- Context Does Not Solve: {context_does_not_solve}\n")
        
        # Decide Primary Root Cause
        if periodic_alias_confirmed > len(errs) / 4:
            if score_function_destroys > periodic_alias_confirmed / 2:
                primary = "NCC_SCORING_PREFERS_PERIODIC_CLONES (LowFreq bias)"
                secondary = "PERIODIC_INFORMATION_AMBIGUITY"
                fix = "Change hybrid scoring to rely strictly on raw NCC for structural fidelity."
            elif true_informational_ambiguity > periodic_alias_confirmed / 2:
                primary = "PERIODIC_INFORMATION_AMBIGUITY"
                secondary = "TEMPLATE_CONTEXT_TOO_SMALL"
                fix = "Propose rejection logic in dataset generator (periodicity_score), or context expansion."
            elif context_starvation > periodic_alias_confirmed / 2:
                primary = "TEMPLATE_CONTEXT_TOO_SMALL"
                secondary = "PERIODIC_INFORMATION_AMBIGUITY"
                fix = "Propose minimal context expansion for extracted reference templates."
            else:
                primary = "PERIODIC_INFORMATION_AMBIGUITY"
                secondary = "UNKNOWN"
                fix = "Investigate multi-context/template-family matching."
        else:
            primary = "COORDINATE/TEMPLATE_GEOMETRY_ERROR"
            secondary = "COARSE_GEOMETRY_FAILURE"
            fix = "Fix coordinate arithmetic in GSPE candidate bounding."

        f.write("\n## 4. Final Root-Cause Classification\n")
        f.write(f"**PRIMARY ROOT CAUSE:** {primary}\n\n")
        f.write(f"**SECONDARY ROOT CAUSE:** {secondary}\n\n")
        f.write(f"**AFFECTED CASES:** {periodic_alias_confirmed} periodic cases > 50px\n\n")
        f.write(f"**SMALLEST MATHEMATICALLY JUSTIFIED CHANGE:** {fix}\n\n")

    print_footer("PERIODIC FORENSIC AUDIT", start_time, True)

if __name__ == "__main__":
    main()
