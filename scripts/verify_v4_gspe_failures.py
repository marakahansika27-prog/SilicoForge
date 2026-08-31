import os
import sys
import json
import cv2
import numpy as np
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.coarse_search.gspe import GlobalSearchProposalEngine

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def classify_failure(row):
    if row['GSPE Error'] <= 50.0:
        return "SUCCESS"
        
    gt_scale = row['GT Scale']
    gt_rot = row['GT Rotation']
    coarse_eval = eval(row['Coarse Evaluated Geometries'])
    
    # 1. Is GT geometry covered in hypotheses?
    scale_diffs = [abs(s - gt_scale) for _, (s, r) in coarse_eval]
    rot_diffs = [abs(r - gt_rot) for _, (s, r) in coarse_eval]
    if min(scale_diffs) > 0.5 or min(rot_diffs) > 2.0:
        return "GEOMETRY_NOT_COVERED"
        
    # Find the hypothesis closest to GT
    best_geom_diff = 9999
    gt_hyp = None
    gt_coarse_rank = -1
    
    # coarse_eval is sorted by score descending
    for i, (score, (s, r)) in enumerate(coarse_eval):
        d = abs(s - gt_scale) + abs(r - gt_rot)
        if d < best_geom_diff:
            best_geom_diff = d
            gt_hyp = (s, r)
            gt_coarse_rank = i
            
    # 2. Did GT geometry make it to top 3?
    if gt_coarse_rank > 2:
        return "GEOMETRY_COARSE_RANK_FAILURE"
        
    # If it was promoted, did the correct geometry actually get selected?
    sel_scale = row['Selected Scale']
    sel_rot = row['Selected Rotation']
    if abs(sel_scale - gt_scale) > 0.5 or abs(sel_rot - gt_rot) > 2.0:
        return "GEOMETRY_PROMOTION_FAILURE"
        
    # We reached full resolution with correct geometry AND it was selected.
    # 3. Was GT point suppressed by NMS?
    if row['GT NMS Suppressed']:
        return "NMS_FAILURE"
        
    # 4. Did GT have a higher hybrid score but lost to a non-periodic point?
    hyb_gt = float(row['Hybrid Score GT'])
    hyb_sel = float(row['Hybrid Score Selected'])
    if hyb_gt > hyb_sel:
        return "SUBPIXEL_FAILURE" # The GT point had a higher score, but a different subpixel peak was chosen
        
    # 5. It lost in score. Was the winner a periodic alias?
    pitch_x = float(row['Pitch X']) / 10.0 # search space
    pitch_y = float(row['Pitch Y']) / 10.0 # search space
    
    dx = abs(float(row['GT Center'][0]) - float(row['GSPE Center'][0]))
    dy = abs(float(row['GT Center'][1]) - float(row['GSPE Center'][1]))
    
    if pitch_x > 0 and pitch_y > 0:
        mod_x = min(dx % pitch_x, pitch_x - (dx % pitch_x))
        mod_y = min(dy % pitch_y, pitch_y - (dy % pitch_y))
        
        # If the spatial error is close to a multiple of pitch (e.g., within 20px) and it's far away (> 50px)
        if mod_x < 20 and mod_y < 20 and float(row['GSPE Error']) > 50.0:
            return "PERIODIC_ALIAS"
            
    # If the score was identical (within 1e-5), but it wasn't periodic alias, it's just a scoring tie-break issue.
    if abs(hyb_gt - hyb_sel) < 1e-4:
        return "CORRECT_GEOMETRY_SPATIAL_FAILURE"
            
    # If not a periodic alias, just scoring failure
    return "SCORING_FAILURE"

def main():
    print("V4 GSPE FORENSICS...")
    
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v4'))
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports'))
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    
    ensure_dir(reports_dir)
    ensure_dir(docs_dir)
    
    csv_out_path = os.path.join(reports_dir, 'V4_GSPE_FAILURE_FORENSICS.csv')
    md_out_path = os.path.join(docs_dir, 'V4_GSPE_FAILURE_FORENSIC_REPORT.md')
    
    gspe = GlobalSearchProposalEngine(top_k=5)
    
    results = []
    
    for arch in ['dram', 'finfet']:
        arch_dir = os.path.join(dataset_dir, arch)
        if not os.path.exists(arch_dir): continue
        
        for case_name in sorted(os.listdir(arch_dir)):
            case_dir = os.path.join(arch_dir, case_name)
            if not os.path.isdir(case_dir): continue
            
            meta_path = os.path.join(case_dir, "metadata.json")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            if not meta.get('target_present', False):
                continue
                
            print(f"Evaluating {case_name}...")
            
            ref_path = os.path.join(case_dir, "reference.png")
            search_path = os.path.join(case_dir, "search.png")
            
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
            
            gt_x = meta['gt_x']
            gt_y = meta['gt_y']
            gt_scale = meta['effective_scale']
            gt_rot = meta['rotation_degrees']
            pitch_x = meta.get('pitch_x', 0)
            pitch_y = meta.get('pitch_y', 0)
            
            # Run GSPE directly
            res = gspe.run({'reference': ref_img, 'search': search_img})
            
            # GSPE Rank 1 Box
            top_box = res['boxes'][0]
            gspe_x = top_box[0] + top_box[2]/2.0
            gspe_y = top_box[1] + top_box[3]/2.0
            sel_scale = top_box[4]
            sel_rot = top_box[5]
            
            err = distance((gt_x, gt_y), (gspe_x, gspe_y))
            
            # Int coords for GT lookup
            ix, iy = int(round(gt_x)), int(round(gt_y))
            ix = min(max(ix, 0), res['res_hybrid'].shape[1]-1)
            iy = min(max(iy, 0), res['res_hybrid'].shape[0]-1)
            
            sx, sy = int(round(gspe_x)), int(round(gspe_y))
            sx = min(max(sx, 0), res['res_hybrid'].shape[1]-1)
            sy = min(max(sy, 0), res['res_hybrid'].shape[0]-1)
            
            raw_gt = res['res_raw'][iy, ix]
            raw_sel = res['res_raw'][sy, sx]
            lf_gt = res['res_lowfreq'][iy, ix]
            lf_sel = res['res_lowfreq'][sy, sx]
            hyb_gt = res['res_hybrid'][iy, ix]
            hyb_sel = res['res_hybrid'][sy, sx]
            
            # NMS check
            nms_suppressed = res['res_nms'][iy, ix] == -1.0
            
            # Find GT coarse rank
            best_diff = 999
            gt_coarse_rank = -1
            coarse_eval = res['coarse_hypotheses']
            for i, (score, (s, r)) in enumerate(coarse_eval):
                d = abs(s - gt_scale) + abs(r - gt_rot)
                if d < best_diff:
                    best_diff = d
                    gt_coarse_rank = i
            
            row = {
                'Case ID': case_name,
                'Architecture': meta['architecture'],
                'GT Center': (gt_x, gt_y),
                'GSPE Center': (gspe_x, gspe_y),
                'GSPE Error': err,
                'GT Scale': gt_scale,
                'GT Rotation': gt_rot,
                'Selected Scale': sel_scale,
                'Selected Rotation': sel_rot,
                'GT Coarse Rank': gt_coarse_rank,
                'GT Reached Full Res': gt_coarse_rank <= 2,
                'Raw Score GT': raw_gt,
                'Raw Score Selected': raw_sel,
                'LowFreq Score GT': lf_gt,
                'LowFreq Score Selected': lf_sel,
                'Hybrid Score GT': hyb_gt,
                'Hybrid Score Selected': hyb_sel,
                'GT NMS Suppressed': nms_suppressed,
                'Pitch X': pitch_x,
                'Pitch Y': pitch_y,
                'Coarse Evaluated Geometries': str(coarse_eval)
            }
            
            row['Failure Category'] = classify_failure(row)
            results.append(row)
            
    # Write CSV
    headers = list(results[0].keys())
    with open(csv_out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Calculate stats
    errors = [r['GSPE Error'] for r in results]
    mean_err = np.mean(errors)
    med_err = np.median(errors)
    p90_err = np.percentile(errors, 90)
    max_err = np.max(errors)
    
    le_1 = sum(1 for e in errors if e <= 1.0)
    le_5 = sum(1 for e in errors if e <= 5.0)
    le_10 = sum(1 for e in errors if e <= 10.0)
    le_25 = sum(1 for e in errors if e <= 25.0)
    le_50 = sum(1 for e in errors if e <= 50.0)
    gt_50 = sum(1 for e in errors if e > 50.0)
    
    cats = {}
    for r in results:
        c = r['Failure Category']
        cats[c] = cats.get(c, 0) + 1
        
    # Write MD Report
    with open(md_out_path, 'w') as f:
        f.write("# V4 GSPE Failure Forensic Report\n\n")
        
        f.write("## Aggregate Statistics\n")
        f.write(f"- Positive Cases: {len(results)}\n")
        f.write(f"- Mean Error: {mean_err:.2f} px\n")
        f.write(f"- Median Error: {med_err:.2f} px\n")
        f.write(f"- P90 Error: {p90_err:.2f} px\n")
        f.write(f"- Max Error: {max_err:.2f} px\n")
        f.write(f"- <= 1 px: {le_1} ({le_1/len(results)*100:.1f}%)\n")
        f.write(f"- <= 5 px: {le_5} ({le_5/len(results)*100:.1f}%)\n")
        f.write(f"- <= 10 px: {le_10} ({le_10/len(results)*100:.1f}%)\n")
        f.write(f"- <= 25 px: {le_25} ({le_25/len(results)*100:.1f}%)\n")
        f.write(f"- <= 50 px: {le_50} ({le_50/len(results)*100:.1f}%)\n")
        f.write(f"- > 50 px: {gt_50} ({gt_50/len(results)*100:.1f}%)\n\n")
        
        f.write("## Failure Classification\n")
        for cat, cnt in cats.items():
            f.write(f"- {cat}: {cnt} ({cnt/len(results)*100:.1f}%)\n")
            
        f.write("\n## Geometry Analysis\n")
        coarse_fails = sum(1 for r in results if not r['GT Reached Full Res'])
        f.write(f"- Coarse Stage Failures (GT didn't reach top-3): {coarse_fails}\n")
        
        f.write("\n## Periodicity Analysis\n")
        periodic_fails = cats.get('PERIODIC_ALIAS', 0)
        f.write(f"- Periodic Aliases (Lost to identical structural peak at distance % pitch == 0): {periodic_fails}\n")
        
        f.write("\n## NMS Analysis\n")
        nms_fails = cats.get('NMS_FAILURE', 0)
        f.write(f"- GT Suppressed by NMS: {nms_fails}\n")
        
        f.write("\n## Root Cause\n")
        f.write("If PERIODIC_ALIAS is the dominant category, the root cause is that the 10x100x100 context window is completely insufficient to break local periodicity inside dense arrays (DRAM/FinFET), resulting in mathematically identical peaks. GSPE is forced to guess randomly among these identical peaks.\n")
        
        f.write("\n## Smallest Justified Fix\n")
        f.write("The smallest mathematically justified fix is to increase the Phase 2 dataset's `reference_img` context. The underlying `base_img_A` has exactly the same macro-structure, but GSPE needs to 'see' more of it to break local periodic ambiguity. We must increase the spatial context of the reference crop.\n")
        
        f.write("\nNO PRODUCTION CODE WAS MODIFIED.\n")

if __name__ == '__main__':
    main()
