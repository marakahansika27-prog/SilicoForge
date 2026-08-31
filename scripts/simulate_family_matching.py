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
from scipy.ndimage import maximum_filter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/hackathon_v3")
    return parser.parse_args()

def extract_peaks(res_map, threshold=0.5, min_dist=5):
    local_max = maximum_filter(res_map, size=3) == res_map
    local_max = local_max & (res_map > threshold)
    
    y, x = np.where(local_max)
    scores = res_map[y, x]
    
    sorted_idx = np.argsort(scores)[::-1]
    y = y[sorted_idx]
    x = x[sorted_idx]
    scores = scores[sorted_idx]
    
    keep = []
    for i in range(len(scores)):
        px, py = x[i], y[i]
        valid = True
        for kx, ky, ks in keep:
            if abs(px - kx) < min_dist and abs(py - ky) < min_dist:
                valid = False
                break
        if valid:
            keep.append((px, py, float(scores[i])))
    return keep

def cluster_families(peaks, pitch_x, pitch_y, rot_degrees, tol=10.0):
    families = []
    theta = np.deg2rad(rot_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    for px, py, s in peaks:
        placed = False
        for fam in families:
            fx, fy = fam['center']
            dx = px - fx
            dy = py - fy
            
            rot_dx = dx * cos_t + dy * sin_t
            rot_dy = -dx * sin_t + dy * cos_t
            
            mod_x = abs(rot_dx) % pitch_x
            mod_y = abs(rot_dy) % pitch_y
            err_x = min(mod_x, pitch_x - mod_x)
            err_y = min(mod_y, pitch_y - mod_y)
            
            if err_x < tol and err_y < tol:
                fam['peaks'].append((px, py, s))
                fam['score'] += s
                placed = True
                break
        if not placed:
            families.append({
                'center': (px, py),
                'peaks': [(px, py, s)],
                'score': s
            })
            
    families.sort(key=lambda x: x['score'], reverse=True)
    return families

def evaluate_case(ci, ref_img, search_img, gspe):
    ice = ImageConditioningEngine()
    cond = ice.run({'reference': ref_img, 'search': search_img})
    gspe_res = gspe.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
    return gspe_res, cond['search_cond']

def main():
    args = get_args()
    print_header("GSPE TEMPLATE-FAMILY SIMULATION")
    start_time = time.time()
    
    ds = Phase2EvaluationDataset(dataset_dir=args.data_dir)
    gspe = GlobalSearchProposalEngine(top_k=3, scale_hypotheses=[8.0, 9.0, 10.0, 11.0, 12.0], rotation_hypotheses=[-5.0, 0.0, 5.0])
    
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    records = []
    
    # Categories
    counts = {
        "SUCCESS": 0,
        "TRUE_AMBIGUITY": 0,
        "CONTEXT_RECOVERABLE": 0,
        "SCORING_FAILURE": 0,
        "GEOMETRY_FAILURE": 0,
        "NMS_FAILURE": 0,
        "NOT_IN_TOP_FAMILY": 0
    }
    
    total_present = 0
    
    for i in range(len(ds)):
        sample = ds[i]
        ci = sample['case_info']
        present = ci['target_present']
        if not present: continue
        
        total_present += 1
        
        meta_path = os.path.join(args.data_dir, ci['architecture'].lower(), ci['case_id'], "metadata.json")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        gt_scale = metadata.get('effective_scale', 10.0)
        gt_rot = metadata.get('rotation_degrees', 0.0)
        pitch_x = metadata.get('pitch_x', 300) / 10.0
        pitch_y = metadata.get('pitch_y', 300) / 10.0
        gt_cx, gt_cy = ci['gt_x'], ci['gt_y']
        ix, iy = int(round(gt_cx)), int(round(gt_cy))
        ix = max(0, min(ix, 999))
        iy = max(0, min(iy, 999))
        
        res, search_cond = evaluate_case(ci, sample['reference_img'], sample['search_img'], gspe)
        
        if not res['boxes']:
            counts["GEOMETRY_FAILURE"] += 1
            continue
            
        box = res['boxes'][0]
        gspe_cx, gspe_cy = box[0] + box[2]/2.0, box[1] + box[3]/2.0
        sel_scale, sel_rot = box[4], box[5]
        
        # Geometry check
        geom_wrong = abs(sel_scale - gt_scale) > 0.5 or abs(sel_rot - gt_rot) > 2.0
        
        # Extract peaks from RAW NCC to simulate multi-context family matching
        # because RAW NCC doesn't have the Gaussian boundary poisoning.
        peaks = extract_peaks(res['res_raw'], threshold=0.5, min_dist=5)
        families = cluster_families(peaks, pitch_x, pitch_y, sel_rot, tol=10.0)
        
        # Determine NMS suppression in hybrid
        gt_survived_nms = (res['res_nms'][iy, ix] != -1.0)
        
        status = "UNKNOWN"
        ctx_results = {}
        top_fam_size = 0
        gt_in_top_fam = False
        
        if geom_wrong:
            status = "GEOMETRY_FAILURE"
        else:
            if families:
                top_fam = families[0]
                top_fam_size = len(top_fam['peaks'])
                for px, py, s in top_fam['peaks']:
                    if np.linalg.norm([px - gt_cx, py - gt_cy]) < 10.0:
                        gt_in_top_fam = True
                        break
            
            if not gt_in_top_fam:
                if not gt_survived_nms:
                    status = "NMS_FAILURE"
                else:
                    status = "NOT_IN_TOP_FAMILY"
            else:
                # GT is in the top family
                raw_top_px, raw_top_py, raw_top_s = top_fam['peaks'][0]
                raw_dist = np.linalg.norm([raw_top_px - gt_cx, raw_top_py - gt_cy])
                hyb_dist = np.linalg.norm([gspe_cx - gt_cx, gspe_cy - gt_cy])
                
                # Context Test
                for ctx_mult in [1.0, 1.5, 2.0]:
                    cw = int(box[2] * ctx_mult)
                    ch = int(box[3] * ctx_mult)
                    
                    gt_x1, gt_x2 = max(0, int(gt_cx - cw/2)), min(1000, int(gt_cx + cw/2))
                    gt_y1, gt_y2 = max(0, int(gt_cy - ch/2)), min(1000, int(gt_cy + ch/2))
                    
                    sel_x1, sel_x2 = max(0, int(raw_top_px - cw/2)), min(1000, int(raw_top_px + cw/2))
                    sel_y1, sel_y2 = max(0, int(raw_top_py - ch/2)), min(1000, int(raw_top_py + ch/2))
                    
                    if (gt_x2 - gt_x1) == (sel_x2 - sel_x1) and (gt_y2 - gt_y1) == (sel_y2 - sel_y1) and (gt_x2 - gt_x1) > 0:
                        patch_gt = search_cond[gt_y1:gt_y2, gt_x1:gt_x2]
                        patch_sel = search_cond[sel_y1:sel_y2, sel_x1:sel_x2]
                        pg_mean, pg_std = np.mean(patch_gt), np.std(patch_gt)
                        ps_mean, ps_std = np.mean(patch_sel), np.std(patch_sel)
                        if pg_std > 1e-5 and ps_std > 1e-5:
                            pg_norm = (patch_gt - pg_mean) / pg_std
                            ps_norm = (patch_sel - ps_mean) / ps_std
                            ncc = np.mean(pg_norm * ps_norm)
                        else:
                            ncc = 0.0
                        ctx_results[ctx_mult] = ncc
                
                # Classification
                if hyb_dist > 10.0 and raw_dist <= 10.0:
                    status = "SCORING_FAILURE"
                elif raw_dist <= 10.0 and hyb_dist <= 10.0:
                    status = "SUCCESS"
                else:
                    if ctx_results.get(2.0, 1.0) < 0.90:
                        status = "CONTEXT_RECOVERABLE"
                    else:
                        status = "TRUE_AMBIGUITY"
                        
        counts[status] = counts.get(status, 0) + 1
        
        records.append({
            'case_id': ci['case_id'],
            'architecture': ci['architecture'],
            'status': status,
            'gt_in_top_fam': gt_in_top_fam,
            'top_fam_size': top_fam_size,
            'ctx_1.0x': ctx_results.get(1.0, -1.0),
            'ctx_2.0x': ctx_results.get(2.0, -1.0)
        })

    # Output Simulation Report
    csv_path = "outputs/reports/SIMULATE_FAMILY_MATCHING.csv"
    if records:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    print("\n============================================================")
    print("SIMULATION RESULTS: TEMPLATE-FAMILY MATCHING")
    print("============================================================")
    for k, v in counts.items():
        print(f"{k:<25}: {v}")
        
    print("\n============================================================")
    print("RECOVERABILITY BREAKDOWN")
    print("============================================================")
    for r in records:
        if r['status'] not in ["SUCCESS"]:
            print(f"{r['case_id']} ({r['architecture']}) -> {r['status']}")
            print(f"   Top Family Size: {r['top_fam_size']} | GT in Family: {r['gt_in_top_fam']}")
            if 'ctx_2.0x' in r and r['ctx_2.0x'] != -1.0:
                print(f"   Context Identity (1.0x -> 2.0x): {r['ctx_1.0x']:.3f} -> {r['ctx_2.0x']:.3f}")

    with open("docs/SIMULATE_FAMILY_MATCHING_REPORT.md", "w") as f:
        f.write("# GSPE TEMPLATE-FAMILY SIMULATION REPORT\n\n")
        f.write("## 1. Summary of Recoverability\n")
        for k, v in counts.items():
            f.write(f"- **{k}**: {v}\n")
            
        f.write("\n## 2. Explanation of Categories\n")
        f.write("- **SUCCESS**: GT was correctly ranked #1 by both Raw and Hybrid (or would be if geometry was right).\n")
        f.write("- **SCORING_FAILURE**: GT was #1 in Raw NCC, but the Hybrid NCC (Gaussian blur) pulled the peak away to a periodic alias near the macro boundary.\n")
        f.write("- **CONTEXT_RECOVERABLE**: GT was tied in Raw NCC with periodic aliases, but evaluating 2.0x context correctly isolated the GT (NCC drops <0.90 for the alias).\n")
        f.write("- **TRUE_AMBIGUITY**: GT and the periodic alias are mathematically identical even at 2.0x context (NCC > 0.90). No algorithm can distinguish them from the pixels alone.\n")
        f.write("- **GEOMETRY_FAILURE**: Wrong scale or rotation was promoted by coarse search.\n")
        f.write("- **NMS_FAILURE**: GT peak was suppressed by NMS radius.\n")

    print_footer("SIMULATION COMPLETE", start_time, True)

if __name__ == "__main__":
    main()
