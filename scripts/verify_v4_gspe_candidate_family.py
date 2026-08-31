import os
import sys
import json
import cv2
import numpy as np
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logger import Profiler

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class SimulationGSPE:
    def __init__(self, scale_hypotheses=None, rotation_hypotheses=None, promotion_k=3, nms_mult=1.0, top_k=10):
        self.top_k = top_k
        self.scale_hypotheses = scale_hypotheses if scale_hypotheses is not None else [8.0, 9.0, 10.0, 11.0, 12.0]
        self.rotation_hypotheses = rotation_hypotheses if rotation_hypotheses is not None else [-5.0, 0.0, 5.0]
        self.promotion_k = promotion_k
        self.nms_mult = nms_mult
        self.stats = {}

    def run(self, inputs: dict) -> dict:
        ref_img = inputs['reference']
        search_img = inputs['search']
        
        h_ref, w_ref = ref_img.shape
        search_h, search_w = search_img.shape
        
        best_res = None
        best_res_raw = None
        best_res_lowfreq = None
        best_scale_map = None
        best_rot_map = None
        best_w_map = None
        best_h_map = None
        
        hypotheses = []
        for scale in self.scale_hypotheses:
            for rot in self.rotation_hypotheses:
                hypotheses.append((scale, rot))
                
        if len(hypotheses) > self.promotion_k:
            search_coarse = cv2.resize(search_img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            ref_coarse = cv2.resize(ref_img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            
            coarse_scores = []
            for scale, rot in hypotheses:
                c_w_ref, c_h_ref = ref_coarse.shape[1], ref_coarse.shape[0]
                c_M_rot = cv2.getRotationMatrix2D((c_w_ref / 2.0, c_h_ref / 2.0), rot, 1.0)
                
                c_cos_val = np.abs(c_M_rot[0, 0])
                c_sin_val = np.abs(c_M_rot[0, 1])
                c_bound_w = int((c_h_ref * c_sin_val) + (c_w_ref * c_cos_val))
                c_bound_h = int((c_h_ref * c_cos_val) + (c_w_ref * c_sin_val))
                
                c_M_rot[0, 2] += (c_bound_w / 2) - (c_w_ref / 2)
                c_M_rot[1, 2] += (c_bound_h / 2) - (c_h_ref / 2)
                
                c_ref_rotated = cv2.warpAffine(ref_coarse, c_M_rot, (c_bound_w, c_bound_h), flags=cv2.INTER_LINEAR)
                                             
                c_theta = np.deg2rad(np.abs(rot))
                c_crop_w = int(c_w_ref / (np.sin(c_theta) + np.cos(c_theta)))
                c_crop_h = int(c_h_ref / (np.sin(c_theta) + np.cos(c_theta)))
                
                c_cx_rot, c_cy_rot = c_bound_w // 2, c_bound_h // 2
                c_y1 = max(0, c_cy_rot - c_crop_h // 2)
                c_y2 = c_y1 + c_crop_h
                c_x1 = max(0, c_cx_rot - c_crop_w // 2)
                c_x2 = c_x1 + c_crop_w
                
                c_ref_rotated_cropped = c_ref_rotated[c_y1:c_y2, c_x1:c_x2]
                
                c_scaled_bound_w = int(round(c_crop_w / scale))
                c_scaled_bound_h = int(round(c_crop_h / scale))
                
                if c_scaled_bound_w <= search_coarse.shape[1] and c_scaled_bound_h <= search_coarse.shape[0]:
                    c_ref_scaled = cv2.resize(c_ref_rotated_cropped, (c_scaled_bound_w, c_scaled_bound_h), interpolation=cv2.INTER_AREA)
                    c_res = cv2.matchTemplate(search_coarse, c_ref_scaled, cv2.TM_CCOEFF_NORMED)
                    _, c_max_val, _, _ = cv2.minMaxLoc(c_res)
                    coarse_scores.append(c_max_val)
                else:
                    coarse_scores.append(-1.0)
                    
            sorted_hypotheses = [x for _, x in sorted(zip(coarse_scores, hypotheses), reverse=True)]
            final_hypotheses = sorted_hypotheses[:self.promotion_k]
            coarse_data = sorted(zip(coarse_scores, hypotheses), reverse=True)
        else:
            final_hypotheses = hypotheses
            coarse_data = [(0.0, h) for h in hypotheses]
            
        for scale, rot in final_hypotheses:
            new_w = int(round(w_ref / scale))
            new_h = int(round(h_ref / scale))
            
            M_rot = cv2.getRotationMatrix2D((w_ref / 2.0, h_ref / 2.0), rot, 1.0)
            cos_val = np.abs(M_rot[0, 0])
            sin_val = np.abs(M_rot[0, 1])
            bound_w = int((h_ref * sin_val) + (w_ref * cos_val))
            bound_h = int((h_ref * cos_val) + (w_ref * sin_val))
            
            M_rot[0, 2] += (bound_w / 2) - (w_ref / 2)
            M_rot[1, 2] += (bound_h / 2) - (h_ref / 2)
            
            ref_rotated = cv2.warpAffine(ref_img, M_rot, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            theta = np.deg2rad(np.abs(rot))
            crop_w = int(w_ref / (np.sin(theta) + np.cos(theta)))
            crop_h = int(h_ref / (np.sin(theta) + np.cos(theta)))
            
            cx_rot = bound_w // 2
            cy_rot = bound_h // 2
            y1 = max(0, cy_rot - crop_h // 2)
            y2 = y1 + crop_h
            x1 = max(0, cx_rot - crop_w // 2)
            x2 = x1 + crop_w
            
            ref_rotated_cropped = ref_rotated[y1:y2, x1:x2]
            
            scaled_bound_w = int(round(crop_w / scale))
            scaled_bound_h = int(round(crop_h / scale))
            
            if scaled_bound_w > search_w or scaled_bound_h > search_h:
                continue
                
            ref_scaled = cv2.resize(ref_rotated_cropped, (scaled_bound_w, scaled_bound_h), interpolation=cv2.INTER_AREA)
            
            res_raw = cv2.matchTemplate(search_img, ref_scaled, cv2.TM_CCOEFF_NORMED)
            ref_blurred = cv2.GaussianBlur(ref_scaled, (31, 31), 15)
            search_blurred = cv2.GaussianBlur(search_img, (31, 31), 15)
            res_lowfreq = cv2.matchTemplate(search_blurred, ref_blurred, cv2.TM_CCOEFF_NORMED)
            
            res = 0.9 * res_raw + 0.1 * res_lowfreq
            
            pad_top = scaled_bound_h // 2
            pad_bottom = search_h - res.shape[0] - pad_top
            pad_left = scaled_bound_w // 2
            pad_right = search_w - res.shape[1] - pad_left
            
            res_centered = cv2.copyMakeBorder(res, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=-1.0)
            res_raw_centered = cv2.copyMakeBorder(res_raw, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=-1.0)
            res_lowfreq_centered = cv2.copyMakeBorder(res_lowfreq, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=-1.0)
            
            if best_res is None:
                best_res = np.full_like(res_centered, -1.0)
                best_res_raw = np.full_like(res_centered, -1.0)
                best_res_lowfreq = np.full_like(res_centered, -1.0)
                best_scale_map = np.zeros_like(res_centered)
                best_rot_map = np.zeros_like(res_centered)
                best_w_map = np.zeros_like(res_centered, dtype=np.int32)
                best_h_map = np.zeros_like(res_centered, dtype=np.int32)
                
            mask = res_centered > best_res
            best_res[mask] = res_centered[mask]
            best_res_raw[mask] = res_raw_centered[mask]
            best_res_lowfreq[mask] = res_lowfreq_centered[mask]
            best_scale_map[mask] = scale
            best_rot_map[mask] = rot
            best_w_map[mask] = scaled_bound_w
            best_h_map[mask] = scaled_bound_h
            
        res = best_res
        
        candidates = []
        res_nms = res.copy() if res is not None else np.zeros((search_h, search_w))
        
        if best_res is not None:
            for _ in range(self.top_k):
                _, max_val, _, max_loc = cv2.minMaxLoc(res_nms)
                if max_val < 0: break
                    
                x, y = max_loc
                delta_x = 0.0
                delta_y = 0.0
                
                if x > 0 and x < res.shape[1] - 1 and y > 0 and y < res.shape[0] - 1:
                    fx_m1 = float(res[y, x - 1])
                    fx_0  = float(res[y, x])
                    fx_p1 = float(res[y, x + 1])
                    denom_x = fx_m1 - 2 * fx_0 + fx_p1
                    if abs(denom_x) > 1e-6:
                        dx = 0.5 * (fx_m1 - fx_p1) / denom_x
                        if abs(dx) <= 0.5: delta_x = dx
                            
                    fy_m1 = float(res[y - 1, x])
                    fy_0  = float(res[y, x])
                    fy_p1 = float(res[y + 1, x])
                    denom_y = fy_m1 - 2 * fy_0 + fy_p1
                    if abs(denom_y) > 1e-6:
                        dy = 0.5 * (fy_m1 - fy_p1) / denom_y
                        if abs(dy) <= 0.5: delta_y = dy
                
                sub_x = x + delta_x
                sub_y = y + delta_y
                
                w_cand = best_w_map[y, x]
                h_cand = best_h_map[y, x]
                scale_cand = best_scale_map[y, x]
                rot_cand = best_rot_map[y, x]
                
                tl_x = sub_x - (w_cand / 2.0)
                tl_y = sub_y - (h_cand / 2.0)
                
                candidates.append({
                    'center_x': sub_x,
                    'center_y': sub_y,
                    'box': (tl_x, tl_y, int(w_cand), int(h_cand)),
                    'scale': scale_cand,
                    'rot': rot_cand,
                    'hyb_score': float(max_val),
                    'raw_score': float(best_res_raw[y, x]),
                    'lf_score': float(best_res_lowfreq[y, x])
                })
                
                nms_r_x = int(max(10, int(w_cand / 4.0)) * self.nms_mult)
                nms_r_y = int(max(10, int(h_cand / 4.0)) * self.nms_mult)
                    
                y1, y2 = max(0, y - nms_r_y), min(res_nms.shape[0], y + nms_r_y)
                x1, x2 = max(0, x - nms_r_x), min(res_nms.shape[1], x + nms_r_x)
                res_nms[y1:y2, x1:x2] = -1.0

        return {
            'candidates': candidates,
            'res_hybrid': res,
            'res_nms': res_nms,
            'coarse_hypotheses': coarse_data
        }

def classify(err, gt_scale, gt_rot, coarse_eval, gspe_x, gspe_y, gt_x, gt_y, pitch_x, pitch_y, hyb_gt, hyb_sel, nms_suppressed, promotion_k):
    if err <= 50.0:
        return "SUCCESS"
        
    scale_diffs = [abs(s - gt_scale) for _, (s, r) in coarse_eval]
    rot_diffs = [abs(r - gt_rot) for _, (s, r) in coarse_eval]
    if min(scale_diffs) > 0.5 or min(rot_diffs) > 2.0:
        return "GEOMETRY_NOT_COVERED"
        
    best_diff = 999
    gt_coarse_rank = -1
    for i, (score, (s, r)) in enumerate(coarse_eval):
        d = abs(s - gt_scale) + abs(r - gt_rot)
        if d < best_diff:
            best_diff = d
            gt_coarse_rank = i
            
    if gt_coarse_rank >= promotion_k:
        return "GEOMETRY_COARSE_RANK_FAILURE"
        
    if nms_suppressed:
        return "NMS_FAILURE"
        
    p_x = pitch_x / 10.0
    p_y = pitch_y / 10.0
    dx = abs(gt_x - gspe_x)
    dy = abs(gt_y - gspe_y)
    
    if p_x > 0 and p_y > 0:
        mod_x = min(dx % p_x, p_x - (dx % p_x))
        mod_y = min(dy % p_y, p_y - (dy % p_y))
        if mod_x < 20 and mod_y < 20:
            if abs(hyb_gt - hyb_sel) < 0.05:
                return "PERIODIC_ALIAS"
            
    if abs(hyb_gt - hyb_sel) < 1e-4:
        return "TRUE_AMBIGUITY"
        
    return "SCORING_FAILURE"

def apply_candidate_family_tiebreak(candidates, score_tolerance=0.015):
    if not candidates:
        return None
        
    max_hyb = candidates[0]['hyb_score']
    family = [c for c in candidates if max_hyb - c['hyb_score'] <= score_tolerance]
    
    # Tie-break 1: highest low-frequency score
    best_c = max(family, key=lambda c: c['lf_score'])
    
    return best_c, len(family)

def main():
    print("V4 GSPE CANDIDATE FAMILY EXPERIMENT...")
    
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v4'))
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports'))
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    
    ensure_dir(reports_dir)
    ensure_dir(docs_dir)
    
    csv_out_path = os.path.join(reports_dir, 'GSPE_CANDIDATE_FAMILY_EXPERIMENT.csv')
    md_out_path = os.path.join(docs_dir, 'GSPE_CANDIDATE_FAMILY_EXPERIMENT_REPORT.md')
    
    cases = []
    for arch in ['dram', 'finfet']:
        arch_dir = os.path.join(dataset_dir, arch)
        if not os.path.exists(arch_dir): continue
        for case_name in sorted(os.listdir(arch_dir)):
            case_dir = os.path.join(arch_dir, case_name)
            if not os.path.isdir(case_dir): continue
            meta_path = os.path.join(case_dir, "metadata.json")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if not meta.get('target_present', False): continue
            
            ref_path = os.path.join(case_dir, "reference.png")
            search_path = os.path.join(case_dir, "search.png")
            cases.append({
                'id': case_name,
                'meta': meta,
                'ref': cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE),
                'search': cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
            })
            
    # Using baseline grid and configuration for fair comparison, but with top_k=10
    base_scales = [8.0, 9.0, 10.0, 11.0, 12.0]
    base_rots = [-5.0, 0.0, 5.0]
    
    results = []
    
    print(f"Loaded {len(cases)} positive cases.")
    
    for idx, c in enumerate(cases):
        print(f"[{idx+1}/{len(cases)}] Evaluating {c['id']}...")
        
        gt_x = c['meta']['gt_x']
        gt_y = c['meta']['gt_y']
        gt_scale = c['meta']['effective_scale']
        gt_rot = c['meta']['rotation_degrees']
        pitch_x = c['meta'].get('pitch_x', 0)
        pitch_y = c['meta'].get('pitch_y', 0)
        
        with Profiler("gspe_run") as p:
            gspe = SimulationGSPE(scale_hypotheses=base_scales, rotation_hypotheses=base_rots, promotion_k=3, nms_mult=1.0, top_k=10)
            res = gspe.run({'reference': c['ref'], 'search': c['search']})
            
        runtime = p.elapsed_ms
        
        candidates = res['candidates']
        if not candidates:
            continue
            
        # 1. Baseline logic: Pick rank 0
        base_c = candidates[0]
        base_err = distance((gt_x, gt_y), (base_c['center_x'], base_c['center_y']))
        
        ix = min(max(int(round(gt_x)), 0), res['res_hybrid'].shape[1]-1)
        iy = min(max(int(round(gt_y)), 0), res['res_hybrid'].shape[0]-1)
        hyb_gt = res['res_hybrid'][iy, ix]
        nms_supp = res['res_nms'][iy, ix] == -1.0
        
        base_cat = classify(base_err, gt_scale, gt_rot, res['coarse_hypotheses'], base_c['center_x'], base_c['center_y'], gt_x, gt_y, pitch_x, pitch_y, hyb_gt, base_c['hyb_score'], nms_supp, 3)
        
        # 2. Candidate Family logic
        new_c, family_size = apply_candidate_family_tiebreak(candidates, score_tolerance=0.015)
        new_err = distance((gt_x, gt_y), (new_c['center_x'], new_c['center_y']))
        new_cat = classify(new_err, gt_scale, gt_rot, res['coarse_hypotheses'], new_c['center_x'], new_c['center_y'], gt_x, gt_y, pitch_x, pitch_y, hyb_gt, new_c['hyb_score'], nms_supp, 3)
        
        # Check if GT was in the family at all
        gt_in_family = False
        for f_cand in candidates[:family_size]:
            if distance((gt_x, gt_y), (f_cand['center_x'], f_cand['center_y'])) <= 50.0:
                gt_in_family = True
                break
                
        results.append({
            'Case ID': c['id'],
            'GT X': gt_x,
            'GT Y': gt_y,
            'Base Error': base_err,
            'Base Cat': base_cat,
            'New Error': new_err,
            'New Cat': new_cat,
            'Family Size': family_size,
            'GT In Family': gt_in_family,
            'Runtime (ms)': runtime
        })
        
    # Write CSV
    headers = list(results[0].keys())
    with open(csv_out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Calculate stats
    base_errs = [r['Base Error'] for r in results]
    new_errs = [r['New Error'] for r in results]
    runtimes = [r['Runtime (ms)'] for r in results]
    
    # Write MD Report
    with open(md_out_path, 'w') as f:
        f.write("# GSPE Candidate Family Experiment Report\n\n")
        
        f.write("## Baseline Metrics (Rank 0 only)\n")
        f.write(f"- Mean: {np.mean(base_errs):.2f}\n")
        f.write(f"- Median: {np.median(base_errs):.2f}\n")
        f.write(f"- P90: {np.percentile(base_errs, 90):.2f}\n")
        f.write(f"- Max: {np.max(base_errs):.2f}\n")
        f.write(f"- <= 5: {sum(1 for e in base_errs if e <= 5.0)}\n")
        f.write(f"- <= 50: {sum(1 for e in base_errs if e <= 50.0)}\n")
        f.write(f"- > 50: {sum(1 for e in base_errs if e > 50.0)}\n\n")
        
        f.write("## Candidate Family Variant Metrics\n")
        f.write(f"- Mean: {np.mean(new_errs):.2f}\n")
        f.write(f"- Median: {np.median(new_errs):.2f}\n")
        f.write(f"- P90: {np.percentile(new_errs, 90):.2f}\n")
        f.write(f"- Max: {np.max(new_errs):.2f}\n")
        f.write(f"- <= 1: {sum(1 for e in new_errs if e <= 1.0)}\n")
        f.write(f"- <= 5: {sum(1 for e in new_errs if e <= 5.0)}\n")
        f.write(f"- <= 10: {sum(1 for e in new_errs if e <= 10.0)}\n")
        f.write(f"- <= 25: {sum(1 for e in new_errs if e <= 25.0)}\n")
        f.write(f"- <= 50: {sum(1 for e in new_errs if e <= 50.0)}\n")
        f.write(f"- > 50: {sum(1 for e in new_errs if e > 50.0)}\n\n")
        
        f.write("## Experiment Analysis\n")
        
        recovered = sum(1 for r in results if r['Base Error'] > 50.0 and r['New Error'] <= 50.0)
        regressed = sum(1 for r in results if r['Base Error'] <= 50.0 and r['New Error'] > 50.0)
        gt_present_in_family = sum(1 for r in results if r['Base Error'] > 50.0 and r['GT In Family'])
        
        geom_fail = sum(1 for r in results if r['New Error'] > 50.0 and 'GEOMETRY' in r['New Cat'])
        ambig_fail = sum(1 for r in results if r['New Error'] > 50.0 and r['New Cat'] in ['TRUE_AMBIGUITY', 'PERIODIC_ALIAS'])
        
        f.write(f"A. How many previous periodic-alias failures become correct? {recovered}\n")
        f.write(f"B. How many previous successes regress? {regressed}\n")
        f.write(f"C. Does candidate-family retention actually recover the GT when it was already present among strong peaks?\n")
        f.write(f"   - GT was present in the retained family in {gt_present_in_family} failing cases.\n")
        f.write(f"   - Of those, {recovered} were successfully recovered by LowFreq tie-breaking.\n")
        f.write(f"D. How many failures are still caused by geometry? {geom_fail}\n")
        f.write(f"E. How many remain fundamentally ambiguous? {ambig_fail}\n")
        f.write(f"F. What is the new mean localization error? {np.mean(new_errs):.2f} px\n")
        f.write(f"G. What is the runtime impact? Average GSPE runtime is {np.mean(runtimes):.2f} ms per case.\n\n")
        
        f.write("NO PRODUCTION CODE WAS MODIFIED.\n")

if __name__ == '__main__':
    main()
