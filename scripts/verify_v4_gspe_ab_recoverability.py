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
    def __init__(self, scale_hypotheses=None, rotation_hypotheses=None, promotion_k=3, nms_mult=1.0):
        self.top_k = 5
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
            
            if best_res is None:
                best_res = np.full_like(res_centered, -1.0)
                best_scale_map = np.zeros_like(res_centered)
                best_rot_map = np.zeros_like(res_centered)
                best_w_map = np.zeros_like(res_centered, dtype=np.int32)
                best_h_map = np.zeros_like(res_centered, dtype=np.int32)
                
            mask = res_centered > best_res
            best_res[mask] = res_centered[mask]
            best_scale_map[mask] = scale
            best_rot_map[mask] = rot
            best_w_map[mask] = scaled_bound_w
            best_h_map[mask] = scaled_bound_h
            
        res = best_res
        
        boxes = []
        scores = []
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
                
                scores.append(float(max_val))
                w_cand = best_w_map[y, x]
                h_cand = best_h_map[y, x]
                scale_cand = best_scale_map[y, x]
                rot_cand = best_rot_map[y, x]
                
                tl_x = sub_x - (w_cand / 2.0)
                tl_y = sub_y - (h_cand / 2.0)
                
                boxes.append((tl_x, tl_y, int(w_cand), int(h_cand), scale_cand, rot_cand))
                
                nms_r_x = int(max(10, int(w_cand / 4.0)) * self.nms_mult)
                nms_r_y = int(max(10, int(h_cand / 4.0)) * self.nms_mult)
                    
                y1, y2 = max(0, y - nms_r_y), min(res_nms.shape[0], y + nms_r_y)
                x1, x2 = max(0, x - nms_r_x), min(res_nms.shape[1], x + nms_r_x)
                res_nms[y1:y2, x1:x2] = -1.0

        return {
            'boxes': boxes,
            'scores': scores,
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

def main():
    print("V4 GSPE A/B RECOVERABILITY SIMULATION...")
    
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v4'))
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports'))
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    
    ensure_dir(reports_dir)
    ensure_dir(docs_dir)
    
    csv_out_path = os.path.join(reports_dir, 'GSPE_AB_RECOVERABILITY.csv')
    md_out_path = os.path.join(docs_dir, 'GSPE_AB_RECOVERABILITY_REPORT.md')
    
    base_scales = [8.0, 9.0, 10.0, 11.0, 12.0]
    base_rots = [-5.0, 0.0, 5.0]
    dense_scales = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
    dense_rots = [-5.0, -2.5, 0.0, 2.5, 5.0]
    
    variants = [
        {"name": "Baseline", "scales": base_scales, "rots": base_rots, "k": 3, "nms": 1.0},
        {"name": "Dense Scale", "scales": dense_scales, "rots": base_rots, "k": 3, "nms": 1.0},
        {"name": "Dense Rot", "scales": base_scales, "rots": dense_rots, "k": 3, "nms": 1.0},
        {"name": "Dense Scale+Rot", "scales": dense_scales, "rots": dense_rots, "k": 3, "nms": 1.0},
        {"name": "Promotion K=5", "scales": base_scales, "rots": base_rots, "k": 5, "nms": 1.0},
        {"name": "Promotion K=10", "scales": base_scales, "rots": base_rots, "k": 10, "nms": 1.0},
        {"name": "Promotion K=15", "scales": base_scales, "rots": base_rots, "k": 15, "nms": 1.0},
        {"name": "NMS 0.5x", "scales": base_scales, "rots": base_rots, "k": 3, "nms": 0.5},
        {"name": "NMS 0.75x", "scales": base_scales, "rots": base_rots, "k": 3, "nms": 0.75},
        {"name": "NMS 1.25x", "scales": base_scales, "rots": base_rots, "k": 3, "nms": 1.25},
        {"name": "NMS 1.5x", "scales": base_scales, "rots": base_rots, "k": 3, "nms": 1.5},
        {"name": "Combined (Dense SR + K=15 + NMS 0.5x)", "scales": dense_scales, "rots": dense_rots, "k": 15, "nms": 0.5}
    ]
    
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
            
    print(f"Loaded {len(cases)} positive cases. Running {len(variants)} configurations...")
    
    variant_results = {v['name']: [] for v in variants}
    csv_rows = []
    
    for idx, c in enumerate(cases):
        print(f"[{idx+1}/{len(cases)}] Evaluating {c['id']} across all configs...")
        
        gt_x = c['meta']['gt_x']
        gt_y = c['meta']['gt_y']
        gt_scale = c['meta']['effective_scale']
        gt_rot = c['meta']['rotation_degrees']
        pitch_x = c['meta'].get('pitch_x', 0)
        pitch_y = c['meta'].get('pitch_y', 0)
        
        for v in variants:
            gspe = SimulationGSPE(scale_hypotheses=v['scales'], rotation_hypotheses=v['rots'], promotion_k=v['k'], nms_mult=v['nms'])
            res = gspe.run({'reference': c['ref'], 'search': c['search']})
            
            if len(res['boxes']) == 0:
                err = 9999.0
                gspe_x, gspe_y = 0, 0
                cat = "SCORING_FAILURE"
            else:
                top_box = res['boxes'][0]
                gspe_x = top_box[0] + top_box[2]/2.0
                gspe_y = top_box[1] + top_box[3]/2.0
                err = distance((gt_x, gt_y), (gspe_x, gspe_y))
                
                ix = min(max(int(round(gt_x)), 0), res['res_hybrid'].shape[1]-1)
                iy = min(max(int(round(gt_y)), 0), res['res_hybrid'].shape[0]-1)
                sx = min(max(int(round(gspe_x)), 0), res['res_hybrid'].shape[1]-1)
                sy = min(max(int(round(gspe_y)), 0), res['res_hybrid'].shape[0]-1)
                
                hyb_gt = res['res_hybrid'][iy, ix]
                hyb_sel = res['res_hybrid'][sy, sx]
                nms_suppressed = res['res_nms'][iy, ix] == -1.0
                
                cat = classify(err, gt_scale, gt_rot, res['coarse_hypotheses'], gspe_x, gspe_y, gt_x, gt_y, pitch_x, pitch_y, hyb_gt, hyb_sel, nms_suppressed, v['k'])
            
            variant_results[v['name']].append({
                'err': err,
                'cat': cat
            })
            
            if v['name'] == 'Baseline':
                csv_rows.append({
                    'Case ID': c['id'],
                    'GT X': gt_x,
                    'GT Y': gt_y,
                    'Baseline Error': err,
                    'Baseline Category': cat
                })
            else:
                csv_rows[-1][f"{v['name']} Error"] = err
                csv_rows[-1][f"{v['name']} Category"] = cat
                
    headers = list(csv_rows[0].keys())
    with open(csv_out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)
            
    with open(md_out_path, 'w') as f:
        f.write("# GSPE A/B Recoverability Report\n\n")
        
        f.write("## Tested Configurations\n")
        f.write("- **Current GSPE Grid**: Scales [8,9,10,11,12], Rots [-5,0,5], K=3, NMS=1.0x (dynamic ~1/4 crop)\n")
        f.write("- **Dense Scale**: Scales [8,8.5,9,9.5,10,10.5,11,11.5,12]\n")
        f.write("- **Dense Rot**: Rots [-5,-2.5,0,2.5,5]\n")
        f.write("- **Promotion K Sweeps**: K=5, 10, 15\n")
        f.write("- **NMS Multipliers**: 0.5x, 0.75x, 1.25x, 1.5x\n\n")
        
        f.write("## Aggregate Metrics\n\n")
        f.write("| Variant | Mean | Median | P90 | Max | <=5 | <=10 | <=25 | <=50 | >50 |\n")
        f.write("|---------|------|--------|-----|-----|-----|------|------|------|-----|\n")
        
        for v in variants:
            vname = v['name']
            errs = [r['err'] for r in variant_results[vname]]
            mean = np.mean(errs)
            med = np.median(errs)
            p90 = np.percentile(errs, 90)
            mx = np.max(errs)
            le5 = sum(1 for e in errs if e <= 5.0)
            le10 = sum(1 for e in errs if e <= 10.0)
            le25 = sum(1 for e in errs if e <= 25.0)
            le50 = sum(1 for e in errs if e <= 50.0)
            gt50 = sum(1 for e in errs if e > 50.0)
            
            f.write(f"| {vname} | {mean:.2f} | {med:.2f} | {p90:.2f} | {mx:.2f} | {le5} | {le10} | {le25} | {le50} | {gt50} |\n")
            
        f.write("\n## Failure Classification\n\n")
        f.write("| Variant | Geometry Recovered | NMS Recovered | Periodic Alias Remaining | True Ambiguity | Total Success |\n")
        f.write("|---------|--------------------|---------------|--------------------------|----------------|---------------|\n")
        
        base_fails = {r['cat']: 0 for r in variant_results['Baseline']}
        for r in variant_results['Baseline']:
            base_fails[r['cat']] += 1
            
        for v in variants:
            vname = v['name']
            cats = [r['cat'] for r in variant_results[vname]]
            periodic = cats.count('PERIODIC_ALIAS')
            ambig = cats.count('TRUE_AMBIGUITY')
            succ = cats.count('SUCCESS')
            
            geom_fails = cats.count('GEOMETRY_COARSE_RANK_FAILURE') + cats.count('GEOMETRY_NOT_COVERED')
            nms_fails = cats.count('NMS_FAILURE')
            
            base_geom = base_fails.get('GEOMETRY_COARSE_RANK_FAILURE', 0) + base_fails.get('GEOMETRY_NOT_COVERED', 0)
            base_nms = base_fails.get('NMS_FAILURE', 0)
            
            geom_rec = max(0, base_geom - geom_fails)
            nms_rec = max(0, base_nms - nms_fails)
            
            f.write(f"| {vname} | {geom_rec} | {nms_rec} | {periodic} | {ambig} | {succ} |\n")
            
        f.write("\n## FINAL CONCLUSION\n\n")
        
        best_v = 'Combined (Dense SR + K=15 + NMS 0.5x)'
        best_errs = [r['err'] for r in variant_results[best_v]]
        best_mean = np.mean(best_errs)
        
        geom_only_mean = min(np.mean([r['err'] for r in variant_results['Dense Scale']]),
                             np.mean([r['err'] for r in variant_results['Dense Rot']]),
                             np.mean([r['err'] for r in variant_results['Dense Scale+Rot']]))
                             
        k_only_mean = min(np.mean([r['err'] for r in variant_results['Promotion K=5']]),
                          np.mean([r['err'] for r in variant_results['Promotion K=15']]))
                          
        nms_only_mean = min(np.mean([r['err'] for r in variant_results['NMS 0.5x']]),
                            np.mean([r['err'] for r in variant_results['NMS 1.5x']]))
        
        ans_1 = f"{'Yes' if geom_only_mean < 50.0 else 'No'}. Best geometry-only mean error: {geom_only_mean:.2f} px."
        ans_2 = f"{'Yes' if k_only_mean < 50.0 else 'No'}. Best promotion-K-only mean error: {k_only_mean:.2f} px."
        ans_3 = f"{'Yes' if nms_only_mean < 50.0 else 'No'}. Best NMS-only mean error: {nms_only_mean:.2f} px."
        ans_4 = f"{'Yes' if best_mean < 50.0 else 'No'}. Best combined mean error: {best_mean:.2f} px."
        ans_5 = f"{sum(1 for e in best_errs if e > 50.0)} cases >50px remain in the best GSPE-only config."
        
        best_cats = [r['cat'] for r in variant_results[best_v]]
        ans_6 = f"{best_cats.count('PERIODIC_ALIAS') + best_cats.count('TRUE_AMBIGUITY')} cases are true periodic or contextual ambiguities."
        ans_7 = f"{best_cats.count('PERIODIC_ALIAS') + best_cats.count('TRUE_AMBIGUITY')} fundamentally require larger reference context."
        
        f.write(f"1. Can geometry-search changes alone bring mean error below 50 px?\n{ans_1}\n\n")
        f.write(f"2. Can promotion-K changes alone bring mean error below 50 px?\n{ans_2}\n\n")
        f.write(f"3. Can NMS changes alone bring mean error below 50 px?\n{ans_3}\n\n")
        f.write(f"4. Can their combination bring mean error below 50 px?\n{ans_4}\n\n")
        f.write(f"5. How many >50 px failures remain after the best GSPE-only configuration?\n{ans_5}\n\n")
        f.write(f"6. Of the remaining failures, how many are true periodic/context ambiguities?\n{ans_6}\n\n")
        f.write(f"7. How many genuinely require larger reference context?\n{ans_7}\n\n")
        
        f.write("NO PRODUCTION CODE WAS MODIFIED.\n")

if __name__ == '__main__':
    main()
