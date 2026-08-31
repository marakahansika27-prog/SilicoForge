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
    def __init__(self, mode="baseline", scale_hypotheses=None, rotation_hypotheses=None):
        self.top_k = 5
        self.mode = mode
        
        self.base_scales = [8.0, 9.0, 10.0, 11.0, 12.0]
        self.base_rots = [-5.0, 0.0, 5.0]
        
        self.scale_hypotheses = scale_hypotheses if scale_hypotheses is not None else self.base_scales
        self.rotation_hypotheses = rotation_hypotheses if rotation_hypotheses is not None else self.base_rots
        self.promotion_k = 3

    def get_coarse_score(self, scale, rot, search_coarse, ref_coarse):
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
        
        if c_scaled_bound_w <= search_coarse.shape[1] and c_scaled_bound_h <= search_coarse.shape[0] and c_scaled_bound_w > 0 and c_scaled_bound_h > 0:
            c_ref_scaled = cv2.resize(c_ref_rotated_cropped, (c_scaled_bound_w, c_scaled_bound_h), interpolation=cv2.INTER_AREA)
            c_res = cv2.matchTemplate(search_coarse, c_ref_scaled, cv2.TM_CCOEFF_NORMED)
            _, c_max_val, _, _ = cv2.minMaxLoc(c_res)
            return c_max_val
        return -1.0

    def run(self, inputs: dict) -> dict:
        ref_img = inputs['reference']
        search_img = inputs['search']
        
        h_ref, w_ref = ref_img.shape
        search_h, search_w = search_img.shape
        
        best_res = None
        best_scale_map = None
        best_rot_map = None
        best_w_map = None
        best_h_map = None
        
        search_coarse = cv2.resize(search_img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ref_coarse = cv2.resize(ref_img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        
        if self.mode != "local_refinement":
            hypotheses = [(s, r) for s in self.scale_hypotheses for r in self.rotation_hypotheses]
            coarse_scores = [self.get_coarse_score(s, r, search_coarse, ref_coarse) for s, r in hypotheses]
            sorted_hypotheses = [x for _, x in sorted(zip(coarse_scores, hypotheses), reverse=True)]
            final_hypotheses = sorted_hypotheses[:self.promotion_k]
        else:
            # 1. Base coarse evaluation
            hypotheses = [(s, r) for s in self.base_scales for r in self.base_rots]
            coarse_scores = [self.get_coarse_score(s, r, search_coarse, ref_coarse) for s, r in hypotheses]
            sorted_hypotheses = [x for _, x in sorted(zip(coarse_scores, hypotheses), reverse=True)]
            top_coarse = sorted_hypotheses[:self.promotion_k]
            
            # 2. Local refinement around Top K at coarse resolution
            local_hyps = set()
            for s, r in top_coarse:
                for ds in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                    for dr in [-2.5, -1.25, 0.0, 1.25, 2.5]:
                        local_hyps.add((s + ds, r + dr))
            
            local_hyps = list(local_hyps)
            local_scores = [self.get_coarse_score(s, r, search_coarse, ref_coarse) for s, r in local_hyps]
            sorted_local = [x for _, x in sorted(zip(local_scores, local_hyps), reverse=True)]
            final_hypotheses = sorted_local[:self.promotion_k]
            
        # Full resolution evaluation
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
            
            if scaled_bound_w > search_w or scaled_bound_h > search_h or scaled_bound_w <= 0 or scaled_bound_h <= 0:
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
                
                nms_r_x = int(max(10, int(w_cand / 4.0)))
                nms_r_y = int(max(10, int(h_cand / 4.0)))
                    
                y1, y2 = max(0, y - nms_r_y), min(res_nms.shape[0], y + nms_r_y)
                x1, x2 = max(0, x - nms_r_x), min(res_nms.shape[1], x + nms_r_x)
                res_nms[y1:y2, x1:x2] = -1.0

        return {
            'boxes': boxes,
            'scores': scores,
            'res_hybrid': res
        }

def evaluate_variant(c, variant):
    with Profiler("gspe_run") as p:
        gspe = SimulationGSPE(
            mode=variant['mode'],
            scale_hypotheses=variant.get('scales'),
            rotation_hypotheses=variant.get('rots')
        )
        res = gspe.run({'reference': c['ref'], 'search': c['search']})
    
    runtime = p.elapsed_ms
    
    if len(res['boxes']) == 0:
        return 9999.0, 0, 0, runtime
        
    top_box = res['boxes'][0]
    gspe_x = top_box[0] + top_box[2]/2.0
    gspe_y = top_box[1] + top_box[3]/2.0
    sel_scale = top_box[4]
    sel_rot = top_box[5]
    
    err = distance((c['gt_x'], c['gt_y']), (gspe_x, gspe_y))
    return err, sel_scale, sel_rot, runtime

def classify_recovery(base_err, ref_err, base_s, base_r, ref_s, ref_r, gt_s, gt_r):
    geom_improved = abs(ref_s - gt_s) + abs(ref_r - gt_r) < abs(base_s - gt_s) + abs(base_r - gt_r) - 1e-4
    geom_unchanged = abs(ref_s - base_s) < 1e-4 and abs(ref_r - base_r) < 1e-4
    geom_regressed = abs(ref_s - gt_s) + abs(ref_r - gt_r) > abs(base_s - gt_s) + abs(base_r - gt_r) + 1e-4
    
    if geom_improved and ref_err < base_err - 1e-2:
        return "GEOMETRY_IMPROVED_AND_LOCALIZATION_IMPROVED"
    elif geom_improved and ref_err >= base_err - 1e-2:
        return "GEOMETRY_IMPROVED_BUT_LOCALIZATION_NOT_IMPROVED"
    elif geom_unchanged and ref_err < base_err - 1e-2:
        return "SPATIAL_RECOVERY_WITHOUT_GEOMETRY_CHANGE"
    elif geom_unchanged:
        return "GEOMETRY_UNCHANGED"
    elif geom_regressed:
        return "GEOMETRY_REGRESSED"
    
    if ref_err < base_err - 1e-2:
        return "SPATIAL_RECOVERY_WITHOUT_GEOMETRY_CHANGE"
    return "OTHER"

def main():
    print("V4 GSPE GEOMETRY REFINEMENT EXPERIMENT...")
    
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v4'))
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports'))
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    
    ensure_dir(reports_dir)
    ensure_dir(docs_dir)
    
    csv_out_path = os.path.join(reports_dir, 'GSPE_GEOMETRY_REFINEMENT.csv')
    md_out_path = os.path.join(docs_dir, 'GSPE_GEOMETRY_REFINEMENT_REPORT.md')
    
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
                'gt_x': meta['gt_x'],
                'gt_y': meta['gt_y'],
                'gt_scale': meta['effective_scale'],
                'gt_rot': meta['rotation_degrees'],
                'ref': cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE),
                'search': cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
            })
            
    variants = [
        {"name": "Baseline", "mode": "baseline", "scales": [8,9,10,11,12], "rots": [-5,0,5]},
        {"name": "Exp A (Scale Refine)", "mode": "dense", "scales": np.arange(8, 12.25, 0.25).tolist(), "rots": [-5,0,5]},
        {"name": "Exp B (Rot Refine)", "mode": "dense", "scales": [8,9,10,11,12], "rots": np.arange(-5, 6.25, 1.25).tolist()},
        {"name": "Exp C (Combined Dense)", "mode": "dense", "scales": np.arange(8, 12.25, 0.25).tolist(), "rots": np.arange(-5, 6.25, 1.25).tolist()},
        {"name": "Exp D (Local Refine)", "mode": "local_refinement"}
    ]
    
    print(f"Loaded {len(cases)} positive cases. Running configurations...")
    
    results = {v['name']: [] for v in variants}
    csv_rows = []
    
    for idx, c in enumerate(cases):
        print(f"[{idx+1}/{len(cases)}] {c['id']}")
        
        row = {
            'Case ID': c['id'],
            'GT Scale': c['gt_scale'],
            'GT Rot': c['gt_rot']
        }
        
        base_err, base_s, base_r, base_rt = evaluate_variant(c, variants[0])
        results['Baseline'].append({'err': base_err, 'scale': base_s, 'rot': base_r, 'rt': base_rt, 'class': "BASELINE"})
        row['Baseline Err'] = base_err
        row['Baseline Scale'] = base_s
        row['Baseline Rot'] = base_r
        
        for v in variants[1:]:
            err, s, r, rt = evaluate_variant(c, v)
            cls = classify_recovery(base_err, err, base_s, base_r, s, r, c['gt_scale'], c['gt_rot'])
            results[v['name']].append({'err': err, 'scale': s, 'rot': r, 'rt': rt, 'class': cls})
            
            row[f"{v['name']} Err"] = err
            row[f"{v['name']} Scale"] = s
            row[f"{v['name']} Rot"] = r
            row[f"{v['name']} Class"] = cls
            
        csv_rows.append(row)
        
    headers = list(csv_rows[0].keys())
    with open(csv_out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)
            
    with open(md_out_path, 'w') as f:
        f.write("# GSPE Geometry Refinement Report\n\n")
        
        f.write("## 1. Configuration Table\n")
        f.write("- **Baseline**: Scales [8,9,10,11,12], Rots [-5,0,5]\n")
        f.write("- **Exp A (Scale Refine)**: Scale step 0.25\n")
        f.write("- **Exp B (Rot Refine)**: Rot step 1.25\n")
        f.write("- **Exp C (Combined Dense)**: Scale step 0.25, Rot step 1.25\n")
        f.write("- **Exp D (Local Refine)**: Coarse top-3 -> local grid (scale +-0.5 step 0.25, rot +-2.5 step 1.25) -> top-3 full res\n\n")
        
        f.write("## 2. Accuracy Table\n")
        f.write("| Variant | Mean | Median | P90 | Max | <=1 | <=5 | <=10 | <=25 | <=50 | >50 |\n")
        f.write("|---------|------|--------|-----|-----|-----|-----|------|------|------|-----|\n")
        for v in variants:
            vn = v['name']
            errs = [r['err'] for r in results[vn]]
            f.write(f"| {vn} | {np.mean(errs):.2f} | {np.median(errs):.2f} | {np.percentile(errs, 90):.2f} | {np.max(errs):.2f} | ")
            f.write(f"{sum(1 for e in errs if e<=1.0)} | {sum(1 for e in errs if e<=5.0)} | {sum(1 for e in errs if e<=10.0)} | ")
            f.write(f"{sum(1 for e in errs if e<=25.0)} | {sum(1 for e in errs if e<=50.0)} | {sum(1 for e in errs if e>50.0)} |\n")
            
        f.write("\n## 3. Geometry Recovery & Classification\n")
        f.write("| Variant | Geom Improved & Loc Improved | Geom Improved, Loc Not | Geom Unchanged | Geom Regressed | Spatial Recovery Without Geom Change |\n")
        f.write("|---------|------------------------------|------------------------|----------------|----------------|--------------------------------------|\n")
        for v in variants[1:]:
            vn = v['name']
            cats = [r['class'] for r in results[vn]]
            c1 = cats.count("GEOMETRY_IMPROVED_AND_LOCALIZATION_IMPROVED")
            c2 = cats.count("GEOMETRY_IMPROVED_BUT_LOCALIZATION_NOT_IMPROVED")
            c3 = cats.count("GEOMETRY_UNCHANGED")
            c4 = cats.count("GEOMETRY_REGRESSED")
            c5 = cats.count("SPATIAL_RECOVERY_WITHOUT_GEOMETRY_CHANGE")
            f.write(f"| {vn} | {c1} | {c2} | {c3} | {c4} | {c5} |\n")
            
        f.write("\n## 4. Per-Case Regression/Recovery Counts\n")
        f.write("| Variant | Recovered (>50 to <=50) | Recovered to <=5 | Regressed (<=50 to >50) | Loc Regressed (>1px) | Loc Improved (>1px) | GT Geom Recovered |\n")
        f.write("|---------|-------------------------|------------------|-------------------------|----------------------|---------------------|-------------------|\n")
        base_errs = [r['err'] for r in results['Baseline']]
        for v in variants[1:]:
            vn = v['name']
            v_errs = [r['err'] for r in results[vn]]
            rec_50 = sum(1 for b, e in zip(base_errs, v_errs) if b > 50 and e <= 50)
            rec_5 = sum(1 for b, e in zip(base_errs, v_errs) if b > 50 and e <= 5)
            reg_50 = sum(1 for b, e in zip(base_errs, v_errs) if b <= 50 and e > 50)
            loc_reg = sum(1 for b, e in zip(base_errs, v_errs) if e > b + 1.0)
            loc_imp = sum(1 for b, e in zip(base_errs, v_errs) if e < b - 1.0)
            
            # GT geom recovered: Cases where Baseline geometry was inaccurate, but Refined is accurate
            b_scales = [r['scale'] for r in results['Baseline']]
            b_rots = [r['rot'] for r in results['Baseline']]
            v_scales = [r['scale'] for r in results[vn]]
            v_rots = [r['rot'] for r in results[vn]]
            gt_scales = [c['gt_scale'] for c in cases]
            gt_rots = [c['gt_rot'] for c in cases]
            
            gt_rec = 0
            for bs, br, vs, vr, gts, gtr in zip(b_scales, b_rots, v_scales, v_rots, gt_scales, gt_rots):
                b_diff = abs(bs - gts) + abs(br - gtr)
                v_diff = abs(vs - gts) + abs(vr - gtr)
                if b_diff > 0.5 and v_diff <= 0.5:
                    gt_rec += 1
                    
            f.write(f"| {vn} | {rec_50} | {rec_5} | {reg_50} | {loc_reg} | {loc_imp} | {gt_rec} |\n")
            
        f.write("\n## 5. Runtime Table\n")
        f.write("| Variant | Mean Runtime (ms/case) |\n")
        f.write("|---------|------------------------|\n")
        for v in variants:
            vn = v['name']
            rts = [r['rt'] for r in results[vn]]
            f.write(f"| {vn} | {np.mean(rts):.2f} |\n")
            
        f.write("\n## 6. Analysis of periodic-alias interaction\n")
        f.write("If 'Geom Improved, Loc Not' is high, it means refining geometry does NOT resolve periodic aliasing. ")
        f.write("Instead, the refined search simply latches onto a mathematically identical periodic peak with the new optimal geometry. ")
        f.write("If 'Recovered (>50 to <=50)' is very low compared to baseline failures, then geometry quantization is NOT the root cause of large >50px errors.\n")
        
        f.write("\n## 7. Recommended smallest mathematically justified refinement strategy\n")
        best_v = 'Exp D (Local Refine)'
        
        f.write("Based on the decision rule:\n")
        f.write("If Exp D (Local Refine) runtime is <1000ms (within 5s limit) and recovers the localization improvements of Exp C (Combined Dense), ")
        f.write("it represents the optimal balance of efficiency and continuous geometry search.\n")
        f.write("However, if the total successful cases (<=50px) across ALL configurations remains stubbornly around 45-50%, ")
        f.write("then geometry refinement fundamentally CANNOT resolve the remaining 50% periodic alias failures. ")
        f.write("In that scenario, geometry refinement provides marginal local benefit (<=5px improvements) but fails to break macroscopic ambiguity, ")
        f.write("requiring us to return to spatial context expansion.\n\n")
        
        f.write("NO PRODUCTION CODE WAS MODIFIED.\n")

if __name__ == '__main__':
    main()
