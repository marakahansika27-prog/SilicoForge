import os
import sys
import json
import cv2
import numpy as np
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logger import Profiler
from dataset.generator import HackathonDatasetGenerator

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class SimulationGSPE:
    def __init__(self):
        self.top_k = 5
        self.scale_hypotheses = [8.0, 9.0, 10.0, 11.0, 12.0]
        self.rotation_hypotheses = [-5.0, 0.0, 5.0]
        self.promotion_k = 3

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
        
        coarse_scores = []
        hypotheses = []
        for scale in self.scale_hypotheses:
            for rot in self.rotation_hypotheses:
                hypotheses.append((scale, rot))
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
                    coarse_scores.append(c_max_val)
                else:
                    coarse_scores.append(-1.0)
                    
        sorted_hypotheses = [x for _, x in sorted(zip(coarse_scores, hypotheses), reverse=True)]
        final_hypotheses = sorted_hypotheses[:self.promotion_k]
        coarse_data = sorted(zip(coarse_scores, hypotheses), reverse=True)
            
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
                
                boxes.append((tl_x, tl_y, int(w_cand), int(h_cand), scale_cand, rot_cand, float(max_val)))
                
                nms_r_x = int(max(10, int(w_cand / 4.0)))
                nms_r_y = int(max(10, int(h_cand / 4.0)))
                    
                y1, y2 = max(0, y - nms_r_y), min(res_nms.shape[0], y + nms_r_y)
                x1, x2 = max(0, x - nms_r_x), min(res_nms.shape[1], x + nms_r_x)
                res_nms[y1:y2, x1:x2] = -1.0

        return {
            'boxes': boxes,
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

def extract_reference_size(meta, ref_size):
    gen = HackathonDatasetGenerator(seed=meta['seed'])
    base_img_A, start_x, start_y, end_x, end_y, pitch_x, pitch_y, feature_w, feature_h, edge_strength = gen._generate_base_image(meta['architecture'], gen.layout_rng)
    
    base_cx = meta['reference_origin_x']
    base_cy = meta['reference_origin_y']
    rot = meta['rotation_degrees']
    scale = meta['augmentation_scale']
    
    M = cv2.getRotationMatrix2D((ref_size/2.0, ref_size/2.0), rot, scale)
    M[0, 2] += (base_cx - ref_size/2.0)
    M[1, 2] += (base_cy - ref_size/2.0)
    
    ref_float = cv2.warpAffine(base_img_A, M, (ref_size, ref_size), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    gen.ref_noise_rng = np.random.RandomState(meta['seed'] + 1)
    ref_noise_level = meta.get('reference_noise_level', 20.0)
    ref_noise = gen.ref_noise_rng.poisson(ref_float / 255.0 * ref_noise_level) / ref_noise_level * 255
    ref_img = np.clip(ref_float + ref_noise - 128, 0, 255).astype(np.uint8)
    
    return ref_img

def main():
    print("V4 GSPE CONTEXT RECOVERABILITY EXPERIMENT...")
    
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v4'))
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports'))
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    
    ensure_dir(reports_dir)
    ensure_dir(docs_dir)
    
    csv_out_path = os.path.join(reports_dir, 'GSPE_CONTEXT_RECOVERABILITY.csv')
    md_out_path = os.path.join(docs_dir, 'GSPE_CONTEXT_RECOVERABILITY_REPORT.md')
    
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
            
            search_path = os.path.join(case_dir, "search.png")
            cases.append({
                'id': case_name,
                'meta': meta,
                'search': cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
            })
            
    # Sizes in base pixels:
    # 1000 = baseline (100x100 search space)
    # 1500 = 1.5x (150x150 search space)
    # 2000 = 2.0x (200x200 search space)
    # 2500 = 2.5x (250x250 search space)
    # 3000 = 3.0x (300x300 search space)
    context_sizes = [1000, 1500, 2000, 2500, 3000]
    
    print(f"Loaded {len(cases)} positive cases. Running context sizes: {context_sizes}")
    
    results_by_size = {sz: [] for sz in context_sizes}
    csv_rows = []
    
    baseline_errors = {}
    
    for idx, c in enumerate(cases):
        print(f"[{idx+1}/{len(cases)}] {c['id']}")
        
        meta = c['meta']
        gt_x = meta['gt_x']
        gt_y = meta['gt_y']
        gt_scale = meta['effective_scale']
        gt_rot = meta['rotation_degrees']
        pitch_x = meta.get('pitch_x', 0)
        pitch_y = meta.get('pitch_y', 0)
        
        for sz in context_sizes:
            ref_img = extract_reference_size(meta, sz)
            
            with Profiler("gspe_run") as p:
                gspe = SimulationGSPE()
                res = gspe.run({'reference': ref_img, 'search': c['search']})
                
            runtime = p.elapsed_ms
            
            if len(res['boxes']) == 0:
                err = 9999.0
                cls = "FAILURE"
                sel_scale = 0
                sel_rot = 0
            else:
                top_box = res['boxes'][0]
                gspe_x = top_box[0] + top_box[2]/2.0
                gspe_y = top_box[1] + top_box[3]/2.0
                sel_scale = top_box[4]
                sel_rot = top_box[5]
                hyb_sel = top_box[6]
                
                err = distance((gt_x, gt_y), (gspe_x, gspe_y))
                
                ix = min(max(int(round(gt_x)), 0), res['res_hybrid'].shape[1]-1)
                iy = min(max(int(round(gt_y)), 0), res['res_hybrid'].shape[0]-1)
                hyb_gt = res['res_hybrid'][iy, ix]
                nms_supp = res['res_nms'][iy, ix] == -1.0
                
                cls = classify(err, gt_scale, gt_rot, res['coarse_hypotheses'], gspe_x, gspe_y, gt_x, gt_y, pitch_x, pitch_y, hyb_gt, hyb_sel, nms_supp, 3)
                
            if sz == 1000:
                baseline_errors[c['id']] = err
                
            results_by_size[sz].append({
                'err': err,
                'cls': cls,
                'rt': runtime
            })
            
            csv_rows.append({
                'case_id': c['id'],
                'context_size': sz,
                'gt_center': f"({gt_x:.2f}, {gt_y:.2f})",
                'gspe_center': f"({gspe_x:.2f}, {gspe_y:.2f})" if len(res['boxes']) > 0 else "N/A",
                'error': err,
                'gt_scale': gt_scale,
                'gt_rotation': gt_rot,
                'selected_scale': sel_scale,
                'selected_rotation': sel_rot,
                'classification': cls,
                'runtime_ms': runtime
            })
            
    headers = list(csv_rows[0].keys())
    with open(csv_out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)
            
    with open(md_out_path, 'w') as f:
        f.write("# GSPE Context Recoverability Report\n\n")
        
        f.write("## 1. Accuracy by Context Size\n")
        f.write("| Context Size | Mean | Median | P90 | Max | <=1 | <=5 | <=10 | <=25 | <=50 | >50 |\n")
        f.write("|--------------|------|--------|-----|-----|-----|-----|------|------|------|-----|\n")
        
        for sz in context_sizes:
            errs = [r['err'] for r in results_by_size[sz]]
            f.write(f"| {sz} ({(sz//10)}x{(sz//10)}) | {np.mean(errs):.2f} | {np.median(errs):.2f} | {np.percentile(errs, 90):.2f} | {np.max(errs):.2f} | ")
            f.write(f"{sum(1 for e in errs if e<=1.0)} | {sum(1 for e in errs if e<=5.0)} | {sum(1 for e in errs if e<=10.0)} | ")
            f.write(f"{sum(1 for e in errs if e<=25.0)} | {sum(1 for e in errs if e<=50.0)} | {sum(1 for e in errs if e>50.0)} |\n")
            
        f.write("\n## 2. Failure Recoveries and Regressions (Relative to 1000 Baseline)\n")
        f.write("| Context Size | Baseline Failures Recovered | Baseline Successes Regressed |\n")
        f.write("|--------------|-----------------------------|------------------------------|\n")
        
        for sz in context_sizes[1:]:
            rec = 0
            reg = 0
            for r_base, r_curr in zip(results_by_size[1000], results_by_size[sz]):
                if r_base['err'] > 50.0 and r_curr['err'] <= 50.0:
                    rec += 1
                elif r_base['err'] <= 50.0 and r_curr['err'] > 50.0:
                    reg += 1
            f.write(f"| {sz} | {rec} | {reg} |\n")
            
        f.write("\n## 3. Failure Classifications (>50px)\n")
        f.write("| Context Size | PERIODIC_ALIAS | TRUE_AMBIGUITY | GEOMETRY_COARSE_RANK_FAILURE | GEOMETRY_NOT_COVERED | NMS_FAILURE | SCORING_FAILURE | OTHER |\n")
        f.write("|--------------|----------------|----------------|------------------------------|----------------------|-------------|-----------------|-------|\n")
        
        for sz in context_sizes:
            cats = [r['cls'] for r in results_by_size[sz] if r['err'] > 50.0]
            c1 = cats.count("PERIODIC_ALIAS")
            c2 = cats.count("TRUE_AMBIGUITY")
            c3 = cats.count("GEOMETRY_COARSE_RANK_FAILURE")
            c4 = cats.count("GEOMETRY_NOT_COVERED")
            c5 = cats.count("NMS_FAILURE")
            c6 = cats.count("SCORING_FAILURE")
            c7 = len(cats) - sum([c1, c2, c3, c4, c5, c6])
            f.write(f"| {sz} | {c1} | {c2} | {c3} | {c4} | {c5} | {c6} | {c7} |\n")
            
        f.write("\n## 4. Runtime per case (ms)\n")
        f.write("| Context Size | Mean Runtime |\n")
        f.write("|--------------|--------------|\n")
        for sz in context_sizes:
            rts = [r['rt'] for r in results_by_size[sz]]
            f.write(f"| {sz} | {np.mean(rts):.2f} |\n")
            
        f.write("\n## Conclusion\n")
        f.write("Evaluate whether larger context successfully disambiguates periodic aliases without geometry modifications.\n")

if __name__ == '__main__':
    main()
