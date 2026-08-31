import os
import sys
import json
import cv2
import numpy as np
import csv
import scipy.ndimage as ndimage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logger import Profiler
from dataset.generator import HackathonDatasetGenerator

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class SimulationGSPE_Oracle:
    def __init__(self):
        self.top_k = 5
        self.scale_hypotheses = [8.0, 9.0, 10.0, 11.0, 12.0]
        self.rotation_hypotheses = [-5.0, 0.0, 5.0]
        self.promotion_k = 3

    def get_correlation_map(self, ref_img, search_img, scale, rot):
        h_ref, w_ref = ref_img.shape
        search_h, search_w = search_img.shape
        
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
            return None, None, None, 0, 0
            
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
        res_lf_centered = cv2.copyMakeBorder(res_lowfreq, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=-1.0)
        
        return res_centered, res_raw_centered, res_lf_centered, scaled_bound_w, scaled_bound_h

    def run_standard(self, inputs: dict) -> dict:
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
            
        for scale, rot in final_hypotheses:
            res_centered, _, _, scaled_bound_w, scaled_bound_h = self.get_correlation_map(ref_img, search_img, scale, rot)
            if res_centered is None: continue
            
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
                w_cand = best_w_map[y, x]
                h_cand = best_h_map[y, x]
                scale_cand = best_scale_map[y, x]
                rot_cand = best_rot_map[y, x]
                
                boxes.append({
                    'x': x, 'y': y,
                    'scale': scale_cand, 'rot': rot_cand,
                    'hyb_score': float(max_val)
                })
                
                nms_r_x = int(max(10, int(w_cand / 4.0)))
                nms_r_y = int(max(10, int(h_cand / 4.0)))
                    
                y1, y2 = max(0, y - nms_r_y), min(res_nms.shape[0], y + nms_r_y)
                x1, x2 = max(0, x - nms_r_x), min(res_nms.shape[1], x + nms_r_x)
                res_nms[y1:y2, x1:x2] = -1.0

        return {
            'boxes': boxes,
            'res_hybrid': res,
            'res_nms': res_nms
        }

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

def classify_oracle(err, std_res_hybrid, std_res_nms, gt_x, gt_y, highest_peak_dist_gt_geom, num_stronger_peaks, nearest_stronger_is_periodic):
    if err <= 50.0:
        return "SUCCESS"
        
    ix = min(max(int(round(gt_x)), 0), std_res_hybrid.shape[1]-1)
    iy = min(max(int(round(gt_y)), 0), std_res_hybrid.shape[0]-1)
    
    # Did GT geometry map fail to produce a peak near GT?
    if highest_peak_dist_gt_geom > 20.0:
        return "GEOMETRY_FAILURE"
        
    # Is the GT center missing from the standard composite map entirely? (i.e. score == -1.0)
    if std_res_hybrid[iy, ix] < 0:
        return "CORRECT_PEAK_MISSING"
        
    # Is GT peak in standard map but suppressed by NMS?
    nms_suppressed = (std_res_nms[iy, ix] == -1.0)
    if nms_suppressed and num_stronger_peaks == 0:
        return "NMS_ONLY_FAILURE"
        
    if num_stronger_peaks > 0:
        if nearest_stronger_is_periodic:
            return "CORRECT_PEAK_EXISTS_BUT_LOSES_TO_ALIAS"
        else:
            return "CORRECT_PEAK_EXISTS_BUT_LOSES_TO_SCORING"
            
    return "TRUE_AMBIGUITY"

def main():
    print("V4 GSPE 3000-CONTEXT ORACLE DIAGNOSTIC...")
    
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v4'))
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports'))
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    
    ensure_dir(reports_dir)
    ensure_dir(docs_dir)
    
    csv_out_path = os.path.join(reports_dir, 'GSPE_3000CONTEXT_ORACLE_DIAGNOSTIC.csv')
    md_out_path = os.path.join(docs_dir, 'GSPE_3000CONTEXT_ORACLE_DIAGNOSTIC_REPORT.md')
    
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
            
    context_size = 3000
    
    csv_rows = []
    
    for idx, c in enumerate(cases):
        meta = c['meta']
        gt_x = meta['gt_x']
        gt_y = meta['gt_y']
        gt_scale = meta['effective_scale']
        gt_rot = meta['rotation_degrees']
        pitch_x = meta.get('pitch_x', 0) / 10.0
        pitch_y = meta.get('pitch_y', 0) / 10.0
        
        ref_img = extract_reference_size(meta, context_size)
        
        gspe = SimulationGSPE_Oracle()
        # 1. Run Standard
        std_res = gspe.run_standard({'reference': ref_img, 'search': c['search']})
        
        if len(std_res['boxes']) == 0:
            continue
            
        top_box = std_res['boxes'][0]
        err = distance((gt_x, gt_y), (top_box['x'], top_box['y']))
        
        if err <= 50.0:
            continue
            
        print(f"Analyzing {c['id']} (>50px error)...")
        
        # 2. Run Oracle (force GT geometry)
        res_gt, res_raw_gt, res_lf_gt, gt_w_cand, gt_h_cand = gspe.get_correlation_map(ref_img, c['search'], gt_scale, gt_rot)
        
        ix = min(max(int(round(gt_x)), 0), res_gt.shape[1]-1)
        iy = min(max(int(round(gt_y)), 0), res_gt.shape[0]-1)
        
        gt_hyb_at_gt_center = float(res_gt[iy, ix])
        gt_raw_at_gt_center = float(res_raw_gt[iy, ix])
        gt_lf_at_gt_center = float(res_lf_gt[iy, ix])
        
        # Max under GT geometry
        _, max_val_gt_geom, _, max_loc_gt_geom = cv2.minMaxLoc(res_gt)
        highest_peak_dist_gt_geom = distance((gt_x, gt_y), max_loc_gt_geom)
        
        # Rank and stronger peaks
        rank_gt = np.sum(res_gt > gt_hyb_at_gt_center) + 1
        
        # Find local maxima stronger than GT
        local_max = (ndimage.maximum_filter(res_gt, size=5) == res_gt) & (res_gt > gt_hyb_at_gt_center)
        ys, xs = np.where(local_max)
        num_stronger_peaks = len(xs)
        
        nearest_stronger_dist = -1.0
        nearest_stronger_is_periodic = False
        if num_stronger_peaks > 0:
            dists = np.sqrt((xs - gt_x)**2 + (ys - gt_y)**2)
            nearest_idx = np.argmin(dists)
            nearest_stronger_dist = float(np.min(dists))
            nearest_x = xs[nearest_idx]
            nearest_y = ys[nearest_idx]
            
            dx = abs(nearest_x - gt_x)
            dy = abs(nearest_y - gt_y)
            mod_x = 0
            mod_y = 0
            if pitch_x > 0 and pitch_y > 0:
                mod_x = min(dx % pitch_x, pitch_x - (dx % pitch_x))
                mod_y = min(dy % pitch_y, pitch_y - (dy % pitch_y))
                nearest_stronger_is_periodic = (mod_x < 20 and mod_y < 20)
                
        nms_suppressed = (std_res['res_nms'][iy, ix] == -1.0)
        gt_survives_before_nms = (std_res['res_hybrid'][iy, ix] > 0)
        
        cls = classify_oracle(err, std_res['res_hybrid'], std_res['res_nms'], gt_x, gt_y, highest_peak_dist_gt_geom, num_stronger_peaks, nearest_stronger_is_periodic)
        
        row = {
            'case_id': c['id'],
            'gt_center': f"({gt_x:.2f}, {gt_y:.2f})",
            'gspe_center': f"({top_box['x']:.2f}, {top_box['y']:.2f})",
            'error': err,
            'gt_hyb_at_gt_center': gt_hyb_at_gt_center,
            'gt_raw_at_gt_center': gt_raw_at_gt_center,
            'gt_lf_at_gt_center': gt_lf_at_gt_center,
            'highest_peak_under_gt_geom': float(max_val_gt_geom),
            'highest_peak_dist_gt_geom': highest_peak_dist_gt_geom,
            'rank_of_gt_center': rank_gt,
            'num_stronger_peaks': num_stronger_peaks,
            'nearest_stronger_dist': nearest_stronger_dist,
            'nearest_stronger_is_periodic': nearest_stronger_is_periodic,
            'gt_survives_before_nms': gt_survives_before_nms,
            'gt_removed_by_nms': nms_suppressed,
            'sel_hyb': top_box['hyb_score'],
            'failure_category': cls
        }
        csv_rows.append(row)
        
    headers = list(csv_rows[0].keys()) if csv_rows else []
    with open(csv_out_path, 'w', newline='') as f:
        if headers:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
                
    with open(md_out_path, 'w') as f:
        f.write("# GSPE 3000-Context Oracle Diagnostic Report\n\n")
        f.write(f"Total evaluated cases (>50px error): {len(csv_rows)}\n\n")
        
        cats = [r['failure_category'] for r in csv_rows]
        f.write("## Oracle Classifications\n")
        f.write(f"- CORRECT_PEAK_EXISTS_BUT_LOSES_TO_ALIAS: {cats.count('CORRECT_PEAK_EXISTS_BUT_LOSES_TO_ALIAS')}\n")
        f.write(f"- CORRECT_PEAK_EXISTS_BUT_LOSES_TO_SCORING: {cats.count('CORRECT_PEAK_EXISTS_BUT_LOSES_TO_SCORING')}\n")
        f.write(f"- NMS_ONLY_FAILURE: {cats.count('NMS_ONLY_FAILURE')}\n")
        f.write(f"- GEOMETRY_FAILURE: {cats.count('GEOMETRY_FAILURE')}\n")
        f.write(f"- CORRECT_PEAK_MISSING: {cats.count('CORRECT_PEAK_MISSING')}\n")
        f.write(f"- TRUE_AMBIGUITY: {cats.count('TRUE_AMBIGUITY')}\n")
        
        f.write("\n## Per-Case Breakdown\n")
        f.write("| Case ID | Category | GT Hyb Score | Max GT Geom Score | Num Stronger Peaks | NMS Suppressed | Nearest Alias Dist |\n")
        f.write("|---------|----------|--------------|-------------------|--------------------|----------------|--------------------|\n")
        
        for r in csv_rows:
            f.write(f"| {r['case_id']} | {r['failure_category']} | {r['gt_hyb_at_gt_center']:.4f} | {r['highest_peak_under_gt_geom']:.4f} | {r['num_stronger_peaks']} | {r['gt_removed_by_nms']} | {r['nearest_stronger_dist']:.2f} |\n")

if __name__ == '__main__':
    main()
