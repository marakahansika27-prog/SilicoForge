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

# --- Raw-only GSPE Monkey-Patch ---
class RawGSPE(GlobalSearchProposalEngine):
    def run(self, inputs: dict) -> dict:
        ref_img = inputs['reference']
        search_img = inputs['search']
        
        search_h, search_w = search_img.shape
        h_ref, w_ref = ref_img.shape
        search_blurred = cv2.GaussianBlur(search_img, (31, 31), 15)
        
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
                
        if len(hypotheses) > 3:
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
            final_hypotheses = sorted_hypotheses[:3]
        else:
            final_hypotheses = hypotheses

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
            
            ref_rotated = cv2.warpAffine(ref_img, M_rot, (bound_w, bound_h), 
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                         
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
            
            ref_scaled = cv2.resize(ref_rotated_cropped, (scaled_bound_w, scaled_bound_h), interpolation=cv2.INTER_AREA)
            
            res_raw = cv2.matchTemplate(search_img, ref_scaled, cv2.TM_CCOEFF_NORMED)
            ref_blurred = cv2.GaussianBlur(ref_scaled, (31, 31), 15)
            res_lowfreq = cv2.matchTemplate(search_blurred, ref_blurred, cv2.TM_CCOEFF_NORMED)
            
            # --- THE ONLY CHANGE: Use pure res_raw ---
            res = res_raw.copy()
            
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
        res_raw = best_res_raw
        res_lowfreq = best_res_lowfreq
        
        boxes = []
        scores = []
        res_nms = res.copy()
        nms_radius = self.nms_radius if self.nms_radius is not None else 10
        
        for _ in range(self.top_k):
            _, max_val, _, max_loc = cv2.minMaxLoc(res_nms)
            if max_val < 0:
                break
                
            x, y = max_loc
            
            # --- SUBPIXEL INTERPOLATION (to match standard GSPE) ---
            delta_x = 0.0
            delta_y = 0.0
            if x > 0 and x < res.shape[1] - 1 and y > 0 and y < res.shape[0] - 1:
                fx_m1 = float(res[y, x - 1])
                fx_0  = float(res[y, x])
                fx_p1 = float(res[y, x + 1])
                denom_x = fx_m1 - 2 * fx_0 + fx_p1
                if abs(denom_x) > 1e-6:
                    dx = 0.5 * (fx_m1 - fx_p1) / denom_x
                    if not (np.isnan(dx) or np.isinf(dx)) and abs(dx) <= 0.5:
                        delta_x = dx
                fy_m1 = float(res[y - 1, x])
                fy_0  = float(res[y, x])
                fy_p1 = float(res[y + 1, x])
                denom_y = fy_m1 - 2 * fy_0 + fy_p1
                if abs(denom_y) > 1e-6:
                    dy = 0.5 * (fy_m1 - fy_p1) / denom_y
                    if not (np.isnan(dy) or np.isinf(dy)) and abs(dy) <= 0.5:
                        delta_y = dy
                        
            sub_x = x + delta_x
            sub_y = y + delta_y
            
            scores.append(float(max_val))
            w_cand = best_w_map[y, x]
            h_cand = best_h_map[y, x]
            scale_cand = best_scale_map[y, x]
            rot_cand = best_rot_map[y, x]
            
            tl_x = float(sub_x) - (w_cand / 2.0)
            tl_y = float(sub_y) - (h_cand / 2.0)
            
            boxes.append((tl_x, tl_y, int(w_cand), int(h_cand), scale_cand, rot_cand))
            
            nms_r_x = max(10, int(w_cand / 4.0)) if self.nms_radius is None else int(nms_radius)
            nms_r_y = max(10, int(h_cand / 4.0)) if self.nms_radius is None else int(nms_radius)
                
            y1, y2 = max(0, y - nms_r_y), min(res_nms.shape[0], y + nms_r_y)
            x1, x2 = max(0, x - nms_r_x), min(res_nms.shape[1], x + nms_r_x)
            res_nms[y1:y2, x1:x2] = -1.0
            
        return {
            'boxes': boxes,
            'scores': scores,
            'res_raw': res_raw,
            'res_lowfreq': res_lowfreq,
            'res_hybrid': res,
            'best_scale_map': best_scale_map,
            'best_rot_map': best_rot_map
        }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/hackathon_v3")
    return parser.parse_args()

def safe_extract(heatmap, x, y):
    ix, iy = int(round(x)), int(round(y))
    ix = max(0, min(ix, heatmap.shape[1] - 1))
    iy = max(0, min(iy, heatmap.shape[0] - 1))
    return float(heatmap[iy, ix])

def main():
    args = get_args()
    print_header("AB SCORING SIMULATION")
    start_time = time.time()
    
    ds = Phase2EvaluationDataset(dataset_dir=args.data_dir)
    scales = [8.0, 9.0, 10.0, 11.0, 12.0]
    rotations = [-5.0, 0.0, 5.0]
    
    gspe_hybrid = GlobalSearchProposalEngine(top_k=1, scale_hypotheses=scales, rotation_hypotheses=rotations)
    gspe_raw = RawGSPE(top_k=1, scale_hypotheses=scales, rotation_hypotheses=rotations)
    
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
        
        ice = ImageConditioningEngine()
        cond = ice.run({'reference': sample['reference_img'], 'search': sample['search_img']})
        
        # 1. Run GSPE Forced to GT Geometry
        gspe_gt = GlobalSearchProposalEngine(top_k=1, scale_hypotheses=[nearest_scale], rotation_hypotheses=[nearest_rot])
        res_gt = gspe_gt.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
        
        # Get scores exactly at the GT location from the GT-forced maps
        gt_raw_score = safe_extract(res_gt['res_raw'], gt_cx, gt_cy)
        gt_lowfreq_score = safe_extract(res_gt['res_lowfreq'], gt_cx, gt_cy)
        gt_hybrid_score = safe_extract(res_gt['res_hybrid'], gt_cx, gt_cy)
        
        # 2. Run Hybrid GSPE
        res_h = gspe_hybrid.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
        
        hyb_cx, hyb_cy = -1.0, -1.0
        hyb_err = -1.0
        hyb_scale, hyb_rot = -1.0, -1.0
        hyb_raw_score = -1.0
        hyb_lowfreq_score = -1.0
        hyb_hybrid_score = -1.0
        
        if res_h['boxes']:
            box = res_h['boxes'][0]
            hyb_cx = box[0] + box[2]/2.0
            hyb_cy = box[1] + box[3]/2.0
            hyb_scale, hyb_rot = box[4], box[5]
            hyb_err = float(np.linalg.norm([hyb_cx - gt_cx, hyb_cy - gt_cy]))
            
            hyb_raw_score = safe_extract(res_h['res_raw'], hyb_cx, hyb_cy)
            hyb_lowfreq_score = safe_extract(res_h['res_lowfreq'], hyb_cx, hyb_cy)
            hyb_hybrid_score = safe_extract(res_h['res_hybrid'], hyb_cx, hyb_cy)
            
        # 3. Run Raw GSPE
        res_r = gspe_raw.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
        
        raw_cx, raw_cy = -1.0, -1.0
        raw_err = -1.0
        raw_scale, raw_rot = -1.0, -1.0
        raw_raw_score = -1.0
        raw_lowfreq_score = -1.0
        raw_hybrid_score = -1.0
        
        if res_r['boxes']:
            box = res_r['boxes'][0]
            raw_cx = box[0] + box[2]/2.0
            raw_cy = box[1] + box[3]/2.0
            raw_scale, raw_rot = box[4], box[5]
            raw_err = float(np.linalg.norm([raw_cx - gt_cx, raw_cy - gt_cy]))
            
            raw_raw_score = safe_extract(res_r['res_raw'], raw_cx, raw_cy)
            raw_lowfreq_score = safe_extract(res_r['res_lowfreq'], raw_cx, raw_cy)
            # For Raw GSPE, res_hybrid IS res_raw, but we can compute what the hybrid score WOULD have been:
            raw_hybrid_score = 0.9 * raw_raw_score + 0.1 * raw_lowfreq_score
            
        records.append({
            'case_id': ci['case_id'],
            'gt_cx': gt_cx,
            'gt_cy': gt_cy,
            'hyb_cx': hyb_cx,
            'hyb_cy': hyb_cy,
            'hyb_err': hyb_err,
            'raw_cx': raw_cx,
            'raw_cy': raw_cy,
            'raw_err': raw_err,
            'hyb_scale': hyb_scale,
            'hyb_rot': hyb_rot,
            'raw_scale': raw_scale,
            'raw_rot': raw_rot,
            'gt_scale': gt_scale,
            'gt_rot': gt_rot,
            'gt_raw_score': gt_raw_score,
            'hyb_raw_score': hyb_raw_score,
            'raw_raw_score': raw_raw_score,
            'gt_lowfreq_score': gt_lowfreq_score,
            'hyb_lowfreq_score': hyb_lowfreq_score,
            'raw_lowfreq_score': raw_lowfreq_score,
            'gt_hybrid_score': gt_hybrid_score,
            'hyb_hybrid_score': hyb_hybrid_score,
            'raw_hybrid_score': raw_hybrid_score
        })
        
    csv_path = "outputs/reports/SIMULATE_AB_SCORING.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
        
    hyb_errs = [r['hyb_err'] for r in records if r['hyb_err'] != -1.0]
    raw_errs = [r['raw_err'] for r in records if r['raw_err'] != -1.0]
    
    def calc_stats(errs):
        errs = np.array(errs)
        if len(errs) == 0: return {}
        return {
            'mean': np.mean(errs),
            'median': np.median(errs),
            'p90': np.percentile(errs, 90),
            'max': np.max(errs),
            'pct_1': np.mean(errs <= 1.0) * 100,
            'pct_5': np.mean(errs <= 5.0) * 100,
            'pct_10': np.mean(errs <= 10.0) * 100,
            'pct_25': np.mean(errs <= 25.0) * 100,
            'pct_50': np.mean(errs <= 50.0) * 100,
            'pct_gt_50': np.mean(errs > 50.0) * 100,
        }
        
    hyb_stats = calc_stats(hyb_errs)
    raw_stats = calc_stats(raw_errs)
    
    improved = 0
    worsened = 0
    unchanged = 0
    
    for r in records:
        if r['raw_err'] < r['hyb_err'] - 1.0: # margin of 1px
            improved += 1
        elif r['raw_err'] > r['hyb_err'] + 1.0:
            worsened += 1
        else:
            unchanged += 1
            
    print("\n============================================================")
    print("A/B SCORING SIMULATION RESULTS")
    print("============================================================")
    print("METRIC                   | HYBRID (A)        | RAW ONLY (B)")
    print("-------------------------|-------------------|-------------------")
    for key in ['mean', 'median', 'p90', 'max']:
        print(f"{key.upper():<24} | {hyb_stats[key]:<17.4f} | {raw_stats[key]:<17.4f}")
    for key in ['pct_1', 'pct_5', 'pct_10', 'pct_25', 'pct_50', 'pct_gt_50']:
        print(f"{key:<24} | {hyb_stats[key]:<16.2f}% | {raw_stats[key]:<16.2f}%")
        
    print("\n------------------------------------------------------------")
    print(f"Number of cases improved by Raw-only : {improved}")
    print(f"Number of cases worsened by Raw-only : {worsened}")
    print(f"Number of cases unchanged            : {unchanged}")
    print("============================================================\n")
    
    with open("docs/SIMULATE_AB_SCORING_REPORT.md", "w") as f:
        f.write("# A/B SCORING SIMULATION REPORT\n\n")
        f.write("## 1. Aggregate Statistics\n")
        f.write("| Metric | Hybrid (A) | Raw Only (B) |\n")
        f.write("|--------|------------|--------------|\n")
        for key in ['mean', 'median', 'p90', 'max']:
            f.write(f"| {key.upper()} | {hyb_stats[key]:.4f} px | {raw_stats[key]:.4f} px |\n")
        for key in ['pct_1', 'pct_5', 'pct_10', 'pct_25', 'pct_50', 'pct_gt_50']:
            f.write(f"| {key} | {hyb_stats[key]:.2f}% | {raw_stats[key]:.2f}% |\n")
            
        f.write(f"\n## 2. Head-to-Head Comparison\n")
        f.write(f"- **Improved**: {improved}\n")
        f.write(f"- **Worsened**: {worsened}\n")
        f.write(f"- **Unchanged**: {unchanged}\n")

    print_footer("A/B SCORING SIMULATION", start_time, True)

if __name__ == "__main__":
    main()
