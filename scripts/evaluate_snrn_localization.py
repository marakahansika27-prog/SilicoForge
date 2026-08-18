import os
import sys
import json
import csv
import math
import time
import numpy as np
import cv2
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.coarse_search.gspe import GlobalSearchProposalEngine
from src.ai_refinement.network import SNRN
from src.utils.logger import Profiler

def center_crop(img, size):
    h_img, w_img = img.shape
    ch, cw = size, size
    y_start = max(0, h_img//2 - ch//2)
    x_start = max(0, w_img//2 - cw//2)
    crop = img[y_start:y_start+ch, x_start:x_start+cw]
    if crop.shape[0] < ch or crop.shape[1] < cw:
        crop = cv2.copyMakeBorder(crop, 0, ch-crop.shape[0], 0, cw-crop.shape[1], cv2.BORDER_CONSTANT, value=0)
    return crop

def main():
    print("Evaluating SNRN Localization Performance...")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v3'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'hackathon_v3', 'snrn_localization'))
    os.makedirs(out_dir, exist_ok=True)
    
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'checkpoints', 'best_model.pth'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SNRN().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    epoch = checkpoint.get("epoch", "Unknown")
    best_val_loss = checkpoint.get("best_val_loss", "Unknown")
    print("Loaded SNRN checkpoint:")
    print(f"  epoch = {epoch}")
    print(f"  best_val_loss = {best_val_loss}")    
    manifest_path = os.path.join(base_dir, 'dataset_manifest.csv')
    cases = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cases.append(row)
            
    gspe = GlobalSearchProposalEngine(top_k=20, nms_radius=10, scale_hypotheses=[10.0], rotation_hypotheses=[0.0])
    
    csv_data = []
    
    base_errors = []
    snrn_errors = []
    
    def calc_metrics(errors):
        if not errors: return 0, 0, 0, 0, 0, 0
        errs = np.array(errors)
        return (
            np.mean(errs),
            np.median(errs),
            np.sqrt(np.mean(errs**2)),
            np.percentile(errs, 90),
            np.percentile(errs, 95),
            np.mean(errs <= 10.0) * 100
        )
    
    total_time = 0
    
    for idx, c in enumerate(cases):
        case_id = c['case_id']
        arch = c['architecture']
        region = c['spatial_region']
        
        gt_x = float(c['gt_x'])
        gt_y = float(c['gt_y'])
        
        ref_path = os.path.join(base_dir, arch.lower(), case_id, 'reference.png')
        search_path = os.path.join(base_dir, arch.lower(), case_id, 'search.png')
        
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        
        t0 = time.time()
        
        # Baseline GSPE
        state = gspe.run({'reference': ref_img, 'search': search_img})
        
        # SNRN Input Prep
        w_ref, h_ref = ref_img.shape[1], ref_img.shape[0]
        scaled_w = int(round(w_ref / 10.0))
        scaled_h = int(round(h_ref / 10.0))
        ref_scaled = cv2.resize(ref_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        ref_patch = center_crop(ref_scaled, 128).astype(np.float32) / 255.0
        ref_t = torch.from_numpy(ref_patch).unsqueeze(0).unsqueeze(0).to(device)
        
        best_snrn_conf = -1
        best_snrn_coord = None
        base_coord = None
        
        snrn_ranks = []
        
        if state['boxes']:
            # Baseline is Top-1 from GSPE
            b0 = state['boxes'][0]
            base_coord = (b0[0] + b0[2]/2.0, b0[1] + b0[3]/2.0)
            
            for rank, (cand_box, cand_score) in enumerate(zip(state['boxes'], state['scores'])):
                tl_x, tl_y, w, h, _, _ = cand_box
                cx = tl_x + w/2.0
                cy = tl_y + h/2.0
                
                M_ext = np.array([
                    [1.0, 0.0, 64.0 - cx],
                    [0.0, 1.0, 64.0 - cy]
                ], dtype=np.float32)
                cand_patch_raw = cv2.warpAffine(search_img, M_ext, (128, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                cand_patch = cand_patch_raw.astype(np.float32) / 255.0
                cand_t = torch.from_numpy(cand_patch).unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    preds = model(ref_t, cand_t)
                    conf = float(preds['confidence'].item())
                    res_dx, res_dy = preds['residual'][0].cpu().numpy()
                    
                refined_cx = cx + res_dx
                refined_cy = cy + res_dy
                
                snrn_ranks.append({
                    'rank': rank + 1,
                    'cx': refined_cx,
                    'cy': refined_cy,
                    'conf': conf
                })
                
                if conf > best_snrn_conf:
                    best_snrn_conf = conf
                    best_snrn_coord = (refined_cx, refined_cy)
        
        t1 = time.time()
        total_time += (t1 - t0)
        
        if base_coord:
            base_err = math.sqrt((base_coord[0] - gt_x)**2 + (base_coord[1] - gt_y)**2)
        else:
            base_err = 1000.0
            
        if best_snrn_coord:
            snrn_err = math.sqrt((best_snrn_coord[0] - gt_x)**2 + (best_snrn_coord[1] - gt_y)**2)
        else:
            snrn_err = 1000.0
            
        # Top-K checks for SNRN
        snrn_sorted = sorted(snrn_ranks, key=lambda x: x['conf'], reverse=True)
        snrn_top_errs = []
        for r in snrn_sorted:
            err = math.sqrt((r['cx'] - gt_x)**2 + (r['cy'] - gt_y)**2)
            snrn_top_errs.append(err)
            
        top1 = 1 if snrn_top_errs and snrn_top_errs[0] <= 10.0 else 0
        top5 = 1 if any(e <= 10.0 for e in snrn_top_errs[:5]) else 0
        top10 = 1 if any(e <= 10.0 for e in snrn_top_errs[:10]) else 0
        top20 = 1 if any(e <= 10.0 for e in snrn_top_errs[:20]) else 0
            
        base_errors.append(base_err)
        snrn_errors.append(snrn_err)
        
        csv_data.append({
            'case_id': case_id, 'arch': arch, 'region': region,
            'base_error': base_err, 'snrn_error': snrn_err,
            'snrn_top1': top1, 'snrn_top5': top5, 'snrn_top10': top10, 'snrn_top20': top20
        })
        
        print(f"[{idx+1}/60] {case_id} ({region}) | Base Error: {base_err:.2f} px | SNRN Error: {snrn_err:.2f} px")

    # CSV write
    with open(os.path.join(out_dir, 'snrn_localization_results.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        w.writeheader()
        w.writerows(csv_data)
        
    b_mean, b_med, b_rmse, b_p90, b_p95, b_succ = calc_metrics(base_errors)
    s_mean, s_med, s_rmse, s_p90, s_p95, s_succ = calc_metrics(snrn_errors)
    
    with open(os.path.join(out_dir, 'SNRN_LOCALIZATION_REPORT.md'), 'w', encoding='utf-8') as f:
        f.write("# SNRN Localization Report\n\n")
        
        f.write("### BASELINE\n")
        f.write(f"Mean Error: {b_mean:.2f} px\n")
        f.write(f"Median Error: {b_med:.2f} px\n")
        f.write(f"RMSE: {b_rmse:.2f} px\n")
        f.write(f"P90: {b_p90:.2f} px\n")
        f.write(f"P95: {b_p95:.2f} px\n")
        f.write(f"Success @10px: {b_succ:.1f}%\n\n")
        
        f.write("### SNRN\n")
        f.write(f"Mean Error: {s_mean:.2f} px\n")
        f.write(f"Median Error: {s_med:.2f} px\n")
        f.write(f"RMSE: {s_rmse:.2f} px\n")
        f.write(f"P90: {s_p90:.2f} px\n")
        f.write(f"P95: {s_p95:.2f} px\n")
        f.write(f"Success @10px: {s_succ:.1f}%\n\n")
        
        f.write("### IMPROVEMENT\n")
        f.write(f"Mean Error reduction: {b_mean - s_mean:+.2f} px\n")
        f.write(f"Median Error reduction: {b_med - s_med:+.2f} px\n")
        f.write(f"RMSE reduction: {b_rmse - s_rmse:+.2f} px\n")
        f.write(f"Success@10px improvement: {s_succ - b_succ:+.1f}%\n\n")
        
        f.write("### Runtime\n")
        f.write(f"Total Inference Time: {total_time:.2f}s (Avg {total_time/len(cases):.3f}s/case)\n\n")
        
        f.write("### Region Breakdown (SNRN)\n")
        f.write("| Region | Mean Error (px) | Success @10px |\n")
        f.write("|---|---|---|\n")
        for reg in ['center', 'interior', 'left_boundary', 'right_boundary', 'top_boundary', 'bottom_boundary', 'top_left_corner', 'bottom_right_corner', 'random']:
            r_errs = [row['snrn_error'] for row in csv_data if reg in row['region']]
            if not r_errs: continue
            r_mean, _, _, _, _, r_succ = calc_metrics(r_errs)
            f.write(f"| {reg} | {r_mean:.2f} | {r_succ:.1f}% |\n")
            
        f.write("\n### Architecture Breakdown (SNRN)\n")
        f.write("| Architecture | Mean Error (px) | Success @10px |\n")
        f.write("|---|---|---|\n")
        for arch in ['DRAM', 'FinFET']:
            r_errs = [row['snrn_error'] for row in csv_data if row['arch'] == arch]
            if not r_errs: continue
            r_mean, _, _, _, _, r_succ = calc_metrics(r_errs)
            f.write(f"| {arch} | {r_mean:.2f} | {r_succ:.1f}% |\n")
            
        f.write("\n### Recall Metrics (SNRN Confidence Ranking)\n")
        top1 = np.mean([row['snrn_top1'] for row in csv_data]) * 100
        top5 = np.mean([row['snrn_top5'] for row in csv_data]) * 100
        top10 = np.mean([row['snrn_top10'] for row in csv_data]) * 100
        top20 = np.mean([row['snrn_top20'] for row in csv_data]) * 100
        f.write(f"Top-1: {top1:.1f}%\n")
        f.write(f"Top-5: {top5:.1f}%\n")
        f.write(f"Top-10: {top10:.1f}%\n")
        f.write(f"Top-20: {top20:.1f}%\n")

    print("\nSNRN Localization Evaluation Complete.")
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
