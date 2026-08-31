import os
import sys
import time
import argparse
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.network import SNRN
from src.ai_refinement.phase2_dataset import Phase2EvaluationDataset
from src.ai_refinement.config import SNRNConfig
from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine
from src.feature_matching.gfee import GeometricFeatureExtractionEngine
from src.registration.srae import SpatialRegistrationAlignmentEngine
from src.localization.localization import ClassicalLocalization
from scripts.verify_utils import print_header, print_footer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/hackathon_v3",
                        help="Path to the real Phase 2 generated dataset.")
    return parser.parse_args()

def extract_snrn_patches(cond_ref, search_img, classical_coord, x, y, w, h):
    # 6. GEOMETRIC SCALE FIX (as per dataset logic)
    scale_ratio = 10.0
    w_ref, h_ref = cond_ref.shape[1], cond_ref.shape[0]
    scaled_w = int(round(w_ref / scale_ratio))
    scaled_h = int(round(h_ref / scale_ratio))
    ref_scaled = cv2.resize(cond_ref, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    
    def center_crop(img, size):
        h_img, w_img = img.shape
        ch, cw = size, size
        y_start = max(0, h_img//2 - ch//2)
        x_start = max(0, w_img//2 - cw//2)
        crop = img[y_start:y_start+ch, x_start:x_start+cw]
        if crop.shape[0] < ch or crop.shape[1] < cw:
            crop = cv2.copyMakeBorder(crop, 0, ch-crop.shape[0], 0, cw-crop.shape[1], cv2.BORDER_CONSTANT, value=0)
        return crop
        
    ref_patch = center_crop(ref_scaled, SNRNConfig.PATCH_SIZE).astype(np.float32) / 255.0
    
    local_class_x = classical_coord[0] - x
    local_class_y = classical_coord[1] - y
    
    M_extract = np.array([
        [1.0, 0.0, (SNRNConfig.PATCH_SIZE / 2.0) - local_class_x],
        [0.0, 1.0, (SNRNConfig.PATCH_SIZE / 2.0) - local_class_y]
    ], dtype=np.float32)
    
    cand_crop = search_img[int(y):int(y+h), int(x):int(x+w)]
    cand_patch_raw = cv2.warpAffine(
        cand_crop, M_extract, 
        (SNRNConfig.PATCH_SIZE, SNRNConfig.PATCH_SIZE), 
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    cand_patch = cand_patch_raw.astype(np.float32) / 255.0
    return ref_patch, cand_patch

def evaluate_case(case_info, ref_img, search_img, model, device):
    ice = ImageConditioningEngine()
    cond = ice.run({'reference': ref_img, 'search': search_img})
    
    gspe = GlobalSearchProposalEngine(top_k=1, scale_hypotheses=[8.0, 9.0, 10.0, 11.0, 12.0], rotation_hypotheses=[-5.0, 0.0, 5.0])
    gspe_res = gspe.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
    
    if not gspe_res['boxes']:
        return None
        
    x, y, w, h, scale_cand, rot_cand = gspe_res['boxes'][0]
    center_x = float(x) + float(w) / 2.0
    center_y = float(y) + float(h) / 2.0
    cand_crop = cond['search_cond'][int(y):int(y+h), int(x):int(x+w)]
    
    gfee = GeometricFeatureExtractionEngine()
    gfee_res = gfee.run({'reference': cond['reference_cond'], 'candidate': cand_crop})
    
    srae = SpatialRegistrationAlignmentEngine()
    srae_res = srae.run({
        'reference': cond['reference_cond'],
        'candidate': cand_crop,
        'kp1': gfee_res['kp1'],
        'kp2': gfee_res['kp2'],
        'matches': gfee_res['good_matches']
    })
    
    loc = ClassicalLocalization()
    if srae_res['stats']['inliers'] < 3:
        classical_coord = np.array([center_x, center_y], dtype=np.float32)
    else:
        mat_w = srae_res['affine_matrix'].copy()
        loc_res = loc.run({
            'affine_matrix': mat_w,
            'candidate_x': float(x),
            'candidate_y': float(y),
            'inliers': srae_res['stats']['inliers'],
            'ref_center_x': 500.0,
            'ref_center_y': 500.0
        })
        classical_coord = np.array([loc_res['dx'], loc_res['dy']], dtype=np.float32)
        
    ref_patch, cand_patch = extract_snrn_patches(cond['reference_cond'], cond['search_cond'], classical_coord, x, y, w, h)
    
    ref_tensor = torch.from_numpy(ref_patch).unsqueeze(0).unsqueeze(0).to(device)
    cand_tensor = torch.from_numpy(cand_patch).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(ref_tensor, cand_tensor)
        pred_delta = preds['residual'].cpu().numpy()[0]
        conf = preds['confidence'].cpu().numpy()[0][0]
        
    ai_coord = classical_coord + pred_delta
    
    return {
        'classical_coord': classical_coord,
        'ai_coord': ai_coord,
        'confidence': conf,
        'scale_cand': scale_cand,
        'rot_cand': rot_cand,
        'hybrid_score': gspe.stats.get('hybrid_top1_score', 0.0),
        'gspe_center_x': center_x,
        'gspe_center_y': center_y,
        'srae_inliers': srae_res['stats']['inliers'],
        'loc_failed': srae_res['stats']['inliers'] < 3
    }

def main():
    args = get_args()
    start_time = time.time()
    name = "INFERENCE"
    print_header(name)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SNRN().to(device)
        model.eval()
        
        ds = Phase2EvaluationDataset(dataset_dir=args.data_dir)
        
        print("================================")
        print("PHASE 2 REAL DATA SOURCE")
        print("================================")
        print(f"Dataset path: {args.data_dir}")
        print(f"Dataset class: {ds.__class__.__name__}")
        print(f"Total cases: {len(ds)}")
        print(f"Positive cases: {ds.positive_count}")
        print(f"Negative cases: {ds.negative_count}")
        print("================================\n")
        
        if len(ds) == 0:
            print("EXECUTION FAIL: Dataset is empty.")
            sys.exit(1)
            
        metrics = {
            'total_samples': 0,
            'DRAM_pos': 0, 'DRAM_neg': 0,
            'FinFET_pos': 0, 'FinFET_neg': 0,
            'c_errs': [], 'a_errs': [], 'gspe_errs': [],
            'gspe_correct': 0,
            'srae_failed': 0,
            'loc_failed': 0,
            'snrn_rejected': 0
        }
        
        for i in range(len(ds)):
            sample = ds[i]
            ci = sample['case_info']
            arch = ci['architecture']
            metrics['total_samples'] += 1
            
            res = evaluate_case(ci, sample['reference_img'], sample['search_img'], model, device)
            
            if ci['target_present']:
                if arch == 'dram': metrics['DRAM_pos'] += 1
                if arch == 'finfet': metrics['FinFET_pos'] += 1
                
                if res is None:
                    # GSPE failed to return boxes
                    metrics['c_errs'].append(1000.0)
                    metrics['a_errs'].append(1000.0)
                    metrics['gspe_errs'].append(1000.0)
                    metrics['srae_failed'] += 1
                    metrics['loc_failed'] += 1
                    metrics['snrn_rejected'] += 1
                else:
                    gt_coord = np.array([ci['gt_x'], ci['gt_y']], dtype=np.float32)
                    c_err = np.linalg.norm(gt_coord - res['classical_coord'])
                    a_err = np.linalg.norm(gt_coord - res['ai_coord'])
                    g_err = np.linalg.norm(gt_coord - np.array([res['gspe_center_x'], res['gspe_center_y']]))
                    
                    metrics['c_errs'].append(c_err)
                    metrics['a_errs'].append(a_err)
                    metrics['gspe_errs'].append(g_err)
                    
                    if g_err <= 50.0:
                        metrics['gspe_correct'] += 1
                    if res['srae_inliers'] < 3:
                        metrics['srae_failed'] += 1
                    if res['loc_failed']:
                        metrics['loc_failed'] += 1
                    if res['confidence'] < 0.5:
                        metrics['snrn_rejected'] += 1
            else:
                if arch == 'dram': metrics['DRAM_neg'] += 1
                if arch == 'finfet': metrics['FinFET_neg'] += 1
                    
        # Compute summary
        c_errs = np.array(metrics['c_errs'])
        a_errs = np.array(metrics['a_errs'])
        
        if len(c_errs) > 0:
            c_mean, c_med, c_max = np.mean(c_errs), np.median(c_errs), np.max(c_errs)
            a_mean, a_med, a_max = np.mean(a_errs), np.median(a_errs), np.max(a_errs)
            imp = ((c_mean - a_mean) / (c_mean + 1e-8)) * 100
            
            pct_1 = np.mean(c_errs <= 1.0) * 100
            pct_5 = np.mean(c_errs <= 5.0) * 100
            pct_10 = np.mean(c_errs <= 10.0) * 100
            pct_25 = np.mean(c_errs <= 25.0) * 100
            pct_50 = np.mean(c_errs <= 50.0) * 100
        else:
            c_mean = c_med = c_max = a_mean = a_med = a_max = imp = 0.0
            pct_1 = pct_5 = pct_10 = pct_25 = pct_50 = 0.0
            
        print("================================")
        print("PHASE 2 NUMERICAL BASELINE METRICS")
        print("================================")
        print(f"1. Samples evaluated: {metrics['total_samples']}")
        print(f"2. DRAM pos/neg: {metrics['DRAM_pos']} / {metrics['DRAM_neg']}")
        print(f"3. FinFET pos/neg: {metrics['FinFET_pos']} / {metrics['FinFET_neg']}")
        print(f"4. Classical mean error: {c_mean:.4f} px")
        print(f"5. Classical median error: {c_med:.4f} px")
        print(f"6. Classical max error: {c_max:.4f} px")
        print(f"7. AI/SNRN mean error: {a_mean:.4f} px")
        print(f"8. AI/SNRN median error: {a_med:.4f} px")
        print(f"9. AI/SNRN max error: {a_max:.4f} px")
        print(f"10. Improvement percentage: {imp:.2f} %")
        print(f"11. Samples within 1 px: {pct_1:.2f} %")
        print(f"12. Samples within 5 px: {pct_5:.2f} %")
        print(f"13. Samples within 10 px: {pct_10:.2f} %")
        print(f"14. Samples within 25 px: {pct_25:.2f} %")
        print(f"15. Samples within 50 px: {pct_50:.2f} %")
        print(f"16. GSPE candidate-vs-GT error (mean): {np.mean(metrics['gspe_errs']):.4f} px")
        print(f"17. GSPE selected correct target: {metrics['gspe_correct']}")
        print(f"18. SRAE failed / insufficient matches: {metrics['srae_failed']}")
        print(f"19. ClassicalLocalization failed: {metrics['loc_failed']}")
        print(f"20. SNRN confidence rejected prediction: {metrics['snrn_rejected']}")
        print("================================\n")
        
        print("EXECUTION STATUS: PASS")
        if a_mean < 50.0:
            print("ACCURACY STATUS: PASS (Mean Error < 50.0 px)")
            print_footer(name, start_time, True)
        else:
            print("ACCURACY STATUS: FAIL (Mean Error >= 50.0 px)")
            print_footer(name, start_time, False)
            sys.exit(1)
            
    except Exception as e:
        print_footer(name, start_time, False)
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
