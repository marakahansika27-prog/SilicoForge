import os
import sys
import time
import argparse
import csv
import torch
import cv2
import numpy as np
from scipy.stats import pearsonr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.network import SNRN
from src.ai_refinement.phase2_dataset import Phase2EvaluationDataset
from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine
from src.feature_matching.gfee import GeometricFeatureExtractionEngine
from src.registration.srae import SpatialRegistrationAlignmentEngine
from src.localization.localization import ClassicalLocalization
from scripts.verify_inference import extract_snrn_patches
from scripts.verify_utils import print_header, print_footer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/hackathon_v3")
    return parser.parse_args()

def safe_pearson(x, y):
    if len(x) < 2 or np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return 0.0
    return pearsonr(x, y)[0]

def evaluate_forensic_case(ci, ref_img, search_img, model, device):
    ice = ImageConditioningEngine()
    cond = ice.run({'reference': ref_img, 'search': search_img})
    
    gspe = GlobalSearchProposalEngine(top_k=1, scale_hypotheses=[8.0, 9.0, 10.0, 11.0, 12.0], rotation_hypotheses=[-5.0, 0.0, 5.0])
    gspe_res = gspe.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
    
    if not gspe_res['boxes']:
        return None
        
    x, y, w, h, scale_cand, rot_cand = gspe_res['boxes'][0]
    center_x = float(x) + float(w) / 2.0
    center_y = float(y) + float(h) / 2.0
    
    gspe_score = gspe_res['scores'][0]
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
    
    inliers = srae_res['stats']['inliers']
    affine_matrix = srae_res['affine_matrix']
    
    loc = ClassicalLocalization()
    if inliers < 3:
        classical_coord = np.array([center_x, center_y], dtype=np.float32)
    else:
        mat_w = affine_matrix.copy()
        loc_res = loc.run({
            'affine_matrix': mat_w,
            'candidate_x': float(x),
            'candidate_y': float(y),
            'inliers': inliers,
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
        conf = float(preds['confidence'].cpu().numpy()[0][0])
        
    ai_coord = classical_coord + pred_delta
    
    # Calculate GSPE Error directly
    gt_coord = np.array([ci['gt_x'], ci['gt_y']], dtype=np.float32)
    gspe_center = np.array([center_x, center_y], dtype=np.float32)
    
    gspe_err = float(np.linalg.norm(gt_coord - gspe_center))
    class_err = float(np.linalg.norm(gt_coord - classical_coord))
    ai_err = float(np.linalg.norm(gt_coord - ai_coord))
    
    if gspe_err <= 5.0:
        classification = "GSPE_CORRECT"
    elif gspe_err > 5.0 and inliers >= 3:
        classification = "GSPE_WRONG"
    elif inliers < 3:
        classification = "SRAE_FAILURE"
    else:
        classification = "LOCALIZATION_FAILURE"
        
    if conf < 0.5:
        classification = "SNRN_FAILURE"
        
    return {
        'case_id': ci['case_id'],
        'architecture': ci['architecture'],
        'gt_x': ci['gt_x'], 'gt_y': ci['gt_y'],
        'gspe_x': center_x, 'gspe_y': center_y,
        'gspe_err': gspe_err,
        'gspe_score': gspe_score,
        'scale_cand': scale_cand,
        'rot_cand': rot_cand,
        'crop_x': x, 'crop_y': y,
        'inliers': inliers,
        'matrix_str': str(affine_matrix.flatten().tolist()) if affine_matrix is not None else "None",
        'classical_x': classical_coord[0], 'classical_y': classical_coord[1],
        'class_err': class_err,
        'snrn_dx': float(pred_delta[0]), 'snrn_dy': float(pred_delta[1]),
        'ai_x': ai_coord[0], 'ai_y': ai_coord[1],
        'ai_err': ai_err,
        'classification': classification
    }

def print_metrics(arch, records):
    if not records:
        return
        
    gspe_errs = np.array([r['gspe_err'] for r in records])
    class_errs = np.array([r['class_err'] for r in records])
    ai_errs = np.array([r['ai_err'] for r in records])
    
    gspe_mean, gspe_med, gspe_max = np.mean(gspe_errs), np.median(gspe_errs), np.max(gspe_errs)
    gspe_p90 = np.percentile(gspe_errs, 90)
    
    pct_1 = np.mean(gspe_errs <= 1.0) * 100
    pct_5 = np.mean(gspe_errs <= 5.0) * 100
    pct_10 = np.mean(gspe_errs <= 10.0) * 100
    pct_25 = np.mean(gspe_errs <= 25.0) * 100
    pct_50 = np.mean(gspe_errs <= 50.0) * 100
    
    gspe_correct = sum(1 for r in records if r['gspe_err'] <= 5.0)
    srae_fail = sum(1 for r in records if r['inliers'] < 3)
    loc_fail = sum(1 for r in records if r['classification'] == 'LOCALIZATION_FAILURE')
    snrn_fail = sum(1 for r in records if r['classification'] == 'SNRN_FAILURE')
    
    corr_gc = safe_pearson(gspe_errs, class_errs)
    corr_ga = safe_pearson(gspe_errs, ai_errs)
    
    print(f"================================")
    print(f"{arch.upper()} GSPE METRICS")
    print(f"================================")
    print(f"GSPE Mean Error  : {gspe_mean:.4f} px")
    print(f"GSPE Median Error: {gspe_med:.4f} px")
    print(f"GSPE P90 Error   : {gspe_p90:.4f} px")
    print(f"GSPE Max Error   : {gspe_max:.4f} px")
    print(f"Accuracy <= 1px  : {pct_1:.2f} %")
    print(f"Accuracy <= 5px  : {pct_5:.2f} %")
    print(f"Accuracy <= 10px : {pct_10:.2f} %")
    print(f"Accuracy <= 25px : {pct_25:.2f} %")
    print(f"Accuracy <= 50px : {pct_50:.2f} %")
    print(f"GSPE Correct     : {gspe_correct} / {len(records)}")
    print(f"SRAE Failures    : {srae_fail}")
    print(f"Loc Failures     : {loc_fail}")
    print(f"SNRN Failures    : {snrn_fail}")
    print(f"Corr(GSPE, Class): {corr_gc:.4f}")
    print(f"Corr(GSPE, AI)   : {corr_ga:.4f}")
    print(f"================================\n")
    
    # Write to markdown file dynamically
    with open("docs/PHASE2_PIPELINE_FORENSIC_AUDIT.md", "a") as f:
        f.write(f"## {arch.upper()} GSPE METRICS\n")
        f.write(f"- GSPE Mean Error: {gspe_mean:.4f} px\n")
        f.write(f"- GSPE Median Error: {gspe_med:.4f} px\n")
        f.write(f"- GSPE P90 Error: {gspe_p90:.4f} px\n")
        f.write(f"- GSPE Accuracy <= 5px: {pct_5:.2f}%\n")
        f.write(f"- SRAE Failures: {srae_fail}\n")
        f.write(f"- Corr(GSPE, Class): {corr_gc:.4f}\n\n")

def main():
    args = get_args()
    print_header("PIPELINE FORENSIC AUDIT")
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNRN().to(device)
    model.eval()
    
    ds = Phase2EvaluationDataset(dataset_dir=args.data_dir)
    
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    # Reset report file
    with open("docs/PHASE2_PIPELINE_FORENSIC_AUDIT.md", "w") as f:
        f.write("# PHASE 2 PIPELINE FORENSIC AUDIT\n\n")
    
    csv_path = "outputs/reports/PHASE2_SAMPLE_FORENSICS.csv"
    records = []
    
    for i in range(len(ds)):
        sample = ds[i]
        ci = sample['case_info']
        if ci['target_present']:
            res = evaluate_forensic_case(ci, sample['reference_img'], sample['search_img'], model, device)
            if res:
                records.append(res)
                
    dram_records = [r for r in records if r['architecture'] == 'dram']
    finfet_records = [r for r in records if r['architecture'] == 'finfet']
    
    print_metrics('DRAM', dram_records)
    print_metrics('FinFET', finfet_records)
    
    # Save CSV
    if records:
        keys = records[0].keys()
        with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(records)
            
        print(f"Full CSV saved to: {csv_path}\n")
        
        # 10 Worst samples
        sorted_records = sorted(records, key=lambda x: x['class_err'], reverse=True)
        print("--- TOP 10 WORST SAMPLES (By Classical Error) ---")
        for i, r in enumerate(sorted_records[:10]):
            print(f"Rank {i+1}: Case {r['case_id']} ({r['architecture']}) | GSPE Err: {r['gspe_err']:.2f} | Class Err: {r['class_err']:.2f} | Class: {r['classification']}")
            
        # Bottleneck Decision
        gspe_errs = np.array([r['gspe_err'] for r in records])
        class_errs = np.array([r['class_err'] for r in records])
        ai_errs = np.array([r['ai_err'] for r in records])
        
        g_mean = np.mean(gspe_errs)
        c_mean = np.mean(class_errs)
        a_mean = np.mean(ai_errs)
        
        if g_mean > 10.0:
            bottleneck = "GSPE"
        elif g_mean <= 10.0 and c_mean > 10.0:
            bottleneck = "SRAE/ClassicalLocalization"
        elif c_mean <= 10.0 and a_mean > 10.0:
            bottleneck = "SNRN"
        else:
            bottleneck = "healthy / secondary issue"
            
        print(f"\n================================")
        print(f"BOTTLENECK DIAGNOSIS")
        print(f"================================")
        print(f"GSPE Mean: {g_mean:.2f} | Class Mean: {c_mean:.2f} | AI Mean: {a_mean:.2f}")
        print(f"DOMINANT BOTTLENECK -> {bottleneck}")
        
        with open("docs/PHASE2_PIPELINE_FORENSIC_AUDIT.md", "a") as f:
            f.write(f"## BOTTLENECK DIAGNOSIS\n")
            f.write(f"**Dominant Bottleneck:** {bottleneck}\n")
            f.write(f"GSPE Mean: {g_mean:.2f} px | Classical Mean: {c_mean:.2f} px\n")
            
    print_footer("PIPELINE FORENSIC AUDIT", start_time, True)

if __name__ == "__main__":
    main()
