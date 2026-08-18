import os
import cv2
import json
import numpy as np
import pandas as pd
import math
import sys

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine

def calculate_distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def main():
    dataset_dir = os.path.join("dataset", "hackathon_v2")
    results_file = os.path.join("outputs", "hackathon_v2", "hackathon_results.csv")
    out_dir = os.path.join("outputs", "hackathon_v2", "candidate_recall")
    os.makedirs(out_dir, exist_ok=True)
    
    # Read current benchmark results
    current_results = {}
    if os.path.exists(results_file):
        df_res = pd.read_csv(results_file)
        for _, row in df_res.iterrows():
            current_results[row['case_id']] = row
            
    # Initialize engines
    ice = ImageConditioningEngine()
    gspe = GlobalSearchProposalEngine(
        top_k=50, 
        nms_radius=10.0,
        scale_hypotheses=[9.0, 9.5, 10.0, 10.5, 11.0],
        rotation_hypotheses=[-5.0, -2.5, 0.0, 2.5, 5.0]
    )
    
    k_values = [1, 3, 5, 10, 20, 50]
    
    recall_results = []
    
    # Track stats
    stats = {
        'total': 0,
        'current_success': 0,
        'current_failure': 0,
        'recall_5px': {k: 0 for k in k_values},
        'recall_10px': {k: 0 for k in k_values},
        'oracle_err': {k: [] for k in k_values},
        'oracle_success_10px': {k: 0 for k in k_values}
    }
    
    failure_forensics = []
    
    for arch in ["dram", "finfet"]:
        arch_dir = os.path.join(dataset_dir, arch)
        if not os.path.exists(arch_dir):
            continue
            
        for case_id in sorted(os.listdir(arch_dir)):
            case_dir = os.path.join(arch_dir, case_id)
            if not os.path.isdir(case_dir):
                continue
                
            print(f"Processing {case_id}...")
            
            with open(os.path.join(case_dir, "metadata.json"), 'r') as f:
                meta = json.load(f)
                
            gt_x = meta['gt_x']
            gt_y = meta['gt_y']
            difficulty = meta['difficulty']
            
            ref_img = cv2.imread(os.path.join(case_dir, "reference.png"), cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(os.path.join(case_dir, "search.png"), cv2.IMREAD_GRAYSCALE)
            
            cond = ice.run({'reference': ref_img, 'search': search_img})
            gspe_res = gspe.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
            
            # Extract centers
            candidates = []
            for i, box in enumerate(gspe_res['boxes']):
                x, y, w, h = box[0], box[1], box[2], box[3]
                cx = float(x) + float(w) / 2.0
                cy = float(y) + float(h) / 2.0
                dist = calculate_distance((cx, cy), (gt_x, gt_y))
                candidates.append({
                    'rank': i + 1,
                    'cx': cx,
                    'cy': cy,
                    'dist': dist
                })
                
            # Compute gt_rank_5px and gt_rank_10px
            gt_rank_5px = -1
            gt_rank_10px = -1
            for cand in candidates:
                if cand['dist'] <= 5.0 and gt_rank_5px == -1:
                    gt_rank_5px = cand['rank']
                if cand['dist'] <= 10.0 and gt_rank_10px == -1:
                    gt_rank_10px = cand['rank']
                    
            # Compute oracle errors
            oracle_errs = {}
            for k in k_values:
                top_k_cands = candidates[:k]
                if top_k_cands:
                    oracle_errs[k] = min(c['dist'] for c in top_k_cands)
                else:
                    oracle_errs[k] = 9999.9
                    
                stats['oracle_err'][k].append(oracle_errs[k])
                if oracle_errs[k] <= 10.0:
                    stats['oracle_success_10px'][k] += 1
                    
            # Update recall stats
            for k in k_values:
                if gt_rank_5px != -1 and gt_rank_5px <= k:
                    stats['recall_5px'][k] += 1
                if gt_rank_10px != -1 and gt_rank_10px <= k:
                    stats['recall_10px'][k] += 1
                    
            stats['total'] += 1
            
            gt_in_top1 = 1 if (gt_rank_10px != -1 and gt_rank_10px <= 1) else 0
            gt_in_top3 = 1 if (gt_rank_10px != -1 and gt_rank_10px <= 3) else 0
            gt_in_top5 = 1 if (gt_rank_10px != -1 and gt_rank_10px <= 5) else 0
            gt_in_top10 = 1 if (gt_rank_10px != -1 and gt_rank_10px <= 10) else 0
            gt_in_top20 = 1 if (gt_rank_10px != -1 and gt_rank_10px <= 20) else 0
            gt_in_top50 = 1 if (gt_rank_10px != -1 and gt_rank_10px <= 50) else 0
            
            res_row = {
                'case_id': case_id,
                'architecture': arch,
                'difficulty': difficulty,
                'gt_x': gt_x,
                'gt_y': gt_y,
                'top1_error': oracle_errs[1],
                'oracle_top3_error': oracle_errs[3],
                'oracle_top5_error': oracle_errs[5],
                'oracle_top10_error': oracle_errs[10],
                'oracle_top20_error': oracle_errs[20],
                'oracle_top50_error': oracle_errs[50],
                'gt_rank_5px': gt_rank_5px,
                'gt_rank_10px': gt_rank_10px,
                'gt_in_top1': gt_in_top1,
                'gt_in_top3': gt_in_top3,
                'gt_in_top5': gt_in_top5,
                'gt_in_top10': gt_in_top10,
                'gt_in_top20': gt_in_top20,
                'gt_in_top50': gt_in_top50
            }
            recall_results.append(res_row)
            
            # Current failure forensics
            if case_id in current_results:
                curr = current_results[case_id]
                err = curr['localization_error_px']
                if err <= 10.0:
                    stats['current_success'] += 1
                else:
                    stats['current_failure'] += 1
                    
                    if gt_rank_10px == -1 or gt_rank_10px > 10:
                        cat = "CATEGORY A: GT NOT in Top-10 within 10 px"
                    else:
                        cat = "CATEGORY B: GT IS in Top-10 within 10 px, selected other"
                        
                    failure_forensics.append({
                        'case_id': case_id,
                        'architecture': arch,
                        'difficulty': difficulty,
                        'current_error': err,
                        'current_pred_x': curr['pred_x'],
                        'current_pred_y': curr['pred_y'],
                        'gt_x': gt_x,
                        'gt_y': gt_y,
                        'gt_rank_5px': gt_rank_5px,
                        'gt_rank_10px': gt_rank_10px,
                        'oracle_top5_error': oracle_errs[5],
                        'oracle_top10_error': oracle_errs[10],
                        'oracle_top20_error': oracle_errs[20],
                        'oracle_top50_error': oracle_errs[50],
                        'failure_category': cat
                    })
                    
    df_recall = pd.DataFrame(recall_results)
    df_recall.to_csv(os.path.join(out_dir, "candidate_recall_results.csv"), index=False)
    
    if failure_forensics:
        df_failures = pd.DataFrame(failure_forensics)
        df_failures.to_csv(os.path.join(out_dir, "failure_forensics.csv"), index=False)
        
    print("\n========================================")
    print("V2 CANDIDATE RECALL FORENSIC")
    print("========================================")
    print(f"Total cases: {stats['total']}")
    print(f"Current successes <=10px: {stats['current_success']}")
    print(f"Current failures >10px: {stats['current_failure']}")
    
    print("\nGT Recall @ 5px:")
    for k in k_values:
        print(f"Top-{k}:\t{stats['recall_5px'][k]} / {stats['total']}")
        
    print("\nGT Recall @ 10px:")
    for k in k_values:
        print(f"Top-{k}:\t{stats['recall_10px'][k]} / {stats['total']}")
        
    print("\n========================================")
    print("CURRENT FAILURE BREAKDOWN")
    print("========================================")
    fails = len(failure_forensics)
    fail_cat_a = sum(1 for f in failure_forensics if "CATEGORY A" in f['failure_category'])
    fail_cat_b = sum(1 for f in failure_forensics if "CATEGORY B" in f['failure_category'])
    fail_gt_in_top20 = sum(1 for f in failure_forensics if f['gt_rank_10px'] != -1 and f['gt_rank_10px'] <= 20)
    
    print(f"Failures where GT NOT in Top-10:\n{fail_cat_a} / {fails}")
    print(f"Failures where GT IS in Top-10:\n{fail_cat_b} / {fails}")
    print(f"Failures where GT IS in Top-20:\n{fail_gt_in_top20} / {fails}")
    
    print("\n========================================")
    print("ORACLE PERFORMANCE")
    print("========================================")
    for k in k_values:
        mean_err = np.mean(stats['oracle_err'][k])
        print(f"Mean Oracle Error @ Top-{k}:\t{mean_err:.2f}")
        
    print("\nSuccess <=10px using oracle:")
    for k in [5, 10, 20, 50]:
        print(f"Top-{k}:\t{stats['oracle_success_10px'][k]}")
        
    print("========================================")

if __name__ == "__main__":
    main()
