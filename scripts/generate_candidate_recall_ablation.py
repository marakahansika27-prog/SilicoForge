import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def extract_top_k(res, top_k, nms_radius, new_w, new_h, gt_x, gt_y):
    res_nms = res.copy()
    candidates = []
    
    for i in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(res_nms)
        if max_val < -0.99:
            break
            
        x, y = max_loc
        delta_x, delta_y = 0.0, 0.0
        
        # Subpixel interpolation
        if 0 < x < res.shape[1] - 1 and 0 < y < res.shape[0] - 1:
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
                    
        c_x = x + delta_x + new_w / 2.0
        c_y = y + delta_y + new_h / 2.0
        
        err = np.linalg.norm(np.array([c_x, c_y]) - np.array([gt_x, gt_y]))
        candidates.append({
            'rank': i + 1,
            'x': c_x,
            'y': c_y,
            'score': float(max_val),
            'err': float(err)
        })
        
        y1, y2 = max(0, y - nms_radius), min(res_nms.shape[0], y + nms_radius)
        x1, x2 = max(0, x - nms_radius), min(res_nms.shape[1], x + nms_radius)
        res_nms[y1:y2, x1:x2] = -1.0
        
    return candidates

def main():
    print("JSON serialization hardening: ENABLED")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    disambiguation_csv = os.path.join(base_dir, 'benchmark', 'results', 'global_disambiguation', 'global_disambiguation_results.csv')
    deep_interior_dir = os.path.join(base_dir, 'benchmark', 'results', 'deep_interior_context')
    
    deep_interior_csv = os.path.join(deep_interior_dir, 'periodic_shift_analysis.csv')
    if not os.path.exists(deep_interior_csv):
        deep_interior_csv = os.path.join(deep_interior_dir, 'deep_interior_context_results.csv')
        
    out_dir = os.path.join(base_dir, 'benchmark', 'results', 'candidate_recall_ablation')
    ensure_dir(out_dir)
    
    if not (os.path.exists(disambiguation_csv) and os.path.exists(deep_interior_csv)):
        print("Error: Required source files missing.")
        return
        
    global_df = pd.read_csv(disambiguation_csv)
    deep_df = pd.read_csv(deep_interior_csv)
    
    metadata = deep_df[["case_id", "context_size", "D", "architecture", "difficulty"]].copy()
    
    case_metadata = {
        str(row["case_id"]): {
            "context_size": int(row["context_size"]),
            "D": float(row["D"]),
            "architecture": str(row["architecture"]),
            "difficulty": str(row["difficulty"]),
        }
        for _, row in metadata.iterrows()
    }
    
    results = []
    
    for case_id in global_df['case_id']:
        case_id = str(case_id)
        if case_id not in case_metadata:
            continue
            
        meta_info = case_metadata[case_id]
        ctx = meta_info["context_size"]
        D = meta_info["D"]
        architecture = meta_info["architecture"]
        difficulty = meta_info["difficulty"]
        
        case_dir = os.path.join(deep_interior_dir, f"context_{int(ctx)}", architecture, difficulty, case_id)
        ref_path = os.path.join(case_dir, "reference.png")
        search_path = os.path.join(case_dir, "search.png")
        meta_path = os.path.join(case_dir, "metadata.json")
        
        if not (os.path.exists(ref_path) and os.path.exists(search_path) and os.path.exists(meta_path)):
            continue
            
        with open(meta_path, 'r') as f:
            case_meta = json.load(f)
            
        gt_x = case_meta['gt_x']
        gt_y = case_meta['gt_y']
        
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        
        if ref_img is None or search_img is None:
            continue
            
        h_ref, w_ref = ref_img.shape
        scale_factor = 10
        new_h, new_w = h_ref // scale_factor, w_ref // scale_factor
        ref_scaled = cv2.resize(ref_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        res_raw = cv2.matchTemplate(search_img, ref_scaled, cv2.TM_CCOEFF_NORMED)
        search_blurred = cv2.GaussianBlur(search_img, (31, 31), 15)
        ref_blurred = cv2.GaussianBlur(ref_scaled, (31, 31), 15)
        res_lowfreq = cv2.matchTemplate(search_blurred, ref_blurred, cv2.TM_CCOEFF_NORMED)
        
        nms_radius = min(new_w, new_h) // 2
        
        # Candidate recall normally operates on the default Hybrid response
        res_hybrid = 0.5 * res_raw + 0.5 * res_lowfreq
        
        # We need K up to 50
        candidates = extract_top_k(res_hybrid, 50, nms_radius, new_w, new_h, gt_x, gt_y)
        if not candidates:
            continue
            
        top1_err = candidates[0]['err']
        nearest_err = min([c['err'] for c in candidates])
        
        gt_rank = None
        for c in candidates:
            if c['err'] <= 0.5:
                gt_rank = c['rank']
                break
                
        def in_top_k(k):
            return any(c['err'] <= 0.5 for c in candidates[:k])
            
        results.append({
            'case_id': case_id,
            'context_size': ctx,
            'D': D,
            'architecture': architecture,
            'difficulty': difficulty,
            'top1_error': top1_err,
            'nearest_distance': nearest_err,
            'gt_rank': gt_rank if gt_rank is not None else 999,
            'gt_in_top1': in_top_k(1),
            'gt_in_top3': in_top_k(3),
            'gt_in_top5': in_top_k(5),
            'gt_in_top10': in_top_k(10),
            'gt_in_top20': in_top_k(20),
            'gt_in_top50': in_top_k(50)
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "candidate_recall_results.csv"), index=False)
    
    total_cases = len(df)
    if total_cases == 0:
        print("No cases processed.")
        return
        
    recall_1 = float(df['gt_in_top1'].mean() * 100)
    recall_3 = float(df['gt_in_top3'].mean() * 100)
    recall_5 = float(df['gt_in_top5'].mean() * 100)
    recall_10 = float(df['gt_in_top10'].mean() * 100)
    recall_20 = float(df['gt_in_top20'].mean() * 100)
    recall_50 = float(df['gt_in_top50'].mean() * 100)
    
    summary = {
        'total_cases': total_cases,
        'recall_k1': recall_1,
        'recall_k3': recall_3,
        'recall_k5': recall_5,
        'recall_k10': recall_10,
        'recall_k20': recall_20,
        'recall_k50': recall_50,
        'mean_nearest_distance': float(df['nearest_distance'].mean())
    }
    
    with open(os.path.join(out_dir, "candidate_recall_summary.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary), f, indent=4)
        
    # Plotting
    ks = [1, 3, 5, 10, 20, 50]
    recalls = [recall_1, recall_3, recall_5, recall_10, recall_20, recall_50]
    
    plt.figure(figsize=(8,6))
    plt.plot(ks, recalls, marker='o', linewidth=2)
    plt.xlabel('Top-K Candidates')
    plt.ylabel('GT Recall (%)')
    plt.title('Candidate Recall vs K')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "recall_vs_k.png"), dpi=150)
    plt.close()
    
    # GT Rank Distribution
    ranks = df[df['gt_rank'] != 999]['gt_rank']
    plt.figure(figsize=(8,6))
    plt.hist(ranks, bins=range(1, 52), alpha=0.7, edgecolor='k')
    plt.xlabel('GT Rank in Candidate List')
    plt.ylabel('Number of Cases')
    plt.title('Distribution of Ground Truth Ranks (when present <= 50)')
    plt.grid(axis='y')
    plt.savefig(os.path.join(out_dir, "gt_rank_distribution.png"), dpi=150)
    plt.close()
    
    # Nearest Candidate Distance
    plt.figure(figsize=(8,6))
    plt.hist(df['nearest_distance'], bins=50, alpha=0.7, edgecolor='k')
    plt.xlabel('Distance from GT to Nearest Top-50 Candidate (px)')
    plt.ylabel('Number of Cases')
    plt.title('Nearest Candidate Distance Distribution')
    plt.grid(axis='y')
    plt.savefig(os.path.join(out_dir, "nearest_candidate_distance.png"), dpi=150)
    plt.close()
    
    def format_breakdown(group_col):
        res = f"### Breakdown by {group_col.capitalize()}\n"
        res += f"| {group_col.capitalize()} | K=1 | K=5 | K=50 | Mean Nearest Dist |\n"
        res += "|---|---|---|---|---|\n"
        for val in sorted(df[group_col].unique()):
            sub = df[df[group_col] == val]
            r1 = sub['gt_in_top1'].mean() * 100
            r5 = sub['gt_in_top5'].mean() * 100
            r50 = sub['gt_in_top50'].mean() * 100
            mnd = sub['nearest_distance'].mean()
            res += f"| {val} | {r1:.1f}% | {r5:.1f}% | {r50:.1f}% | {mnd:.1f} px |\n"
        return res + "\n"

    best_k_jump = ""
    max_jump = 0
    for i in range(len(ks)-1):
        jump = recalls[i+1] - recalls[i]
        if jump > max_jump:
            max_jump = jump
            best_k_jump = f"from K={ks[i]} to K={ks[i+1]}"

    report = f"""# Candidate Recall Ablation Report

## 1. Experimental Objective
Determine whether localization failures in the Global Disambiguation benchmark are caused by missing the GT during candidate generation (NMS recall) or by misranking the GT.

## 2. Global Candidate Recall
- **Total Cases Analyzed**: {total_cases}
- **K=1 Recall**: {recall_1:.1f}%
- **K=3 Recall**: {recall_3:.1f}%
- **K=5 Recall**: {recall_5:.1f}%
- **K=10 Recall**: {recall_10:.1f}%
- **K=20 Recall**: {recall_20:.1f}%
- **K=50 Recall**: {recall_50:.1f}%

## 3. Breakdown Tables
{format_breakdown('context_size')}
{format_breakdown('architecture')}
{format_breakdown('difficulty')}

## 4. Analytical Findings

**Q1: What percentage of cases contain GT at K=1?**
{recall_1:.1f}%

**Q2: K=3?**
{recall_3:.1f}%

**Q3: K=5?**
{recall_5:.1f}%

**Q4: K=10?**
{recall_10:.1f}%

**Q5: K=20?**
{recall_20:.1f}%

**Q6: K=50?**
{recall_50:.1f}%

**Q7: At what K does candidate recall substantially increase?**
The largest absolute increase in recall occurs {best_k_jump}.

**Q8: Is GT commonly absent from the candidate set?**
{"Yes, even at K=50 a significant percentage of cases miss the GT entirely." if recall_50 < 95.0 else "No, by K=50 the GT is almost always captured."}

**Q9: Is the problem candidate generation or candidate ranking?**
{"The problem is severely compounded by candidate ranking. The GT is often present in the list but fails to secure the K=1 spot." if (recall_50 - recall_1) > 20.0 else "The problem is primarily candidate generation, as expanding K does not recover the GT."}

**Q10: Does increasing K appear sufficient to solve the problem?**
{"Yes, increasing K to at least 10 or 20 recovers the majority of missed Ground Truths, provided they can be subsequently re-ranked." if recall_20 > 80.0 else "No, increasing K up to 50 still leaves a substantial fraction of cases unresolved."}
"""

    with open(os.path.join(out_dir, "CANDIDATE_RECALL_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
        
    print("========================================")
    print("CANDIDATE RECALL ABLATION COMPLETE")
    print("========================================")
    print(f"Cases analyzed: {total_cases}")
    print(f"K=1 recall: {recall_1:.1f}%")
    print(f"K=3 recall: {recall_3:.1f}%")
    print(f"K=5 recall: {recall_5:.1f}%")
    print(f"K=10 recall: {recall_10:.1f}%")
    print(f"K=20 recall: {recall_20:.1f}%")
    print(f"K=50 recall: {recall_50:.1f}%")
    print("========================================")

if __name__ == '__main__':
    main()
