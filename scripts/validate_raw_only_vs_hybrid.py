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
            'score': max_val,
            'err': err
        })
        
        y1, y2 = max(0, y - nms_radius), min(res_nms.shape[0], y + nms_radius)
        x1, x2 = max(0, x - nms_radius), min(res_nms.shape[1], x + nms_radius)
        res_nms[y1:y2, x1:x2] = -1.0
        
    return candidates

def get_coordinate_scores(res, px, py):
    h, w = res.shape
    ix = int(round(px))
    iy = int(round(py))
    ix = max(0, min(w - 1, ix))
    iy = max(0, min(h - 1, iy))
    return float(res[iy, ix])

def plot_bar_comparison(val05, val10, labels, title, ylabel, filename, out_dir):
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, val05, width, label='Alpha 0.5 (Hybrid)')
    rects2 = ax.bar(x + width/2, val10, width, label='Alpha 1.0 (Raw)')
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()

def main():
    print("JSON serialization hardening: ENABLED")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    disambiguation_csv = os.path.join(base_dir, 'benchmark', 'results', 'global_disambiguation', 'global_disambiguation_results.csv')
    deep_interior_dir = os.path.join(base_dir, 'benchmark', 'results', 'deep_interior_context')
    
    deep_interior_csv = os.path.join(deep_interior_dir, 'periodic_shift_analysis.csv')
    if not os.path.exists(deep_interior_csv):
        deep_interior_csv = os.path.join(deep_interior_dir, 'deep_interior_context_results.csv')
        
    out_dir = os.path.join(base_dir, 'benchmark', 'results', 'raw_only_validation')
    ensure_dir(out_dir)
    
    if not os.path.exists(disambiguation_csv) or not os.path.exists(deep_interior_csv):
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
            print(f"Missing metadata for {case_id}")
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
        gt_map_x = gt_x - new_w / 2.0
        gt_map_y = gt_y - new_h / 2.0
        
        for alpha in [0.5, 1.0]:
            res_hybrid = alpha * res_raw + (1.0 - alpha) * res_lowfreq
            
            candidates = extract_top_k(res_hybrid, 5, nms_radius, new_w, new_h, gt_x, gt_y)
            if not candidates:
                continue
                
            top1 = candidates[0]
            gt_score = get_coordinate_scores(res_hybrid, gt_map_x, gt_map_y)
            
            gt_in_top1 = top1['err'] <= 0.5
            gt_in_top3 = any(c['err'] <= 0.5 for c in candidates[:3])
            gt_in_top5 = any(c['err'] <= 0.5 for c in candidates[:5])
            
            gt_rank = None
            for c in candidates:
                if c['err'] <= 0.5:
                    gt_rank = c['rank']
                    break
                    
            dx = top1['x'] - gt_x
            dy = top1['y'] - gt_y
            nearest_shift_x = float(round(dx / 3.0) * 3.0)
            periodic_residual_x = float(abs(dx - nearest_shift_x))
            nearest_shift_y = float(round(dy / 3.0) * 3.0)
            periodic_residual_y = float(abs(dy - nearest_shift_y))
            is_periodic = bool((periodic_residual_x < 0.1) and (periodic_residual_y < 0.1) and (top1['err'] > 1.0))
            
            results.append({
                'case_id': case_id,
                'context_size': ctx,
                'D': D,
                'architecture': architecture,
                'difficulty': difficulty,
                'alpha': alpha,
                'top1_x': top1['x'],
                'top1_y': top1['y'],
                'top1_error': top1['err'],
                'top1_is_correct': gt_in_top1,
                'gt_rank': gt_rank,
                'gt_in_top3': gt_in_top3,
                'gt_in_top5': gt_in_top5,
                'top1_score': top1['score'],
                'gt_score': gt_score,
                'top1_minus_gt': top1['score'] - gt_score,
                'periodic_ambiguity_flag': is_periodic
            })
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "raw_only_vs_hybrid_results.csv"), index=False)
    
    # Calculate transitions
    cases_05 = df[df['alpha'] == 0.5].set_index('case_id')
    cases_10 = df[df['alpha'] == 1.0].set_index('case_id')
    
    total_cases = len(cases_05)
    
    acc_05 = float(cases_05['top1_is_correct'].mean() * 100)
    acc_10 = float(cases_10['top1_is_correct'].mean() * 100)
    
    fail_to_succ = sum((~cases_05['top1_is_correct']) & (cases_10['top1_is_correct']))
    succ_to_fail = sum((cases_05['top1_is_correct']) & (~cases_10['top1_is_correct']))
    net_improvement = fail_to_succ - succ_to_fail
    
    stats_05 = {
        'accuracy': acc_05,
        'mean_error': float(cases_05['top1_error'].mean()),
        'rmse': float(np.sqrt((cases_05['top1_error']**2).mean())),
        'gt_in_top5': float(cases_05['gt_in_top5'].mean() * 100),
        'periodic_failure_rate': float(cases_05['periodic_ambiguity_flag'].mean() * 100)
    }
    
    stats_10 = {
        'accuracy': acc_10,
        'mean_error': float(cases_10['top1_error'].mean()),
        'rmse': float(np.sqrt((cases_10['top1_error']**2).mean())),
        'gt_in_top5': float(cases_10['gt_in_top5'].mean() * 100),
        'periodic_failure_rate': float(cases_10['periodic_ambiguity_flag'].mean() * 100)
    }
    
    summary = {
        'total_cases': total_cases,
        'alpha_05_stats': stats_05,
        'alpha_10_stats': stats_10,
        'transitions': {
            'failure_to_success': int(fail_to_succ),
            'success_to_failure': int(succ_to_fail),
            'net_improvement': int(net_improvement)
        }
    }
    
    with open(os.path.join(out_dir, "raw_only_vs_hybrid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary), f, indent=4)
        
    # Plot comparisons
    plot_bar_comparison([stats_05['accuracy']], [stats_10['accuracy']], ['Accuracy'], 'Top-1 Accuracy (%)', 'Accuracy (%)', 'accuracy_comparison.png', out_dir)
    plot_bar_comparison([stats_05['rmse']], [stats_10['rmse']], ['RMSE'], 'Root Mean Square Error (px)', 'Error (px)', 'rmse_comparison.png', out_dir)
    plot_bar_comparison([stats_05['mean_error']], [stats_10['mean_error']], ['Mean Error'], 'Mean Error (px)', 'Error (px)', 'mean_error_comparison.png', out_dir)
    plot_bar_comparison([stats_05['gt_in_top5']], [stats_10['gt_in_top5']], ['GT in Top-5'], 'GT in Top-5 (%)', 'Rate (%)', 'gt_top5_comparison.png', out_dir)
    plot_bar_comparison([stats_05['periodic_failure_rate']], [stats_10['periodic_failure_rate']], ['Periodic Failure'], 'Periodic Failure Rate (%)', 'Rate (%)', 'periodic_failure_comparison.png', out_dir)
    
    # Generate tables
    def generate_md_table(group_col):
        res = "### By " + group_col.capitalize() + "\n"
        res += "| " + group_col.capitalize() + " | Alpha 0.5 Acc | Alpha 1.0 Acc | Delta |\n"
        res += "|---|---|---|---|\n"
        for val in sorted(df[group_col].unique()):
            sub_05 = cases_05[cases_05[group_col] == val]
            sub_10 = cases_10[cases_10[group_col] == val]
            a05 = sub_05['top1_is_correct'].mean() * 100
            a10 = sub_10['top1_is_correct'].mean() * 100
            res += f"| {val} | {a05:.1f}% | {a10:.1f}% | {a10-a05:+.1f}% |\n"
        return res + "\n"

    report = f"""# Raw-Only vs Hybrid Validation Report

## 1. Experimental Objective
Compare the standard Frozen Champion (Alpha=0.5 Hybrid) against a strict Raw-only (Alpha=1.0) configuration on the full 96-case Deep Interior Disambiguation dataset, to establish whether dropping the LowFreq fusion is a net positive.

## 2. Global Results
- **Cases Evaluated**: {total_cases}
- **Failure → Success**: {fail_to_succ} cases
- **Success → Failure**: {succ_to_fail} cases
- **Net Improvement**: {net_improvement:+} cases

## 3. Breakdown Tables
{generate_md_table('context_size')}
{generate_md_table('architecture')}
{generate_md_table('difficulty')}

## 4. Analytical Findings

**Q1: Does alpha=1.0 outperform alpha=0.5 on the full 96-case set?**
{"Yes" if acc_10 > acc_05 else "No"}. Alpha=1.0 achieved {acc_10:.1f}%, compared to Alpha=0.5 at {acc_05:.1f}%.

**Q2: How many cases improve?**
{fail_to_succ} cases successfully flipped from failure to correct localization.

**Q3: How many cases regress?**
{succ_to_fail} cases regressed from correct to failure.

**Q4/Q5: Change Rates**
- Fails → Successes: {fail_to_succ}
- Successes → Fails: {succ_to_fail}

**Q6: Does Raw-only reduce periodic ambiguity?**
{"Yes" if stats_10['periodic_failure_rate'] < stats_05['periodic_failure_rate'] else "No"}. Periodic failure rate went from {stats_05['periodic_failure_rate']:.1f}% to {stats_10['periodic_failure_rate']:.1f}%.

**Q7: Does Raw-only improve deep-interior cases?**
{"Yes" if acc_10 > acc_05 else "No"}.

**Q8: Does LowFreq still provide any measurable benefit?**
{"Yes, there were regressions when it was removed." if succ_to_fail > 0 else "No, removing it caused zero regressions on this 96-case subset."}

**Q9: Is Raw-only sufficiently better to justify a production experiment?**
{"Yes. The net improvement is substantial enough to warrant a formal A/B test." if net_improvement > 0 else "No. Raw-only performs worse or yields negligible benefit."}
"""

    with open(os.path.join(out_dir, "RAW_ONLY_VALIDATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
        
    print("========================================")
    print("RAW-ONLY VS HYBRID VALIDATION COMPLETE")
    print("========================================")
    print(f"Cases: {total_cases}\n")
    print("ALPHA 0.5:")
    print(f"Accuracy: {stats_05['accuracy']:.2f}%")
    print(f"Mean Error: {stats_05['mean_error']:.2f} px")
    print(f"RMSE: {stats_05['rmse']:.2f} px")
    print(f"GT Top-5: {stats_05['gt_in_top5']:.2f}%\n")
    print("ALPHA 1.0:")
    print(f"Accuracy: {stats_10['accuracy']:.2f}%")
    print(f"Mean Error: {stats_10['mean_error']:.2f} px")
    print(f"RMSE: {stats_10['rmse']:.2f} px")
    print(f"GT Top-5: {stats_10['gt_in_top5']:.2f}%\n")
    print(f"Failure -> Success: {fail_to_succ}")
    print(f"Success -> Failure: {succ_to_fail}")
    print(f"Net improvement: {net_improvement}")
    print("========================================")

if __name__ == '__main__':
    main()
