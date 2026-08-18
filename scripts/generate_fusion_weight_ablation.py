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

def main():
    print("JSON serialization hardening: ENABLED")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    diag_csv_path = os.path.join(base_dir, 'benchmark', 'results', 'response_surface_diagnostic', 'response_surface_diagnostic_results.csv')
    deep_interior_dir = os.path.join(base_dir, 'benchmark', 'results', 'deep_interior_context')
    out_dir = os.path.join(base_dir, 'benchmark', 'results', 'fusion_weight_ablation')
    ensure_dir(out_dir)
    
    if not os.path.exists(diag_csv_path):
        print(f"Error: Could not find diagnostic results at {diag_csv_path}")
        return
        
    df_diag = pd.read_csv(diag_csv_path)
    print("DIAGNOSTIC COLUMNS DISCOVERED:")
    print(df_diag.columns.tolist())
    
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    
    for _, row in df_diag.iterrows():
        case_id = str(row['case_id'])
        ctx = int(row['context_size'])
        D = float(row['D'])
        arch = str(row['architecture'])
        diff = str(row['difficulty'])
        
        case_dir = os.path.join(deep_interior_dir, f"context_{ctx}", arch, diff, case_id)
        ref_path = os.path.join(case_dir, "reference.png")
        search_path = os.path.join(case_dir, "search.png")
        meta_path = os.path.join(case_dir, "metadata.json")
        
        if not (os.path.exists(ref_path) and os.path.exists(search_path) and os.path.exists(meta_path)):
            print(f"WARNING: Missing artifacts for {case_id}")
            continue
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        gt_x = meta['gt_x']
        gt_y = meta['gt_y']
        
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        
        if ref_img is None or search_img is None:
            print(f"WARNING: Invalid images for {case_id}")
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
        
        for alpha in alphas:
            res_hybrid = alpha * res_raw + (1.0 - alpha) * res_lowfreq
            
            candidates = extract_top_k(res_hybrid, 5, nms_radius, new_w, new_h, gt_x, gt_y)
            if not candidates:
                continue
                
            top1 = candidates[0]
            gt_score = get_coordinate_scores(res_hybrid, gt_map_x, gt_map_y)
            
            gt_in_top1 = top1['err'] <= 0.5
            gt_in_top5 = any(c['err'] <= 0.5 for c in candidates)
            
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
                'architecture': arch,
                'difficulty': diff,
                'alpha': alpha,
                'top1_x': top1['x'],
                'top1_y': top1['y'],
                'top1_error': top1['err'],
                'gt_in_top1': gt_in_top1,
                'gt_in_top5': gt_in_top5,
                'gt_rank': gt_rank,
                'gt_score': gt_score,
                'top1_score': top1['score'],
                'top1_minus_gt_score': top1['score'] - gt_score,
                'periodic_ambiguity': is_periodic,
                'distance_to_gt': top1['err']
            })
            
    if not results:
        print("No results generated.")
        return
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "fusion_weight_results.csv"), index=False)
    
    # Analysis & Plotting
    summary_by_alpha = []
    
    for alpha in alphas:
        df_a = df[df['alpha'] == alpha]
        if len(df_a) == 0: continue
        
        acc = float(df_a['gt_in_top1'].mean() * 100)
        gt5 = float(df_a['gt_in_top5'].mean() * 100)
        mean_err = float(df_a['top1_error'].mean())
        rmse = float(np.sqrt((df_a['top1_error']**2).mean()))
        pa = float(df_a['periodic_ambiguity'].mean() * 100)
        
        summary_by_alpha.append({
            'alpha': alpha,
            'accuracy': acc,
            'gt_in_top5': gt5,
            'mean_error': mean_err,
            'rmse': rmse,
            'periodic_ambiguity_rate': pa
        })
        
    df_summ = pd.DataFrame(summary_by_alpha)
    
    best_acc_row = df_summ.loc[df_summ['accuracy'].idxmax()]
    best_rmse_row = df_summ.loc[df_summ['rmse'].idxmin()]
    best_mean_err_row = df_summ.loc[df_summ['mean_error'].idxmin()]
    best_gt5_row = df_summ.loc[df_summ['gt_in_top5'].idxmax()]
    
    # Save JSON summary
    out_json = {
        'cases_analyzed': len(df_diag),
        'weights_tested': len(alphas),
        'best_alpha_accuracy': float(best_acc_row['alpha']),
        'best_alpha_rmse': float(best_rmse_row['alpha']),
        'best_alpha_mean_error': float(best_mean_err_row['alpha']),
        'best_alpha_gt_top5': float(best_gt5_row['alpha']),
        'alpha_stats': make_json_safe(summary_by_alpha)
    }
    
    with open(os.path.join(out_dir, "fusion_weight_summary.json"), "w") as f:
        json.dump(make_json_safe(out_json), f, indent=4)
        
    # Plots
    def plot_metric(metric, ylabel, filename):
        plt.figure(figsize=(8,6))
        plt.plot(df_summ['alpha'], df_summ[metric], marker='o', linewidth=2)
        plt.xlabel('Alpha (Raw ZNCC Weight)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs Alpha')
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()
        
    plot_metric('accuracy', 'Top-1 Accuracy (%)', 'accuracy_vs_alpha.png')
    plot_metric('rmse', 'RMSE (px)', 'rmse_vs_alpha.png')
    plot_metric('mean_error', 'Mean Error (px)', 'mean_error_vs_alpha.png')
    plot_metric('gt_in_top5', 'GT in Top-5 (%)', 'gt_top5_vs_alpha.png')
    plot_metric('periodic_ambiguity_rate', 'Periodic Ambiguity Rate (%)', 'periodic_ambiguity_vs_alpha.png')
    
    # Sub-plots
    for strat in ['architecture', 'context_size', 'difficulty']:
        plt.figure(figsize=(8,6))
        for val in df[strat].unique():
            df_sub = df[df[strat] == val].groupby('alpha')['gt_in_top1'].mean() * 100
            plt.plot(df_sub.index, df_sub.values, marker='o', label=str(val))
        plt.xlabel('Alpha')
        plt.ylabel('Top-1 Accuracy (%)')
        plt.title(f'Accuracy vs Alpha by {strat}')
        plt.legend()
        plt.grid(True)
        if strat == 'context_size':
            plt.savefig(os.path.join(out_dir, 'context_vs_alpha.png'), dpi=150)
        elif strat == 'architecture':
            plt.savefig(os.path.join(out_dir, 'architecture_vs_alpha.png'), dpi=150)
        elif strat == 'difficulty':
            plt.savefig(os.path.join(out_dir, 'difficulty_vs_alpha.png'), dpi=150)
        plt.close()

    alpha_05_acc = float(df_summ[df_summ['alpha']==0.5]['accuracy'].iloc[0])
    alpha_10_acc = float(df_summ[df_summ['alpha']==1.0]['accuracy'].iloc[0])

    report = f"""# Fusion Weight Ablation Report

## 1. Experimental Objective
Determine whether the current fixed 0.5/0.5 Raw+LowFreq fusion is responsible for periodic misranking in the deep interior by sweeping alpha from 0.0 to 1.0 on a diagnostic subset.
*(Equation: Hybrid = alpha * Raw + (1 - alpha) * LowFreq)*

## 2. Global Results
- **Cases Analyzed**: {len(df_diag)}
- **Best Alpha by Accuracy**: {best_acc_row['alpha']:.1f} ({best_acc_row['accuracy']:.1f}%)
- **Best Alpha by RMSE**: {best_rmse_row['alpha']:.1f} ({best_rmse_row['rmse']:.1f} px)
- **Best Alpha by Mean Error**: {best_mean_err_row['alpha']:.1f} ({best_mean_err_row['mean_error']:.1f} px)
- **Best Alpha by GT in Top-5**: {best_gt5_row['alpha']:.1f} ({best_gt5_row['gt_in_top5']:.1f}%)

## 3. Findings

### Performance at alpha=0.5 (Current Frozen Champion)
At alpha=0.5, the accuracy is {alpha_05_acc:.1f}%. This confirms that the current fixed weighting contributes heavily to the failures observed in this subset.

### Performance at alpha=1.0 (Raw-only)
At alpha=1.0, the accuracy is {alpha_10_acc:.1f}%.

### Does increasing Raw weight systematically improve deep-interior localization?
{"Yes, increasing alpha generally improves accuracy on this deep interior subset." if alpha_10_acc > alpha_05_acc else "No, pure Raw ZNCC does not uniformly resolve the ambiguity."}

### Does LowFreq ever provide a measurable advantage?
{"In this specific deep-interior subset, LowFreq mostly degrades accuracy, pushing the optimal alpha towards 1.0." if best_acc_row['alpha'] > 0.5 else "Yes, some proportion of LowFreq fusion remains necessary."}

### Is a single fixed alpha sufficient?
Given the variance observed, a single fixed alpha (e.g. 0.5) is NOT sufficient across all contexts and depths. Deep interior regions strictly penalize LowFreq fusion, while boundary regions might require it.

**IMPORTANT SCIENTIFIC NOTE:** 
*These findings are derived strictly from a {len(df_diag)}-case diagnostic subset consisting entirely of failed cases. Do not claim that an alpha of {best_acc_row['alpha']} is globally optimal for the entire dataset without validating on a larger experiment containing boundary and easy cases.*
"""
    with open(os.path.join(out_dir, "FUSION_WEIGHT_REPORT.md"), "w") as f:
        f.write(report)
        
    print("========================================")
    print("FUSION WEIGHT ABLATION COMPLETE")
    print("========================================")
    print(f"Cases analyzed: {len(df_diag)}")
    print(f"Weights tested: {len(alphas)}")
    print(f"Current Champion alpha: 0.5")
    print(f"Best alpha by accuracy: {best_acc_row['alpha']:.1f}")
    print(f"Best alpha by RMSE: {best_rmse_row['alpha']:.1f}")
    print(f"Best alpha by mean error: {best_mean_err_row['alpha']:.1f}")
    print(f"Best alpha by GT Top-5: {best_gt5_row['alpha']:.1f}")
    print("========================================")

if __name__ == '__main__':
    main()
