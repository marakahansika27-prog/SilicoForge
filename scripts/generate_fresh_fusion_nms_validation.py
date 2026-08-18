import os
import sys
import json
import time
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

def generate_deep_interior_case(seed, architecture, difficulty, offset_x, offset_y, context_margin):
    rs = np.random.RandomState(seed)
    base_size = 10240
    base_img = np.zeros((base_size, base_size), dtype=np.float32)
    start_x, end_x, start_y, end_y = 2100, 8100, 2100, 8100
    
    if architecture == "DRAM":
        for i in range(150, base_size, 300):
            if not (start_x <= i < end_x): continue
            for j in range(150, base_size, 300):
                if not (start_y <= j < end_y): continue
                cv2.circle(base_img, (i, j), 60, 180, -1)
                cv2.rectangle(base_img, (i-45, j-45), (i+45, j+45), 100, 6)
    elif architecture == "FinFET":
        for i in range(150, base_size, 200):
            if not (start_x <= i < end_x): continue
            cv2.line(base_img, (i, start_y), (i, end_y), 180, 30)
        for j in range(150, base_size, 600):
            if not (start_y <= j < end_y): continue
            cv2.line(base_img, (start_x, j), (end_x, j), 120, 15)
            
    cv2.rectangle(base_img, (start_x-200, start_y-200), (end_x+200, end_y+200), 50, 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    gradient = cv2.morphologyEx(base_img, cv2.MORPH_GRADIENT, kernel)
    sem_img = np.clip(base_img + gradient * 1.5, 0, 255).astype(np.float32)
    
    noise_level = 50
    blur = 0
    if difficulty == "easy": noise_level = 20
    elif difficulty == "moderate": noise_level = 50; blur = 3
    elif difficulty == "hard": noise_level = 90; blur = 5
        
    exp_x1 = int(offset_x) - context_margin
    exp_y1 = int(offset_y) - context_margin
    exp_x2 = int(offset_x) + 900 + context_margin
    exp_y2 = int(offset_y) + 900 + context_margin
    
    pad_left = max(0, -exp_x1)
    pad_top = max(0, -exp_y1)
    pad_right = max(0, exp_x2 - base_size)
    pad_bottom = max(0, exp_y2 - base_size)
    
    valid_x1, valid_y1 = max(0, exp_x1), max(0, exp_y1)
    valid_x2, valid_y2 = min(base_size, exp_x2), min(base_size, exp_y2)
    
    ref_valid = sem_img[valid_y1:valid_y2, valid_x1:valid_x2].copy()
    
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        ref_float = np.pad(ref_valid, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
    else:
        ref_float = ref_valid
        
    gt_x = (offset_x + 450.0) / 10.0
    gt_y = (offset_y + 450.0) / 10.0
    
    search_float = cv2.resize(sem_img, (1024, 1024), interpolation=cv2.INTER_AREA)
    if blur > 0: search_float = cv2.GaussianBlur(search_float, (blur, blur), 0)
        
    ref_noise = rs.poisson(ref_float / 255.0 * 20) / 20 * 255
    ref_img = np.clip(ref_float + ref_noise - 128, 0, 255).astype(np.uint8)
    search_noise = rs.poisson(search_float / 255.0 * noise_level) / noise_level * 255
    search_img = np.clip(search_float + search_noise - 128, 0, 255).astype(np.uint8)
    
    return ref_img, search_img, gt_x, gt_y

def extract_top_k(res, top_k, nms_radius, new_w, new_h, gt_x, gt_y):
    res_nms = res.copy()
    candidates = []
    
    for i in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(res_nms)
        if max_val < -0.99:
            break
            
        x, y = max_loc
        delta_x, delta_y = 0.0, 0.0
        
        if 0 < x < res.shape[1] - 1 and 0 < y < res.shape[0] - 1:
            fx_m1, fx_0, fx_p1 = float(res[y, x-1]), float(res[y, x]), float(res[y, x+1])
            denom_x = fx_m1 - 2*fx_0 + fx_p1
            if abs(denom_x) > 1e-6:
                dx = 0.5 * (fx_m1 - fx_p1) / denom_x
                if abs(dx) <= 0.5: delta_x = dx
                    
            fy_m1, fy_0, fy_p1 = float(res[y-1, x]), float(res[y, x]), float(res[y+1, x])
            denom_y = fy_m1 - 2*fy_0 + fy_p1
            if abs(denom_y) > 1e-6:
                dy = 0.5 * (fy_m1 - fy_p1) / denom_y
                if abs(dy) <= 0.5: delta_y = dy
                    
        c_x = x + delta_x + new_w / 2.0
        c_y = y + delta_y + new_h / 2.0
        
        err = np.linalg.norm(np.array([c_x, c_y]) - np.array([gt_x, gt_y]))
        candidates.append({
            'rank': i + 1,
            'x': float(c_x),
            'y': float(c_y),
            'score': float(max_val),
            'err': float(err)
        })
        
        y1, y2 = max(0, int(y - nms_radius)), min(res_nms.shape[0], int(y + nms_radius))
        x1, x2 = max(0, int(x - nms_radius)), min(res_nms.shape[1], int(x + nms_radius))
        res_nms[y1:y2, x1:x2] = -1.0
        
    return candidates

def is_local_max(res, px, py):
    h, w = res.shape
    ix, iy = int(round(px)), int(round(py))
    ix = max(0, min(w - 1, ix))
    iy = max(0, min(h - 1, iy))
    val = res[iy, ix]
    y1, y2 = max(0, iy-1), min(h, iy+2)
    x1, x2 = max(0, ix-1), min(w, ix+2)
    neighborhood = res[y1:y2, x1:x2]
    return bool(val >= np.max(neighborhood))

def process_config(res_raw, res_lowfreq, alpha, nms_factor, base_radius, new_w, new_h, gt_x, gt_y):
    res_hybrid = alpha * res_raw + (1.0 - alpha) * res_lowfreq
    current_radius = max(1, int(base_radius * nms_factor))
    
    gt_map_x, gt_map_y = gt_x - new_w / 2.0, gt_y - new_h / 2.0
    gt_is_local_max = is_local_max(res_hybrid, gt_map_x, gt_map_y)
    
    candidates = extract_top_k(res_hybrid, 50, current_radius, new_w, new_h, gt_x, gt_y)
    if not candidates:
        return None
        
    top1_err = candidates[0]['err']
    
    def in_top_k(k):
        return any(c['err'] <= 0.5 for c in candidates[:k])
        
    gt_in_top1 = in_top_k(1)
    gt_in_top5 = in_top_k(5)
    gt_in_top50 = in_top_k(50)
    nearest_err = min([c['err'] for c in candidates])
    gt_survives = gt_in_top50
    
    top1 = candidates[0]
    dx, dy = top1['x'] - gt_x, top1['y'] - gt_y
    nearest_shift_x, nearest_shift_y = float(round(dx / 3.0) * 3.0), float(round(dy / 3.0) * 3.0)
    is_periodic = bool((abs(dx - nearest_shift_x) < 0.1) and (abs(dy - nearest_shift_y) < 0.1) and (top1_err > 1.0))
    
    return {
        'top1_accuracy': gt_in_top1,
        'top5_recall': gt_in_top5,
        'top50_recall': gt_in_top50,
        'mean_localization_error': top1_err,
        'gt_present_before_nms': gt_is_local_max,
        'gt_surviving_nms': gt_survives,
        'candidate_count': len(candidates),
        'nearest_candidate_distance': nearest_err,
        'periodic_ambiguity_rate': is_periodic
    }

def main():
    print("JSON serialization hardening: ENABLED")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = os.path.join(base_dir, 'benchmark', 'results', 'fresh_fusion_nms_validation')
    ensure_dir(out_dir)
    
    context_sizes = [900, 1350, 1750, 2000]
    D_positions = [1000, 1250, 1500, 2000]
    archs = ["DRAM", "FinFET"]
    diffs = ["easy", "moderate", "hard"]
    
    planned_cases = []
    case_idx = 1
    for ctx in context_sizes:
        for d in D_positions:
            for arch in archs:
                for diff in diffs:
                    planned_cases.append({
                        'case_id': f"fresh_val_{case_idx:04d}",
                        'context_size': ctx,
                        'D': d,
                        'architecture': arch,
                        'difficulty': diff,
                        'seed': 8000 + case_idx
                    })
                    case_idx += 1
                    
    results = []
    for spec in planned_cases:
        x_off = 2100 + spec['D']
        y_off = 2100 + spec['D']
        
        ref_img, search_img, gt_x, gt_y = generate_deep_interior_case(
            spec['seed'], spec['architecture'], spec['difficulty'], x_off, y_off, spec['context_size']
        )
        
        h_ref, w_ref = ref_img.shape
        new_h, new_w = h_ref // 10, w_ref // 10
        ref_scaled = cv2.resize(ref_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        res_raw = cv2.matchTemplate(search_img, ref_scaled, cv2.TM_CCOEFF_NORMED)
        search_blurred = cv2.GaussianBlur(search_img, (31, 31), 15)
        ref_blurred = cv2.GaussianBlur(ref_scaled, (31, 31), 15)
        res_lowfreq = cv2.matchTemplate(search_blurred, ref_blurred, cv2.TM_CCOEFF_NORMED)
        
        base_radius = min(new_w, new_h) // 2
        
        # A) FROZEN CHAMPION
        metrics_a = process_config(res_raw, res_lowfreq, 0.5, 1.00, base_radius, new_w, new_h, gt_x, gt_y)
        
        # B) PROPOSED CONFIGURATION
        metrics_b = process_config(res_raw, res_lowfreq, 1.0, 0.50, base_radius, new_w, new_h, gt_x, gt_y)
        
        if not metrics_a or not metrics_b:
            continue
            
        case_res = {
            'case_id': spec['case_id'],
            'context_size': spec['context_size'],
            'D': spec['D'],
            'architecture': spec['architecture'],
            'difficulty': spec['difficulty'],
            'seed': spec['seed']
        }
        for k, v in metrics_a.items():
            case_res[f'champ_{k}'] = v
        for k, v in metrics_b.items():
            case_res[f'prop_{k}'] = v
            
        results.append(case_res)
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "fresh_validation_results.csv"), index=False)
    
    total_cases = len(df)
    
    def agg_metrics(prefix):
        return {
            'accuracy': float(df[f'{prefix}_top1_accuracy'].mean() * 100),
            'top5_recall': float(df[f'{prefix}_top5_recall'].mean() * 100),
            'top50_recall': float(df[f'{prefix}_top50_recall'].mean() * 100),
            'mean_error': float(df[f'{prefix}_mean_localization_error'].mean()),
            'median_error': float(df[f'{prefix}_mean_localization_error'].median()),
            'rmse': float(np.sqrt((df[f'{prefix}_mean_localization_error']**2).mean())),
            'gt_present_before_nms': float(df[f'{prefix}_gt_present_before_nms'].mean() * 100),
            'gt_surviving_nms': float(df[f'{prefix}_gt_surviving_nms'].mean() * 100),
            'mean_candidate_count': float(df[f'{prefix}_candidate_count'].mean()),
            'mean_nearest_distance': float(df[f'{prefix}_nearest_candidate_distance'].mean()),
            'periodic_rate': float(df[f'{prefix}_periodic_ambiguity_rate'].mean() * 100)
        }
        
    champ = agg_metrics('champ')
    prop = agg_metrics('prop')
    
    # Paired comparisons
    df['diff_acc'] = df['prop_top1_accuracy'].astype(int) - df['champ_top1_accuracy'].astype(int)
    df['diff_rmse_raw'] = df['prop_mean_localization_error']**2 - df['champ_mean_localization_error']**2
    df['diff_mean_err'] = df['prop_mean_localization_error'] - df['champ_mean_localization_error']
    df['diff_top5'] = df['prop_top5_recall'].astype(int) - df['champ_top5_recall'].astype(int)
    df['diff_top50'] = df['prop_top50_recall'].astype(int) - df['champ_top50_recall'].astype(int)
    
    champ_fail_prop_succ = sum((~df['champ_top1_accuracy']) & (df['prop_top1_accuracy']))
    champ_succ_prop_fail = sum((df['champ_top1_accuracy']) & (~df['prop_top1_accuracy']))
    both_succ = sum((df['champ_top1_accuracy']) & (df['prop_top1_accuracy']))
    both_fail = sum((~df['champ_top1_accuracy']) & (~df['prop_top1_accuracy']))
    
    summary = {
        'total_cases': total_cases,
        'champion': champ,
        'proposed': prop,
        'paired_differences': {
            'accuracy_diff': float(df['diff_acc'].mean() * 100),
            'mean_error_diff': float(df['diff_mean_err'].mean()),
            'top5_diff': float(df['diff_top5'].mean() * 100),
            'top50_diff': float(df['diff_top50'].mean() * 100)
        },
        'transitions': {
            'champion_fails_proposed_succeeds': int(champ_fail_prop_succ),
            'champion_succeeds_proposed_fails': int(champ_succ_prop_fail),
            'both_succeed': int(both_succ),
            'both_fail': int(both_fail)
        }
    }
    
    with open(os.path.join(out_dir, "fresh_validation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary), f, indent=4)
        
    # Breakdowns
    def get_breakdown(col):
        res = f"### Breakdown by {col.capitalize()}\n"
        res += f"| {col.capitalize()} | Champ Acc | Prop Acc | Champ RMSE | Prop RMSE | Champ GT Surv | Prop GT Surv |\n"
        res += "|---|---|---|---|---|---|---|\n"
        for val in sorted(df[col].unique()):
            sub = df[df[col] == val]
            c_acc = sub['champ_top1_accuracy'].mean() * 100
            p_acc = sub['prop_top1_accuracy'].mean() * 100
            c_rmse = np.sqrt((sub['champ_mean_localization_error']**2).mean())
            p_rmse = np.sqrt((sub['prop_mean_localization_error']**2).mean())
            c_surv = sub['champ_gt_surviving_nms'].mean() * 100
            p_surv = sub['prop_gt_surviving_nms'].mean() * 100
            res += f"| {val} | {c_acc:.1f}% | {p_acc:.1f}% | {c_rmse:.1f} | {p_rmse:.1f} | {c_surv:.1f}% | {p_surv:.1f}% |\n"
        return res + "\n"

    # Plotting
    labels = ['Frozen Champion (0.5, 1.00R)', 'Proposed (1.0, 0.50R)']
    
    def plot_bar(vals, ylabel, title, filename):
        plt.figure(figsize=(8,6))
        plt.bar(labels, vals, color=['blue', 'green'], alpha=0.7)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()
        
    plot_bar([champ['accuracy'], prop['accuracy']], 'Accuracy (%)', 'Top-1 Accuracy Comparison', 'top1_accuracy_comparison.png')
    plot_bar([champ['rmse'], prop['rmse']], 'RMSE (px)', 'RMSE Comparison', 'rmse_comparison.png')
    plot_bar([champ['mean_error'], prop['mean_error']], 'Mean Error (px)', 'Mean Localization Error', 'mean_error_comparison.png')
    plot_bar([champ['gt_surviving_nms'], prop['gt_surviving_nms']], 'GT Survival (%)', 'GT Survival After NMS', 'gt_survival_comparison.png')
    
    plt.figure(figsize=(10,6))
    plt.hist(df['champ_mean_localization_error'], bins=30, alpha=0.5, label='Champion Error')
    plt.hist(df['prop_mean_localization_error'], bins=30, alpha=0.5, label='Proposed Error')
    plt.xlabel('Error (px)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Paired Error Distributions')
    plt.savefig(os.path.join(out_dir, 'paired_error_difference.png'), dpi=150)
    plt.close()
    
    acc_diff = prop['accuracy'] - champ['accuracy']
    rmse_diff = champ['rmse'] - prop['rmse']  # positive means improvement
    err_diff = champ['mean_error'] - prop['mean_error']
    surv_diff = prop['gt_surviving_nms'] - champ['gt_surviving_nms']
    
    def does_generalize(col):
        accs = df.groupby(col).apply(lambda x: (x['prop_top1_accuracy'].mean() - x['champ_top1_accuracy'].mean()) * 100)
        return "Yes" if all(accs > 0) else ("Yes, mostly" if sum(accs > 0) >= len(accs)/2 else "No")

    report = f"""# Fresh Fusion & NMS Validation Report

## 1. Experimental Objective
Validate whether the (Alpha=1.0, NMS=0.50R) improvement observed in earlier ablations generalizes strictly to a fresh, independent validation set of {total_cases} difficult cases.

## 2. Answers to Diagnostic Questions

**Q1. Does alpha=1.0 + NMS=0.50R improve Top-1 accuracy on fresh data?**
{"Yes" if acc_diff > 0 else "No"}. Accuracy shifted from {champ['accuracy']:.1f}% to {prop['accuracy']:.1f}%.

**Q2. Does it improve RMSE?**
{"Yes" if rmse_diff > 0 else "No"}. RMSE shifted from {champ['rmse']:.1f} to {prop['rmse']:.1f} px.

**Q3. Does it improve mean/median localization error?**
{"Yes" if err_diff > 0 else "No"}. Mean error shifted from {champ['mean_error']:.1f} to {prop['mean_error']:.1f} px. Median shifted from {champ['median_error']:.1f} to {prop['median_error']:.1f} px.

**Q4. Does it increase GT survival after NMS?**
{"Yes" if surv_diff > 0 else "No"}. Survival shifted from {champ['gt_surviving_nms']:.1f}% to {prop['gt_surviving_nms']:.1f}%.

**Q5. Does it improve Top-5 or Top-50 recall?**
{"Yes" if (prop['top5_recall'] > champ['top5_recall']) else "No"}. Top-5 shifted from {champ['top5_recall']:.1f}% to {prop['top5_recall']:.1f}%.

**Q6. Does the improvement generalize across DRAM and FinFET?**
{does_generalize('architecture')}.

**Q7. Does it generalize across easy/moderate/hard?**
{does_generalize('difficulty')}.

**Q8. Does it generalize across the difficult distance/context regimes?**
{does_generalize('context_size')}.

**Q9. How many cases switch from failure → success?**
{champ_fail_prop_succ} cases successfully transitioned to PASS.

**Q10. How many switch from success → failure?**
{champ_succ_prop_fail} cases regressed to FAIL.

**Q11. Is there enough evidence to justify testing the proposed configuration as a new Champion?**
{"Yes. The massive accuracy gains strictly generalize to a completely fresh independent seed set without significant regression." if (acc_diff > 5.0 and champ_succ_prop_fail == 0) else ("Yes, the net improvement is positive, but formal A/B testing is required." if acc_diff > 0 else "No, it failed to generalize positively.")}

## 3. Breakdowns
{get_breakdown('architecture')}
{get_breakdown('difficulty')}
{get_breakdown('context_size')}
"""
    with open(os.path.join(out_dir, "FRESH_FUSION_NMS_VALIDATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
        
    print("\n========================================")
    print("FRESH FUSION x NMS VALIDATION COMPLETE")
    print("========================================")
    print(f"Fresh cases: {total_cases}\n")
    print(f"{'Configuration':<30} {'Accuracy':<10} {'Top-5':<10} {'RMSE':<10}")
    print(f"{'Frozen Champion (0.5, 1.00R)':<30} {champ['accuracy']:<10.1f} {champ['top5_recall']:<10.1f} {champ['rmse']:<10.1f}")
    print(f"{'Proposed (1.0, 0.50R)':<30} {prop['accuracy']:<10.1f} {prop['top5_recall']:<10.1f} {prop['rmse']:<10.1f}\n")
    print(f"Failure -> Success: {champ_fail_prop_succ}")
    print(f"Success -> Failure: {champ_succ_prop_fail}\n")
    print(f"RMSE improvement: {rmse_diff:+.1f} px")
    print(f"Accuracy improvement: {acc_diff:+.1f}%")
    print(f"GT survival improvement: {surv_diff:+.1f}%")
    print("========================================")

if __name__ == '__main__':
    main()
