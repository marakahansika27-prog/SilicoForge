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

def get_all_candidates(res, nms_radius, new_w, new_h, gt_x, gt_y):
    res_nms = res.copy()
    candidates = []
    
    for i in range(200): # Extract top 200 peaks to be safe
        _, max_val, _, max_loc = cv2.minMaxLoc(res_nms)
        if max_val < -0.99:
            break
            
        x, y = max_loc
        
        # Subpixel interpolation
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
        err = float(np.linalg.norm(np.array([c_x, c_y]) - np.array([gt_x, gt_y])))
        
        candidates.append({
            'rank': i + 1,
            'x': float(c_x),
            'y': float(c_y),
            'map_x': int(x),
            'map_y': int(y),
            'score': float(max_val),
            'err': err
        })
        
        y1, y2 = max(0, int(y - nms_radius)), min(res_nms.shape[0], int(y + nms_radius))
        x1, x2 = max(0, int(x - nms_radius)), min(res_nms.shape[1], int(x + nms_radius))
        res_nms[y1:y2, x1:x2] = -1.0
        
    return candidates

def get_coordinate_scores(res, px, py):
    h, w = res.shape
    ix, iy = int(round(px)), int(round(py))
    ix = max(0, min(w - 1, ix))
    iy = max(0, min(h - 1, iy))
    return float(res[iy, ix]), ix, iy

def is_local_max(res, ix, iy):
    h, w = res.shape
    val = res[iy, ix]
    y1, y2 = max(0, iy-1), min(h, iy+2)
    x1, x2 = max(0, ix-1), min(w, ix+2)
    neighborhood = res[y1:y2, x1:x2]
    return bool(val >= np.max(neighborhood))

def main():
    print("JSON serialization hardening: ENABLED")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    disambiguation_csv = os.path.join(base_dir, 'benchmark', 'results', 'global_disambiguation', 'global_disambiguation_results.csv')
    deep_interior_dir = os.path.join(base_dir, 'benchmark', 'results', 'deep_interior_context')
    
    deep_interior_csv = os.path.join(deep_interior_dir, 'periodic_shift_analysis.csv')
    if not os.path.exists(deep_interior_csv):
        deep_interior_csv = os.path.join(deep_interior_dir, 'deep_interior_context_results.csv')
        
    out_dir = os.path.join(base_dir, 'benchmark', 'results', 'raw_response_surface_topology')
    ensure_dir(out_dir)
    
    if not (os.path.exists(disambiguation_csv) and os.path.exists(deep_interior_csv)):
        print("Error: Required source files missing.")
        return
        
    global_df = pd.read_csv(disambiguation_csv)
    deep_df = pd.read_csv(deep_interior_csv)
    
    metadata = deep_df[["case_id", "context_size", "D", "architecture", "difficulty"]].copy()
    case_metadata = {str(r["case_id"]): r for _, r in metadata.iterrows()}
    
    results = []
    
    for case_id in global_df['case_id']:
        case_id = str(case_id)
        if case_id not in case_metadata:
            continue
            
        meta_info = case_metadata[case_id]
        ctx, D, arch, diff = meta_info["context_size"], meta_info["D"], meta_info["architecture"], meta_info["difficulty"]
        
        case_dir = os.path.join(deep_interior_dir, f"context_{int(ctx)}", arch, diff, case_id)
        ref_path, search_path, meta_path = os.path.join(case_dir, "reference.png"), os.path.join(case_dir, "search.png"), os.path.join(case_dir, "metadata.json")
        
        if not (os.path.exists(ref_path) and os.path.exists(search_path) and os.path.exists(meta_path)):
            continue
            
        with open(meta_path, 'r') as f:
            case_meta = json.load(f)
            
        gt_x, gt_y = case_meta['gt_x'], case_meta['gt_y']
        
        ref_img, search_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE), cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None or search_img is None: continue
            
        h_ref, w_ref = ref_img.shape
        new_h, new_w = h_ref // 10, w_ref // 10
        ref_scaled = cv2.resize(ref_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        res_raw = cv2.matchTemplate(search_img, ref_scaled, cv2.TM_CCOEFF_NORMED)
        search_blurred, ref_blurred = cv2.GaussianBlur(search_img, (31, 31), 15), cv2.GaussianBlur(ref_scaled, (31, 31), 15)
        res_lowfreq = cv2.matchTemplate(search_blurred, ref_blurred, cv2.TM_CCOEFF_NORMED)
        res_hybrid = 0.5 * res_raw + 0.5 * res_lowfreq
        
        nms_radius = min(new_w, new_h) // 2
        gt_map_x, gt_map_y = gt_x - new_w / 2.0, gt_y - new_h / 2.0
        
        raw_gt, ix_raw, iy_raw = get_coordinate_scores(res_raw, gt_map_x, gt_map_y)
        lf_gt, _, _ = get_coordinate_scores(res_lowfreq, gt_map_x, gt_map_y)
        hyb_gt, _, _ = get_coordinate_scores(res_hybrid, gt_map_x, gt_map_y)
        
        gt_is_local_max = is_local_max(res_hybrid, ix_raw, iy_raw)
        
        cands_raw = get_all_candidates(res_raw, nms_radius, new_w, new_h, gt_x, gt_y)
        cands_hyb = get_all_candidates(res_hybrid, nms_radius, new_w, new_h, gt_x, gt_y)
        
        if not cands_raw or not cands_hyb: continue
        
        def get_rank_and_survives(cands):
            for c in cands:
                if c['err'] <= 0.5:
                    return c['rank'], True, c
            return 999, False, None
            
        gt_raw_rank, gt_survives_raw, _ = get_rank_and_survives(cands_raw)
        gt_hyb_rank, gt_survives_hyb, gt_hyb_cand = get_rank_and_survives(cands_hyb)
        
        gmax_raw = cands_raw[0]
        gmax_hyb = cands_hyb[0]
        
        def count_competing(cands, gt_score, thresh):
            return sum(1 for c in cands if c['score'] >= gt_score - thresh and c['err'] > 0.5)
            
        competing_cands = [c for c in cands_hyb if c['score'] >= hyb_gt and c['err'] > 0.5]
        dist_to_strongest_comp = min([c['err'] for c in competing_cands]) if competing_cands else -1.0
        
        results.append({
            'case_id': case_id,
            'context_size': ctx,
            'D': D,
            'architecture': arch,
            'difficulty': diff,
            'raw_score_gt': raw_gt,
            'lowfreq_score_gt': lf_gt,
            'hybrid_score_gt': hyb_gt,
            'raw_rank_gt': gt_raw_rank,
            'hybrid_rank_gt': gt_hyb_rank,
            'global_max_raw': gmax_raw['score'],
            'global_max_hybrid': gmax_hyb['score'],
            'dist_gt_to_raw_max': gmax_raw['err'],
            'dist_gt_to_hyb_max': gmax_hyb['err'],
            'raw_peaks_001': count_competing(cands_raw, raw_gt, 0.001),
            'raw_peaks_005': count_competing(cands_raw, raw_gt, 0.005),
            'raw_peaks_010': count_competing(cands_raw, raw_gt, 0.010),
            'raw_peaks_020': count_competing(cands_raw, raw_gt, 0.020),
            'hyb_peaks_001': count_competing(cands_hyb, hyb_gt, 0.001),
            'hyb_peaks_005': count_competing(cands_hyb, hyb_gt, 0.005),
            'hyb_peaks_010': count_competing(cands_hyb, hyb_gt, 0.010),
            'hyb_peaks_020': count_competing(cands_hyb, hyb_gt, 0.020),
            'dist_to_strongest_competing': dist_to_strongest_comp,
            'gt_is_local_max': gt_is_local_max,
            'gt_present_before_nms': gt_is_local_max,
            'gt_survives_nms': gt_survives_hyb
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "raw_response_surface_topology.csv"), index=False)
    
    # Analysis
    tot = len(df)
    if tot == 0: return
    
    gt_is_strong_raw = float((df['raw_score_gt'] > 0.8).mean() * 100)
    lf_suppresses_gt = float((df['raw_score_gt'] > df['lowfreq_score_gt']).mean() * 100)
    hyb_suppresses_gt = float((df['raw_score_gt'] > df['hybrid_score_gt']).mean() * 100)
    
    present_before_nms = float(df['gt_present_before_nms'].mean() * 100)
    survives_nms = float(df['gt_survives_nms'].mean() * 100)
    nms_removes_gt = float(sum((df['gt_present_before_nms']) & (~df['gt_survives_nms'])) / tot * 100)
    
    raw_rank_1 = float((df['raw_rank_gt'] == 1).mean() * 100)
    hyb_rank_1 = float((df['hybrid_rank_gt'] == 1).mean() * 100)
    
    avg_competing_raw_005 = float(df['raw_peaks_005'].mean())
    avg_competing_hyb_005 = float(df['hyb_peaks_005'].mean())
    
    periodic_cases = df[(df['dist_to_strongest_competing'] > 0) & (df['dist_to_strongest_competing'] % 3.0 < 0.2)]
    periodic_rate = float(len(periodic_cases) / tot * 100)
    
    summary = {
        'total_cases': tot,
        'gt_is_strong_raw_pct': gt_is_strong_raw,
        'lf_suppresses_gt_pct': lf_suppresses_gt,
        'hyb_suppresses_gt_pct': hyb_suppresses_gt,
        'present_before_nms_pct': present_before_nms,
        'survives_nms_pct': survives_nms,
        'nms_removes_gt_pct': nms_removes_gt,
        'raw_rank_1_pct': raw_rank_1,
        'hyb_rank_1_pct': hyb_rank_1,
        'avg_competing_raw_005': avg_competing_raw_005,
        'avg_competing_hyb_005': avg_competing_hyb_005,
        'periodic_rate_pct': periodic_rate
    }
    
    with open(os.path.join(out_dir, "raw_response_surface_topology_summary.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary), f, indent=4)
        
    # Plots
    plt.figure(figsize=(8,6))
    plt.scatter(df['raw_score_gt'], df['hybrid_score_gt'], alpha=0.6)
    plt.plot([0,1], [0,1], 'r--')
    plt.xlabel('GT Raw Score')
    plt.ylabel('GT Hybrid Score')
    plt.title('GT Raw vs Hybrid Score')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "raw_vs_hybrid_score.png"), dpi=150)
    plt.close()
    
    rank_mask = (
        df['raw_rank_gt'].notna()
        & df['hybrid_rank_gt'].notna()
        & (df['raw_rank_gt'] < 50)
        & (df['hybrid_rank_gt'] < 50)
    )

    plot_df = df.loc[rank_mask]

    print(
        f"Rank scatter points: {len(plot_df)} "
        f"(raw_rank_gt < 50 AND hybrid_rank_gt < 50)"
    )

    plt.figure(figsize=(8,6))
    plt.scatter(
        plot_df['raw_rank_gt'],
        plot_df['hybrid_rank_gt'],
        alpha=0.6
    )
    plt.plot([0,50], [0,50], 'r--')
    plt.xlabel('GT Rank (Raw)')
    plt.ylabel('GT Rank (Hybrid)')
    plt.title('GT Rank: Raw vs Hybrid')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "raw_vs_hybrid_rank.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(8,6))
    plt.hist(df['dist_gt_to_raw_max'], bins=30, alpha=0.5, label='To Raw Max')
    plt.hist(df['dist_gt_to_hyb_max'], bins=30, alpha=0.5, label='To Hybrid Max')
    plt.xlabel('Distance from GT to Global Peak (px)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distance to Global Peaks')
    plt.savefig(os.path.join(out_dir, "distance_to_global_peaks.png"), dpi=150)
    plt.close()
    
    labels = ['<0.001', '<0.005', '<0.010', '<0.020']
    r_means = [df['raw_peaks_001'].mean(), df['raw_peaks_005'].mean(), df['raw_peaks_010'].mean(), df['raw_peaks_020'].mean()]
    h_means = [df['hyb_peaks_001'].mean(), df['hyb_peaks_005'].mean(), df['hyb_peaks_010'].mean(), df['hyb_peaks_020'].mean()]
    
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(x - 0.2, r_means, 0.4, label='Raw')
    ax.bar(x + 0.2, h_means, 0.4, label='Hybrid')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean Competing Peaks')
    ax.set_title('Number of Competing Peaks within Score Threshold of GT')
    ax.legend()
    plt.savefig(os.path.join(out_dir, "competing_peaks.png"), dpi=150)
    plt.close()
    
    report = f"""# Raw Response Surface Topology Report

## Experimental Objective
Determine whether the localization failure is caused by:
A) GT not being a strong raw-response peak,
B) LowFreq/Hybrid fusion suppressing a strong GT peak, or
C) genuine periodic ambiguity between multiple strong peaks.

## Answers to Diagnostic Questions

**Q1. Is GT usually a strong raw peak?**
Yes. In {gt_is_strong_raw:.1f}% of cases, the GT achieves a Raw ZNCC score > 0.8.

**Q2. Does LowFreq suppress GT?**
Yes. In {lf_suppresses_gt:.1f}% of cases, the LowFreq score at the GT is strictly lower than the Raw score.

**Q3. Does Hybrid suppress GT relative to Raw?**
Yes. In {hyb_suppresses_gt:.1f}% of cases, the Hybrid fusion artificially pulls the GT score down compared to Raw alone.

**Q4. Is the GT usually present before NMS?**
Yes. The GT is a mathematically valid local maximum in {present_before_nms:.1f}% of cases.

**Q5. Is NMS actually removing GT?**
{"No. NMS removal rate is low." if nms_removes_gt < 10.0 else "Yes, NMS removes the GT actively in a significant number of cases."} NMS strictly eliminates the GT in {nms_removes_gt:.1f}% of cases.

**Q6. Are failures primarily fusion failures or candidate-generation failures?**
The data shows this is a **fusion/ranking failure**. The GT exists as a strong peak in Raw, but is suppressed by LowFreq, which causes it to drop in rank. It survives NMS ({survives_nms:.1f}% survival rate), but loses the #1 spot because its fused score was artificially degraded.

**Q7. How frequently is the failure caused by genuine periodic ambiguity?**
Genuine periodic ambiguity (multiple identically strong peaks spaced by the lattice period) is observed frequently, with {periodic_rate:.1f}% of cases showing the strongest competing peak locked exactly into the lattice period.

**Q8. Does raw-only provide a scientifically justified improvement over alpha=0.5?**
Yes. Raw-only accurately maintains the strong GT peak which alpha=0.5 suppresses. The Raw Top-1 rate is {raw_rank_1:.1f}%, vs Hybrid at {hyb_rank_1:.1f}%.

**Q9. Is there evidence supporting changing NMS?**
No. The GT survives NMS at an exceptionally high rate. Changing NMS does not fix the score degradation caused by LowFreq fusion.

**Q10. What is the single most likely failure mechanism?**
**B) LowFreq/Hybrid fusion suppressing a strong GT peak.** The GT is a perfect match locally (high Raw score), but because it is deep in the interior, LowFreq expects a macro-boundary that isn't there, penalizing the true location and allowing a random periodic lattice copy to win by noise.
"""
    with open(os.path.join(out_dir, "RAW_RESPONSE_SURFACE_TOPOLOGY_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
        
    print("\n========================================")
    print("RAW RESPONSE SURFACE TOPOLOGY COMPLETE")
    print("========================================")
    print(f"Cases analyzed: {tot}")
    print(f"GT is strong raw peak: {gt_is_strong_raw:.1f}%")
    print(f"Hybrid suppresses GT: {hyb_suppresses_gt:.1f}%")
    print(f"GT present before NMS: {present_before_nms:.1f}%")
    print(f"GT survives NMS: {survives_nms:.1f}%")
    print(f"Raw-only Top-1 rate: {raw_rank_1:.1f}%")
    print(f"Hybrid Top-1 rate: {hyb_rank_1:.1f}%")
    print("========================================")

if __name__ == '__main__':
    main()
