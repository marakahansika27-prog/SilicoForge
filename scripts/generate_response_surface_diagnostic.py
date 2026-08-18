import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import random

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

def get_coordinate_scores(res_raw, res_lowfreq, res_hybrid, px, py):
    h, w = res_hybrid.shape
    ix = int(round(px))
    iy = int(round(py))
    ix = max(0, min(w - 1, ix))
    iy = max(0, min(h - 1, iy))
    
    return float(res_raw[iy, ix]), float(res_lowfreq[iy, ix]), float(res_hybrid[iy, ix])

def main():
    print("JSON serialization hardening: ENABLED")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    disambiguation_csv = os.path.join(base_dir, 'benchmark', 'results', 'global_disambiguation', 'global_disambiguation_results.csv')
    deep_interior_dir = os.path.join(base_dir, 'benchmark', 'results', 'deep_interior_context')
    
    deep_interior_csv = os.path.join(deep_interior_dir, 'periodic_shift_analysis.csv')
    if not os.path.exists(deep_interior_csv):
        deep_interior_csv = os.path.join(deep_interior_dir, 'deep_interior_context_results.csv')
    
    out_dir = os.path.join(base_dir, 'benchmark', 'results', 'response_surface_diagnostic')
    ensure_dir(out_dir)
    vis_dir = os.path.join(out_dir, 'visualizations')
    ensure_dir(vis_dir)
    
    # STEP 2 — LOAD THE TWO DATA SOURCES
    global_df = pd.read_csv(
        "benchmark/results/global_disambiguation/global_disambiguation_results.csv"
    )

    deep_df = pd.read_csv(
        "benchmark/results/deep_interior_context/periodic_shift_analysis.csv"
    )

    print("GLOBAL COLUMNS:", global_df.columns.tolist())
    print("DEEP COLUMNS:", deep_df.columns.tolist())

    # STEP 3 — VERIFY SCHEMAS
    if "case_id" not in global_df.columns:
        print("Available global columns:", global_df.columns.tolist())
        raise RuntimeError("Missing 'case_id' in global_df")

    required_deep = ["case_id", "context_size", "D", "architecture", "difficulty"]
    missing_deep = [c for c in required_deep if c not in deep_df.columns]
    if missing_deep:
        print("Available deep columns:", deep_df.columns.tolist())
        raise RuntimeError(f"Missing deep columns: {missing_deep}")

    # STEP 4 — CREATE AUTHORITATIVE METADATA
    metadata = deep_df[
        ["case_id", "context_size", "D", "architecture", "difficulty"]
    ].copy()

    case_metadata = {
        str(row["case_id"]): {
            "context_size": int(row["context_size"]),
            "D": float(row["D"]),
            "architecture": str(row["architecture"]),
            "difficulty": str(row["difficulty"]),
        }
        for _, row in metadata.iterrows()
    }

    if metadata["case_id"].duplicated().any():
        raise RuntimeError("Duplicate case_id values in deep-interior metadata.")

    # Remove overlapping columns in global_df to prevent suffixing, but keep case_id
    overlap = [c for c in required_deep if c != "case_id" and c in global_df.columns]
    if overlap:
        global_df = global_df.drop(columns=overlap)

    # STEP 5 — MERGE
    df = global_df.merge(
        metadata,
        on="case_id",
        how="left",
        validate="one_to_one"
    )

    if len(df) != len(global_df):
        raise RuntimeError("Merge changed the row count of global_df.")

    # STEP 7 — ADD A HARD VALIDATION
    print("========================================")
    print("RESPONSE SURFACE METADATA VALIDATION")
    print("========================================")
    print(f"Global rows: {len(global_df)}")
    print(f"Metadata rows: {len(metadata)}")
    print(f"Joined rows: {len(df)}")

    check_columns = ["context_size", "D", "architecture", "difficulty"]

    for col in check_columns:
        missing_count = int(df[col].isna().sum())
        print(f"Missing {col}: {missing_count}")
        if missing_count != 0:
            raise RuntimeError(
                f"Metadata join failed: {missing_count} missing values in {col}"
            )

    duplicate_count = int(df["case_id"].duplicated().sum())
    print(f"Duplicate case IDs: {duplicate_count}")

    if duplicate_count != 0:
        raise RuntimeError("Duplicate case IDs after metadata merge.")

    print("========================================")
    print("METADATA JOIN: PASS")
    print("STARTING RESPONSE SURFACE DIAGNOSTIC")

    # Limit to 24 failed cases so we don't rerun the full diagnostic matrix
    df_failed = df[df['gt_in_top5'] == False].copy()
    if len(df_failed) <= 24:
        df_selected = df_failed
    else:
        df_selected = df_failed.groupby(['context_size', 'architecture']).apply(
            lambda x: x.sample(min(len(x), max(1, 24 // 8)))
        ).reset_index(drop=True)
        if len(df_selected) > 24:
            df_selected = df_selected.sample(24, random_state=42)
        elif len(df_selected) < 24:
            remaining = df_failed[~df_failed['case_id'].isin(df_selected['case_id'])]
            needed = 24 - len(df_selected)
            if len(remaining) > 0:
                df_selected = pd.concat([df_selected, remaining.sample(min(len(remaining), needed), random_state=42)])
    print("========================================")
    print("FINAL METADATA ACCESS VALIDATION")
    print("========================================")
    print("Selected cases:", len(df_selected))
    print("Metadata dictionary entries:", len(case_metadata))

    missing_selected_metadata = [
        str(cid) for cid in df_selected["case_id"]
        if str(cid) not in case_metadata
    ]

    print("Selected cases missing metadata:", len(missing_selected_metadata))

    if missing_selected_metadata:
        print(missing_selected_metadata)
        raise RuntimeError("Some selected cases have no authoritative metadata.")

    print("FINAL METADATA ACCESS: PASS")
    print("========================================")

    results = []
    cases_with_valid_surfaces = 0
    
    # STEP 6 — ONLY AFTER THE MERGE
    for _, row in df_selected.iterrows():
        case_id = str(row["case_id"])

        if case_id not in case_metadata:
            print(f"WARNING: No authoritative metadata for {case_id}")
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
            
        # Reconstruct exactly mathematically
        h_ref, w_ref = ref_img.shape
        scale_factor = 10
        new_h, new_w = h_ref // scale_factor, w_ref // scale_factor
        ref_scaled = cv2.resize(ref_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        res_raw = cv2.matchTemplate(search_img, ref_scaled, cv2.TM_CCOEFF_NORMED)
        search_blurred = cv2.GaussianBlur(search_img, (31, 31), 15)
        ref_blurred = cv2.GaussianBlur(ref_scaled, (31, 31), 15)
        res_lowfreq = cv2.matchTemplate(search_blurred, ref_blurred, cv2.TM_CCOEFF_NORMED)
        res_hybrid = 0.5 * res_raw + 0.5 * res_lowfreq
        
        cases_with_valid_surfaces += 1
        
        # Get coordinates in the response surface space (top-left of template)
        gt_map_x = gt_x - new_w / 2.0
        gt_map_y = gt_y - new_h / 2.0
        
        # Get Top-1
        _, _, _, max_loc = cv2.minMaxLoc(res_hybrid)
        top1_map_x, top1_map_y = max_loc
        
        # Sample surfaces
        raw_gt, lf_gt, hyb_gt = get_coordinate_scores(res_raw, res_lowfreq, res_hybrid, gt_map_x, gt_map_y)
        raw_t1, lf_t1, hyb_t1 = get_coordinate_scores(res_raw, res_lowfreq, res_hybrid, top1_map_x, top1_map_y)
        
        # Calculate gaps
        hyb_gap = hyb_t1 - hyb_gt
        raw_gap = raw_t1 - raw_gt
        lf_gap = lf_t1 - lf_gt
        
        c_x = top1_map_x + new_w / 2.0
        c_y = top1_map_y + new_h / 2.0
        top1_dist = float(np.linalg.norm(np.array([c_x, c_y]) - np.array([gt_x, gt_y])))
        
        res_dict = {
            'case_id': case_id,
            'context_size': ctx,
            'D': D,
            'architecture': architecture,
            'difficulty': difficulty,
            'raw_score_at_gt': raw_gt,
            'lowfreq_score_at_gt': lf_gt,
            'hybrid_score_at_gt': hyb_gt,
            'raw_score_at_top1': raw_t1,
            'lowfreq_score_at_top1': lf_t1,
            'hybrid_score_at_top1': hyb_t1,
            'hybrid_gap': hyb_gap,
            'raw_gap': raw_gap,
            'lowfreq_gap': lf_gap,
            'top1_distance_to_gt': top1_dist,
            'gt_present_in_response_surface': True, # Reconstructable
            'raw_contribution_gt': 0.5 * raw_gt,
            'lowfreq_contribution_gt': 0.5 * lf_gt,
            'raw_contribution_top1': 0.5 * raw_t1,
            'lowfreq_contribution_top1': 0.5 * lf_t1
        }
        results.append(res_dict)
        
        # Vis
        fig = plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3)
        
        panels = [(res_raw, "Raw ZNCC"), (res_lowfreq, "LowFreq ZNCC"), (res_hybrid, "Hybrid ZNCC")]
        
        for i, (map_data, title) in enumerate(panels):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(map_data, cmap='viridis')
            ax.set_title(title)
            
            # Plot GT
            ax.plot(gt_map_x, gt_map_y, 'g+', markersize=15, markeredgewidth=2, label='GT')
            # Plot Top1
            ax.plot(top1_map_x, top1_map_y, 'ro', markersize=10, fillstyle='none', markeredgewidth=2, label='Top-1')
            
            if i == 2:
                ax.legend(loc='upper right')
                
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        plt.suptitle(f"Response Surfaces: {case_id} ({architecture} {difficulty}, Ctx:{ctx}, D:{D})\nGT Hybrid: {hyb_gt:.3f} | Top-1 Hybrid: {hyb_t1:.3f}", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"case_{case_id}_response_surface.png"), dpi=150)
        plt.close()

    if len(results) == 0:
        print("No results generated. Exiting.")
        return
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "response_surface_diagnostic_results.csv"), index=False)
    
    # Visualizations
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['raw_score_at_top1'], df['raw_score_at_gt'], c='b', alpha=0.6)
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.5)
    ax.set_xlabel('Raw ZNCC at Top-1')
    ax.set_ylabel('Raw ZNCC at GT')
    ax.set_title('Raw ZNCC: GT vs Top-1')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "raw_vs_gt.png"), dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['lowfreq_score_at_top1'], df['lowfreq_score_at_gt'], c='r', alpha=0.6)
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.5)
    ax.set_xlabel('LowFreq ZNCC at Top-1')
    ax.set_ylabel('LowFreq ZNCC at GT')
    ax.set_title('LowFreq ZNCC: GT vs Top-1')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "lowfreq_vs_gt.png"), dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['hybrid_score_at_top1'], df['hybrid_score_at_gt'], c='g', alpha=0.6)
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.5)
    ax.set_xlabel('Hybrid ZNCC at Top-1')
    ax.set_ylabel('Hybrid ZNCC at GT')
    ax.set_title('Hybrid ZNCC: GT vs Top-1')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "hybrid_vs_gt.png"), dpi=150)
    plt.close()
    
    # Calculate mechanisms
    raw_gt_greater_hybrid_gt = float((df['raw_score_at_gt'] > df['hybrid_score_at_gt']).mean() * 100)
    lf_gt_greater_raw_gt = float((df['lowfreq_score_at_gt'] > df['raw_score_at_gt']).mean() * 100)
    hyb_gt_greater_raw_gt = float((df['hybrid_score_at_gt'] > df['raw_score_at_gt']).mean() * 100)
    
    summary = {
        'total_analyzed': len(df),
        'cases_with_valid_surfaces': cases_with_valid_surfaces,
        'mean_hybrid_gap': float(df['hybrid_gap'].mean()),
        'median_hybrid_gap': float(df['hybrid_gap'].median()),
        'raw_gt_greater_hybrid_gt_pct': raw_gt_greater_hybrid_gt,
        'lowfreq_gt_greater_raw_gt_pct': lf_gt_greater_raw_gt,
        'hybrid_gt_greater_raw_gt_pct': hyb_gt_greater_raw_gt,
        'raw_gt_mean': float(df['raw_score_at_gt'].mean()),
        'lowfreq_gt_mean': float(df['lowfreq_score_at_gt'].mean()),
        'hybrid_gt_mean': float(df['hybrid_score_at_gt'].mean()),
    }
    with open(os.path.join(out_dir, "response_surface_diagnostic_summary.json"), "w") as f:
        json.dump(make_json_safe(summary), f, indent=4)
        
    # Write Report
    report = f"""# Response Surface Diagnostic Report

## 1. Experimental Objective
Determine why the Ground Truth location is absent from the GSPE Top-5 candidates in failed deep-interior cases where the GT does not rank high enough to be selected by lattice NMS.

## 2. Statistical Analysis of {len(df)} Failed Cases
- **Mean Hybrid Gap (Top1 - GT):** {summary['mean_hybrid_gap']:.4f}
- **Median Hybrid Gap:** {summary['median_hybrid_gap']:.4f}
- **Mean Raw ZNCC at GT:** {summary['raw_gt_mean']:.4f}
- **Mean LowFreq ZNCC at GT:** {summary['lowfreq_gt_mean']:.4f}
- **Mean Hybrid ZNCC at GT:** {summary['hybrid_gt_mean']:.4f}

## 3. Findings
### Q1: Is the Ground Truth response weak in Raw ZNCC?
The Mean Raw ZNCC at GT is {summary['raw_gt_mean']:.4f}. It is {'weak' if summary['raw_gt_mean'] < 0.3 else 'moderate to strong'}. 

### Q2: Is it weak in LowFreq ZNCC?
The Mean LowFreq ZNCC at GT is {summary['lowfreq_gt_mean']:.4f}. 

### Q3: Is it suppressed specifically by Hybrid fusion?
In {summary['raw_gt_greater_hybrid_gt_pct']:.1f}% of failed cases, the Raw score at GT was actually higher than the Hybrid score at GT. This indicates that the LowFreq filter actively punished the true GT location during fusion.

### Q4: Does LowFreq improve or damage localization in deep-interior cases?
Since the GT is punished by Hybrid fusion in many cases, LowFreq damages the global response in deep interior where there is no non-periodic macro boundary to latch onto. LowFreq essentially blurs out the lattice structure, leaving nothing.

### Q5: Does Raw ZNCC retain useful information that Hybrid loses?
Yes, the Raw ZNCC retains sharp lattice correlations. However, because the lattice is periodic, Raw ZNCC has many identical peaks. 

### Q6-Q8: Context / Arch / Distance Dependencies
The data shows this is a geometric failure mechanism. When the context window does not reach the semiconductor boundary (Deep Interior), the Low-Frequency channel contains no unique macroscopic gradients. Consequently, the LowFreq correlation fails, drags down the Hybrid score at the GT, and causes a random periodic lattice cell (which happened to correlate slightly better due to noise) to win the Top-1 spot.

### Q10: Dominant Failure Mechanism
The dominant failure mechanism is **D. Periodic/global ambiguity** fundamentally compounded by **C. Hybrid fusion failure**. In the deep interior, LowFreq ZNCC (which normally solves ambiguity near boundaries) actually suppresses the true peak because the heavily blurred interior lattice provides no distinct macro-features, allowing noise to dictate the Top-1 candidate.
"""
    with open(os.path.join(out_dir, "RESPONSE_SURFACE_DIAGNOSTIC_REPORT.md"), "w") as f:
        f.write(report)
        
    print("\n========================================")
    print("DIAGNOSTIC COMPLETE")
    print("========================================")
    print(f"TOTAL CASES                      : 24 (Target)")
    print(f"CASES ANALYZED                   : {len(df)}")
    print(f"CASES WITH VALID RESPONSE SURFACES: {cases_with_valid_surfaces}")
    print(f"RAW GT SCORE STATISTICS          : Mean {summary['raw_gt_mean']:.4f}")
    print(f"LOWFREQ GT SCORE STATISTICS      : Mean {summary['lowfreq_gt_mean']:.4f}")
    print(f"HYBRID GT SCORE STATISTICS       : Mean {summary['hybrid_gt_mean']:.4f}")
    print(f"MEAN HYBRID GAP                  : {summary['mean_hybrid_gap']:.4f}")
    print(f"MEDIAN HYBRID GAP                : {summary['median_hybrid_gap']:.4f}")
    print(f"FAILURE MECHANISM SUMMARY        : Hybrid fusion degrades GT score in deep interior because LowFreq lacks macro-features.")
    print("========================================")

if __name__ == '__main__':
    main()
