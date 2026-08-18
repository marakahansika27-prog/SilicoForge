import os
import sys
import json
import csv
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cv2

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

def generate_context_ablation_case(seed, architecture, difficulty, offset_x, offset_y, context_margin):
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

def run_isolated_gspe_matcher(ref_img, search_img):
    h_ref, w_ref = ref_img.shape
    new_h, new_w = h_ref // 10, w_ref // 10
    
    ref_scaled = cv2.resize(ref_img.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    search_f32 = search_img.astype(np.float32)
    
    res_raw = cv2.matchTemplate(search_f32, ref_scaled, cv2.TM_CCOEFF_NORMED)
    search_blurred = cv2.GaussianBlur(search_f32, (31, 31), 15)
    ref_blurred = cv2.GaussianBlur(ref_scaled, (31, 31), 15)
    res_lowfreq = cv2.matchTemplate(search_blurred, ref_blurred, cv2.TM_CCOEFF_NORMED)
    
    res_hybrid = 0.5 * res_raw + 0.5 * res_lowfreq
    
    _, max_val, _, max_loc = cv2.minMaxLoc(res_hybrid)
    ix, iy = max_loc
    
    center_x = float(ix) + float(new_w) / 2.0
    center_y = float(iy) + float(new_h) / 2.0
    
    z_size = 1
    y1, y2 = max(0, iy - z_size), min(res_hybrid.shape[0], iy + z_size + 1)
    x1, x2 = max(0, ix - z_size), min(res_hybrid.shape[1], ix + z_size + 1)
    
    dx_sub, dy_sub = 0.0, 0.0
    if (y2 - y1) == 3 and (x2 - x1) == 3:
        patch = res_hybrid[y1:y2, x1:x2]
        dx_sub = (patch[1,0] - patch[1,2]) / (2 * (patch[1,0] + patch[1,2] - 2*patch[1,1]) + 1e-6)
        dy_sub = (patch[0,1] - patch[2,1]) / (2 * (patch[0,1] + patch[2,1] - 2*patch[1,1]) + 1e-6)
        dx_sub = np.clip(dx_sub, -1.0, 1.0)
        dy_sub = np.clip(dy_sub, -1.0, 1.0)
        
    final_x = center_x + dx_sub
    final_y = center_y + dy_sub
    
    raw_score = float(res_raw[iy, ix])
    lf_score = float(res_lowfreq[iy, ix])
    
    return center_x, center_y, final_x, final_y, float(max_val), raw_score, lf_score, res_raw, res_lowfreq, res_hybrid, ref_scaled

def render_case_visualization(case_dir, spec, res_dict, ref_img, search_img, raw_map, lf_map, hyb_map, ref_scaled):
    # input_visualization.png
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ref_img, cmap='gray')
    ax1.set_title(f"Reference ({ref_img.shape[1]}x{ref_img.shape[0]})\nContext Margin: {spec['context_size']}px")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(search_img, cmap='gray')
    ax2.set_title(f"Search Image (D: {spec['D']})\n{spec['architecture']} | {spec['difficulty']}")
    
    # GT rectangle
    h_ref, w_ref = ref_scaled.shape
    gt_tl_x = res_dict['gt_x'] - w_ref/2.0
    gt_tl_y = res_dict['gt_y'] - h_ref/2.0
    ax2.add_patch(Rectangle((gt_tl_x, gt_tl_y), w_ref, h_ref, linewidth=2, edgecolor='g', facecolor='none', label='GT Area'))
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "input_visualization.png"), dpi=100)
    plt.close()
    
    # result_visualization.png
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3)
    
    ax_search = fig.add_subplot(gs[0, :2])
    ax_search.imshow(search_img, cmap='gray')
    ax_search.add_patch(Rectangle((gt_tl_x, gt_tl_y), w_ref, h_ref, linewidth=2, edgecolor='g', facecolor='none'))
    
    pred_tl_x = res_dict['final_x'] - w_ref/2.0
    pred_tl_y = res_dict['final_y'] - h_ref/2.0
    ax_search.add_patch(Rectangle((pred_tl_x, pred_tl_y), w_ref, h_ref, linewidth=2, edgecolor='r', linestyle='--', facecolor='none'))
    
    ax_search.set_title(f"Result Overlay - Error: {res_dict['final_error']:.2f}px ({'PASS' if res_dict['success'] else 'FAIL'})")
    
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis('off')
    info_text = (
        f"Case: {spec['case_id']}\n"
        f"Context: {spec['context_size']}px\n"
        f"D: {spec['D']}px\n"
        f"Arch: {spec['architecture']}\n"
        f"Diff: {spec['difficulty']}\n\n"
        f"Raw NCC: {res_dict['raw_score']:.4f}\n"
        f"LF NCC: {res_dict['lowfreq_score']:.4f}\n"
        f"Hybrid: {res_dict['hybrid_score']:.4f}\n\n"
        f"dx: {res_dict['dx']:.2f}\n"
        f"dy: {res_dict['dy']:.2f}\n"
        f"Periodic Ambiguity: {res_dict['periodic_ambiguity_flag']}"
    )
    ax_info.text(0.1, 0.5, info_text, fontsize=12, family='monospace', va='center')
    
    ax_raw = fig.add_subplot(gs[1, 0])
    ax_raw.imshow(raw_map, cmap='viridis')
    ax_raw.set_title("Raw NCC")
    
    ax_lf = fig.add_subplot(gs[1, 1])
    ax_lf.imshow(lf_map, cmap='viridis')
    ax_lf.set_title("LowFreq NCC")
    
    ax_hyb = fig.add_subplot(gs[1, 2])
    ax_hyb.imshow(hyb_map, cmap='viridis')
    ax_hyb.set_title("Hybrid NCC")
    
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "result_visualization.png"), dpi=100)
    plt.close()

def validate_experimental_design(results):
    df = pd.DataFrame(results)
    
    # 5. Total planned cases = 792.
    assert len(df) == 792, f"Expected 792 cases, got {len(df)}"
    
    contexts = df['context_size'].unique()
    assert len(contexts) == 11, f"Expected 11 contexts, got {len(contexts)}"
    
    for ctx in contexts:
        ctx_df = df[df['context_size'] == ctx]
        # 1. Every context has exactly 72 cases.
        assert len(ctx_df) == 72, f"Context {ctx} has {len(ctx_df)} cases, expected 72"
        # 2. Every context has 36 DRAM and 36 FinFET cases.
        assert len(ctx_df[ctx_df['architecture'] == 'DRAM']) == 36
        assert len(ctx_df[ctx_df['architecture'] == 'FinFET']) == 36
        # 3. Every context has 24 easy, 24 moderate, and 24 hard cases.
        assert len(ctx_df[ctx_df['difficulty'] == 'easy']) == 24
        assert len(ctx_df[ctx_df['difficulty'] == 'moderate']) == 24
        assert len(ctx_df[ctx_df['difficulty'] == 'hard']) == 24
        
        # 4. Every context × architecture × difficulty combination has exactly 12 cases.
        for arch in ['DRAM', 'FinFET']:
            for diff in ['easy', 'moderate', 'hard']:
                combo = ctx_df[(ctx_df['architecture'] == arch) & (ctx_df['difficulty'] == diff)]
                assert len(combo) == 12, f"Context {ctx}, {arch}, {diff} has {len(combo)} cases, expected 12"
                
                # 6. All 12 spatial positions are represented
                assert len(combo['D'].unique()) == 12, f"Not all 12 positions represented in {ctx} {arch} {diff}"
                
    # 7. No duplicate case IDs.
    assert len(df['case_id'].unique()) == 792, "Duplicate case IDs detected"
    
    print("========================================")
    print("CONTEXT SIZE ABLATION")
    print("========================================")
    print(f"Context sizes   : {len(contexts)}")
    print(f"Spatial positions: 12")
    print(f"Architectures   : 2")
    print(f"Difficulties    : 3")
    print(f"Expected cases  : 792")
    print(f"Generated cases : {len(df)}")
    print("========================================")
    
def write_output_manifest(out_dir):
    content = """# Context Size Ablation Output Manifest

## 1. Experimental Design
- **Total Cases:** 792
- **Context Sizes:** 0, 25, 50, 75, 100, 150, 200, 300, 450, 600, 900 (11 total)
- **Spatial Positions (D):** -450, -300, -200, -150, -100, -50, 0, 100, 300, 500, 1000, 2000 (12 total)
- **Architectures:** DRAM, FinFET (2 total)
- **Difficulties:** easy, moderate, hard (3 total)
- **Factorial Grid:** 11 × 12 × 2 × 3 = 792 cases.

## 2. Expected Folder Structure
For every case, a dedicated output directory is created:
`benchmark/results/context_ablation/context_{size}/{architecture}/{difficulty}/{case_id}/`

## 3. Expected Artifacts Per Case
Each case folder contains exactly 6 artifacts:
1. `reference.png` - The exact context-expanded reference image provided to the matcher.
2. `search.png` - The exact search image.
3. `metadata.json` - Generation seeds, sizes, coordinates, and deterministic inputs.
4. `result.json` - Raw pipeline outputs, errors, scores, and periodicity metrics.
5. `input_visualization.png` - Overlays of GT and contextual boundaries.
6. `result_visualization.png` - Reprojection of heatmaps, error vectors, and pass/fail states.

## 4. Reproducibility
The `reference.png` and `search.png` files are strictly the immutable images tested by the algorithm. They are never regenerated. The `metadata.json` stores the exact RandomState seed used for physical noise rendering.
"""
    with open(os.path.join(out_dir, "CONTEXT_ABLATION_OUTPUT_MANIFEST.md"), "w") as f:
        f.write(content)

def generate_analytical_reports(df, out_dir):
    # Curves
    ctx_groups = df.groupby('context_size').agg(accuracy=('success', lambda x: x.mean() * 100), mean_err=('final_error', 'mean'), rmse=('final_error', lambda x: np.sqrt(np.mean(x**2)))).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ctx_groups['context_size'], ctx_groups['accuracy'], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Contextual Margin (px)', fontsize=12)
    ax.set_ylabel('Accuracy @ ±0.5 px (%)', color='b', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.title('Localization Accuracy vs Global Context Size', fontsize=14, weight='bold')
    plt.savefig(os.path.join(out_dir, "visualizations", "context_accuracy_curve.png"), dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ctx_groups['context_size'], ctx_groups['rmse'], 'r-s', linewidth=2, markersize=8)
    ax.set_xlabel('Contextual Margin (px)', fontsize=12)
    ax.set_ylabel('RMSE (px)', color='r', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.title('RMSE vs Global Context Size', fontsize=14, weight='bold')
    plt.savefig(os.path.join(out_dir, "visualizations", "context_rmse_curve.png"), dpi=150)
    plt.close()

def main():
    print("JSON serialization hardening: ENABLED")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = os.path.join(base_dir, 'benchmark', 'results', 'context_ablation')
    
    # Clean directory safely
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    
    vis_dir = os.path.join(out_dir, 'visualizations')
    ensure_dir(vis_dir)
    
    context_sizes = [0, 25, 50, 75, 100, 150, 200, 300, 450, 600, 900]
    D_positions = [-450, -300, -200, -150, -100, -50, 0, 100, 300, 500, 1000, 2000]
    archs = ["DRAM", "FinFET"]
    diffs = ["easy", "moderate", "hard"]
    
    planned_cases = []
    case_idx = 1
    
    # FULL FACTORIAL GENERATION
    for ctx in context_sizes:
        for d in D_positions:
            for arch in archs:
                for diff in diffs:
                    planned_cases.append({
                        'case_id': f"ctx_case_{case_idx:04d}",
                        'context_size': ctx,
                        'D': d,
                        'architecture': arch,
                        'difficulty': diff,
                        'seed': 5000 + case_idx
                    })
                    case_idx += 1
                    
    # STRUCTURAL VALIDATION (MUST PASS BEFORE EXECUTION)
    validate_experimental_design(planned_cases)
    write_output_manifest(out_dir)
    
    # 792-CASE EXECUTION
    results = []
    
    for spec in planned_cases:
        case_dir = os.path.join(out_dir, f"context_{spec['context_size']}", spec['architecture'], spec['difficulty'], spec['case_id'])
        ensure_dir(case_dir)
        
        x_off = 2100 + spec['D']
        y_off = 2100 + spec['D']
        
        t0 = time.time()
        ref_img, search_img, gt_x, gt_y = generate_context_ablation_case(spec['seed'], spec['architecture'], spec['difficulty'], x_off, y_off, spec['context_size'])
        c_x, c_y, f_x, f_y, hyb_score, raw_score, lf_score, raw_map, lf_map, hyb_map, ref_scaled = run_isolated_gspe_matcher(ref_img, search_img)
        runtime = time.time() - t0
        
        gt = np.array([gt_x, gt_y], dtype=np.float32)
        f_coord = np.array([f_x, f_y], dtype=np.float32)
        f_err = float(np.linalg.norm(f_coord - gt))
        success = f_err <= 0.5
        
        dx = float(f_x - gt_x)
        dy = float(f_y - gt_y)
        
        nearest_shift_x = float(round(dx / 3.0) * 3.0)
        periodic_residual_x = float(abs(dx - nearest_shift_x))
        nearest_shift_y = float(round(dy / 3.0) * 3.0)
        periodic_residual_y = float(abs(dy - nearest_shift_y))
        
        is_periodic = bool((periodic_residual_x < 0.1) and (periodic_residual_y < 0.1) and (f_err > 1.0))
        
        # Image artifacts
        cv2.imwrite(os.path.join(case_dir, "reference.png"), ref_img)
        cv2.imwrite(os.path.join(case_dir, "search.png"), search_img)
        
        # Metadata
        meta_dict = {
            'case_id': spec['case_id'], 'context_size': spec['context_size'],
            'spatial_position': 'symmetric', 'D': spec['D'],
            'pre_array_context': max(0, -spec['D']),
            'architecture': spec['architecture'], 'difficulty': spec['difficulty'],
            'seed': spec['seed'], 'gt_x': gt_x, 'gt_y': gt_y,
            'image_dimensions': [10240, 10240], 'reference_dimensions': [ref_img.shape[1], ref_img.shape[0]],
            'search_dimensions': [1024, 1024]
        }
        with open(os.path.join(case_dir, "metadata.json"), "w") as f:
            json.dump(make_json_safe(meta_dict), f, indent=4)
            
        # Result dict
        res_dict = {
            'case_id': spec['case_id'], 'context_size': spec['context_size'],
            'D': spec['D'], 'pre_array_context': max(0, -spec['D']),
            'architecture': spec['architecture'], 'difficulty': spec['difficulty'], 'seed': spec['seed'],
            'gt_x': gt_x, 'gt_y': gt_y, 'final_x': f_x, 'final_y': f_y,
            'dx': dx, 'dy': dy, 'final_error': f_err, 'success': success, 'runtime_seconds': runtime,
            'raw_score': raw_score, 'lowfreq_score': lf_score, 'hybrid_score': hyb_score,
            'nearest_shift_x': nearest_shift_x, 'periodic_residual_x': periodic_residual_x,
            'nearest_shift_y': nearest_shift_y, 'periodic_residual_y': periodic_residual_y,
            'periodic_ambiguity_flag': is_periodic
        }
        with open(os.path.join(case_dir, "result.json"), "w") as f:
            json.dump(make_json_safe(res_dict), f, indent=4)
            
        # Vis
        render_case_visualization(case_dir, spec, res_dict, ref_img, search_img, raw_map, lf_map, hyb_map, ref_scaled)
        
        results.append(res_dict)
        print(f"[{spec['case_id']}] Ctx:{spec['context_size']} D:{spec['D']} | {spec['architecture']} {spec['difficulty']} | ERR: {f_err:.2f}px")
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "context_ablation_results.csv"), index=False)
    
    summary = {
        'total_cases': len(df),
        'successful': int(df['success'].sum()),
        'accuracy': float(df['success'].mean() * 100),
        'mean_error': float(df['final_error'].mean()),
        'median_error': float(df['final_error'].median()),
        'rmse': float(np.sqrt((df['final_error']**2).mean())),
        'max_error': float(df['final_error'].max())
    }
    with open(os.path.join(out_dir, "context_ablation_summary.json"), "w") as f:
        json.dump(make_json_safe(summary), f, indent=4)
        
    generate_analytical_reports(df, out_dir)
    
    print("\n========================================")
    print("COMPLETION REPORT")
    print("========================================")
    print(f"Total cases executed: {len(df)}")
    print(f"Unique cases: {len(df['case_id'].unique())}")
    print(f"Cases per context: {len(df) // 11}")
    print(f"Cases per architecture: {len(df) // 2}")
    print(f"Cases per difficulty: {len(df) // 3}")
    print(f"Missing combinations: 0")
    print(f"Duplicate case IDs: 0")
    print(f"Final CSV path: {os.path.join(out_dir, 'context_ablation_results.csv')}")
    print(f"Final summary JSON path: {os.path.join(out_dir, 'context_ablation_summary.json')}")
    print(f"Visualization path: {os.path.join(out_dir, 'visualizations')}")
    print("========================================\n")

if __name__ == '__main__':
    main()
