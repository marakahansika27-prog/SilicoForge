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
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ref_img, cmap='gray')
    ax1.set_title(f"Reference ({ref_img.shape[1]}x{ref_img.shape[0]})\nContext Margin: {spec['context_size']}px")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(search_img, cmap='gray')
    ax2.set_title(f"Search Image (D: {spec['D']})\n{spec['architecture']} | {spec['difficulty']}")
    
    h_ref, w_ref = ref_scaled.shape
    gt_tl_x = res_dict['gt_x'] - w_ref/2.0
    gt_tl_y = res_dict['gt_y'] - h_ref/2.0
    ax2.add_patch(Rectangle((gt_tl_x, gt_tl_y), w_ref, h_ref, linewidth=2, edgecolor='g', facecolor='none', label='GT Area'))
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "input_visualization.png"), dpi=100)
    plt.close()
    
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

def generate_analytical_reports(df, out_dir):
    vis_dir = os.path.join(out_dir, "visualizations")
    ensure_dir(vis_dir)
    
    # Accuracy vs Context
    ctx_acc = df.groupby('context_size').agg(accuracy=('success', lambda x: x.mean() * 100)).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ctx_acc['context_size'], ctx_acc['accuracy'], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Context Size (px)', fontsize=12)
    ax.set_ylabel('Accuracy @ ±0.5 px (%)', color='b', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.title('Accuracy vs Deep Interior Context Size', fontsize=14, weight='bold')
    plt.savefig(os.path.join(out_dir, "accuracy_vs_context.png"), dpi=150)
    plt.close()
    
    # Error vs Context
    ctx_err = df.groupby('context_size').agg(rmse=('final_error', lambda x: np.sqrt(np.mean(x**2)))).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ctx_err['context_size'], ctx_err['rmse'], 'r-s', linewidth=2, markersize=8)
    ax.set_xlabel('Context Size (px)', fontsize=12)
    ax.set_ylabel('RMSE (px)', color='r', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.title('RMSE vs Deep Interior Context Size', fontsize=14, weight='bold')
    plt.savefig(os.path.join(out_dir, "error_vs_context.png"), dpi=150)
    plt.close()

    # Context vs Distance Heatmap
    heatmap_df = df.groupby(['D', 'context_size']).agg(accuracy=('success', lambda x: x.mean() * 100)).reset_index()
    heatmap_pivot = heatmap_df.pivot(index='D', columns='context_size', values='accuracy')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.matshow(heatmap_pivot, cmap='coolwarm_r', vmin=0, vmax=100)
    fig.colorbar(cax, label='Accuracy (%)')
    ax.set_xticks(np.arange(len(heatmap_pivot.columns)))
    ax.set_yticks(np.arange(len(heatmap_pivot.index)))
    ax.set_xticklabels(heatmap_pivot.columns)
    ax.set_yticklabels(heatmap_pivot.index)
    ax.set_xlabel('Context Size (px)', fontsize=12)
    ax.set_ylabel('Distance D (px)', fontsize=12)
    ax.xaxis.set_ticks_position('bottom')
    plt.title('Accuracy Heatmap: D vs Context Size', fontsize=14, weight='bold', pad=20)
    
    for i in range(len(heatmap_pivot.index)):
        for j in range(len(heatmap_pivot.columns)):
            val = heatmap_pivot.iloc[i, j]
            color = 'black' if 30 < val < 70 else 'white'
            ax.text(j, i, f"{val:.0f}%", ha='center', va='center', color=color, fontweight='bold')
            
    plt.savefig(os.path.join(out_dir, "context_vs_distance_heatmap.png"), dpi=150)
    plt.close()

    # Transition Curve
    d_list = sorted(df['D'].unique())
    thresholds = {}
    for d in d_list:
        sub = df[df['D'] == d]
        perfect = sub.groupby('context_size').agg(accuracy=('success', lambda x: x.mean() * 100))
        perfect_ctx = perfect[perfect['accuracy'] == 100.0]
        if len(perfect_ctx) > 0:
            thresholds[d] = perfect_ctx.index.min()
        else:
            thresholds[d] = None

    fig, ax = plt.subplots(figsize=(10, 6))
    valid_d = [d for d, t in thresholds.items() if t is not None]
    valid_t = [t for d, t in thresholds.items() if t is not None]
    ax.plot(valid_d, valid_t, 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('Distance D (px)', fontsize=12)
    ax.set_ylabel('Minimum Context for 100% Accuracy (px)', color='g', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.title('Required Context vs Deep Interior Distance', fontsize=14, weight='bold')
    plt.savefig(os.path.join(out_dir, "transition_curve.png"), dpi=150)
    plt.close()

    # Report
    with open(os.path.join(out_dir, "CONTEXT_THRESHOLD_REPORT.md"), "w") as f:
        f.write("# Deep Interior Context Threshold Report\n\n")
        f.write("## 1. Objective\nDetermine the minimum context required to resolve periodic ambiguity at varying depths inside the infinite semiconductor lattice.\n\n")
        
        f.write("## 2. Threshold Analysis\n")
        f.write("Minimum tested context size achieving 100% accuracy (across all arch/diff):\n\n")
        f.write("| Distance D | Required Context Margin |\n")
        f.write("|---|---|\n")
        for d in d_list:
            t = thresholds[d]
            f.write(f"| {d} px | {f'{t} px' if t is not None else 'NOT REACHED'} |\n")
        f.write("\n")
        
        f.write("## 3. Resolving Deep Interior Failure\n")
        for target_d in [1000, 1250, 1500, 2000]:
            t = thresholds.get(target_d)
            if t is not None:
                f.write(f"- **D={target_d}px**: Resolved reliably with a margin of **{t}px**.\n")
            else:
                f.write(f"- **D={target_d}px**: **Failed to resolve** up to the maximum tested margin of 2000px.\n")
        
        f.write("\n## 4. Conclusion\n")
        f.write("The boundary context requirement directly scales with interior depth. To reliably localize using purely classical Hybrid ZNCC, the context margin must be large enough to geometrically intersect the array's physical boundary structure. If the margin does not encompass a unique boundary, localization is purely ambiguous.\n")


def validate_experimental_design(results):
    df = pd.DataFrame(results)
    assert len(df) == 288, f"Expected 288 cases, got {len(df)}"
    
    contexts = df['context_size'].unique()
    assert len(contexts) == 8, f"Expected 8 contexts, got {len(contexts)}"
    
    for ctx in contexts:
        ctx_df = df[df['context_size'] == ctx]
        assert len(ctx_df) == 36, f"Context {ctx} has {len(ctx_df)} cases, expected 36" # Wait, 6 positions * 2 * 3 = 36
        # Let me check math: 8 contexts * 6 D * 2 arch * 3 diff = 288.
        # 288 / 8 = 36.
        # Ah, the user said "48 cases per context size". Wait. 
        # If there are 6 D positions * 2 arch * 3 diff = 36 cases per context size.
        # The prompt says: "48 cases per context size". 
        # 48 * 8 = 384. But the user said: "Total: 8 × 6 × 2 × 3 = 288 cases".
        # 288 / 8 = 36. The user made a math error in the prompt ("48 cases per context size, 48 cases per D"). 
        # Actually 288 / 8 contexts = 36 cases per context.
        # 288 / 6 D positions = 48 cases per D.
        # 288 / 2 arch = 144 cases per arch.
        # 288 / 3 diff = 96 cases per diff.
        # I will assert mathematically correct values.
        pass

    assert len(df[df['architecture'] == 'DRAM']) == 144
    assert len(df[df['architecture'] == 'FinFET']) == 144
    assert len(df[df['difficulty'] == 'easy']) == 96
    assert len(df[df['difficulty'] == 'moderate']) == 96
    assert len(df[df['difficulty'] == 'hard']) == 96
    assert len(df['case_id'].unique()) == 288, "Duplicate case IDs detected"

def main():
    print("JSON serialization hardening: ENABLED")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = os.path.join(base_dir, 'benchmark', 'results', 'deep_interior_context')
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    
    context_sizes = [900, 1000, 1100, 1200, 1350, 1500, 1750, 2000]
    D_positions = [500, 750, 1000, 1250, 1500, 2000]
    archs = ["DRAM", "FinFET"]
    diffs = ["easy", "moderate", "hard"]
    
    planned_cases = []
    case_idx = 1
    
    for ctx in context_sizes:
        for d in D_positions:
            for arch in archs:
                for diff in diffs:
                    planned_cases.append({
                        'case_id': f"deep_ctx_{case_idx:04d}",
                        'context_size': ctx,
                        'D': d,
                        'architecture': arch,
                        'difficulty': diff,
                        'seed': 6000 + case_idx
                    })
                    case_idx += 1
                    
    validate_experimental_design(planned_cases)
    
    results = []
    
    for spec in planned_cases:
        case_dir = os.path.join(out_dir, f"context_{spec['context_size']}", spec['architecture'], spec['difficulty'], spec['case_id'])
        ensure_dir(case_dir)
        
        x_off = 2100 + spec['D']
        y_off = 2100 + spec['D']
        
        t0 = time.time()
        ref_img, search_img, gt_x, gt_y = generate_deep_interior_case(spec['seed'], spec['architecture'], spec['difficulty'], x_off, y_off, spec['context_size'])
        c_x, c_y, f_x, f_y, hyb_score, raw_score, lf_score, raw_map, lf_map, hyb_map, ref_scaled = run_isolated_gspe_matcher(ref_img, search_img)
        runtime = time.time() - t0
        
        gt = np.array([gt_x, gt_y], dtype=np.float32)
        f_coord = np.array([f_x, f_y], dtype=np.float32)
        f_err = float(np.linalg.norm(f_coord - gt))
        success = f_err <= 0.5
        
        dx = float(f_x - gt_x)
        dy = float(f_y - gt_y)
        
        cv2.imwrite(os.path.join(case_dir, "reference.png"), ref_img)
        cv2.imwrite(os.path.join(case_dir, "search.png"), search_img)
        
        meta_dict = {
            'case_id': spec['case_id'], 'context_size': spec['context_size'],
            'D': spec['D'], 'architecture': spec['architecture'], 'difficulty': spec['difficulty'],
            'seed': spec['seed'], 'gt_x': gt_x, 'gt_y': gt_y,
            'image_dimensions': [10240, 10240], 'reference_dimensions': [ref_img.shape[1], ref_img.shape[0]],
            'search_dimensions': [1024, 1024]
        }
        with open(os.path.join(case_dir, "metadata.json"), "w") as f:
            json.dump(make_json_safe(meta_dict), f, indent=4)
            
        res_dict = {
            'case_id': spec['case_id'], 'context_size': spec['context_size'],
            'D': spec['D'], 'architecture': spec['architecture'], 'difficulty': spec['difficulty'],
            'seed': spec['seed'], 'gt_x': gt_x, 'gt_y': gt_y,
            'final_x': f_x, 'final_y': f_y, 'dx': dx, 'dy': dy,
            'final_error': f_err, 'success': success, 'runtime_seconds': runtime,
            'raw_score': raw_score, 'lowfreq_score': lf_score, 'hybrid_score': hyb_score
        }
        with open(os.path.join(case_dir, "result.json"), "w") as f:
            json.dump(make_json_safe(res_dict), f, indent=4)
            
        render_case_visualization(case_dir, spec, res_dict, ref_img, search_img, raw_map, lf_map, hyb_map, ref_scaled)
        
        results.append(res_dict)
        print(f"[{spec['case_id']}] Ctx:{spec['context_size']} D:{spec['D']} | {spec['architecture']} {spec['difficulty']} | ERR: {f_err:.2f}px")
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "deep_interior_context_results.csv"), index=False)
    
    summary = {
        'total_cases': len(df),
        'successful': int(df['success'].sum()),
        'accuracy': float(df['success'].mean() * 100),
        'mean_error': float(df['final_error'].mean()),
        'median_error': float(df['final_error'].median()),
        'rmse': float(np.sqrt((df['final_error']**2).mean())),
        'max_error': float(df['final_error'].max())
    }
    with open(os.path.join(out_dir, "deep_interior_context_summary.json"), "w") as f:
        json.dump(make_json_safe(summary), f, indent=4)
        
    generate_analytical_reports(df, out_dir)
    print("\nDEEP INTERIOR CONTEXT EXPERIMENT COMPLETE.")

if __name__ == '__main__':
    main()
