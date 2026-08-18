import os
import sys
import json
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.integration.pipeline import HybridNavigationPipeline

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_spatial_benchmark_case(seed, architecture, difficulty, offset_x, offset_y):
    """Deterministic spatial extraction of the benchmark case without modifying core physics."""
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
        
    ref_float = sem_img[int(offset_y):int(offset_y)+900, int(offset_x):int(offset_x)+900].copy()
    gt_x = (offset_x + 450.0) / 10.0
    gt_y = (offset_y + 450.0) / 10.0
    
    search_float = cv2.resize(sem_img, (1024, 1024), interpolation=cv2.INTER_AREA)
    if blur > 0: search_float = cv2.GaussianBlur(search_float, (blur, blur), 0)
        
    ref_noise = rs.poisson(ref_float / 255.0 * 20) / 20 * 255
    ref_img = np.clip(ref_float + ref_noise - 128, 0, 255).astype(np.uint8)
    search_noise = rs.poisson(search_float / 255.0 * noise_level) / noise_level * 255
    search_img = np.clip(search_float + search_noise - 128, 0, 255).astype(np.uint8)
    
    return ref_img, search_img, gt_x, gt_y

def get_zone_labels(r, c):
    y_zones = ["TOP_EDGE", "UPPER_INTERIOR", "CENTER", "LOWER_INTERIOR", "BOTTOM_EDGE"]
    x_zones = ["LEFT_EDGE", "LEFT_INTERIOR", "LEFT_INTERIOR", "CENTER_COLUMN", "CENTER_COLUMN", "RIGHT_INTERIOR", "RIGHT_INTERIOR", "RIGHT_EDGE"]
    y_label = y_zones[r]
    x_label = x_zones[c]
    
    if "EDGE" in y_label and "EDGE" in x_label: global_zone = "CORNER"
    elif "EDGE" in y_label or "EDGE" in x_label: global_zone = "EDGE"
    elif "INTERIOR" in y_label and "INTERIOR" in x_label: global_zone = "NEAR_EDGE"
    elif "CENTER" in y_label and "CENTER" in x_label: global_zone = "CENTER"
    else: global_zone = "INTERIOR"
    
    return y_label, x_label, global_zone

def reconstruct_heatmaps(ref_img, search_img):
    h_ref, w_ref = ref_img.shape
    new_h, new_w = h_ref // 10, w_ref // 10
    ref_scaled = cv2.resize(ref_img.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    search_f32 = search_img.astype(np.float32)
    res_raw = cv2.matchTemplate(search_f32, ref_scaled, cv2.TM_CCOEFF_NORMED)
    search_blurred = cv2.GaussianBlur(search_f32, (31, 31), 15)
    ref_blurred = cv2.GaussianBlur(ref_scaled, (31, 31), 15)
    res_lowfreq = cv2.matchTemplate(search_blurred, ref_blurred, cv2.TM_CCOEFF_NORMED)
    res_hybrid = 0.5 * res_raw + 0.5 * res_lowfreq
    return res_raw, res_lowfreq, res_hybrid, ref_scaled

def render_case_visualization(row, ref_img, search_img, out_path):
    raw_map, lf_map, hyb_map, ref_scaled = reconstruct_heatmaps(ref_img, search_img)
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1.5, 1])
    
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_ref.imshow(ref_img, cmap='gray')
    ax_ref.set_title(f"Reference Template\nZone: {row['global_zone']}")
    ax_ref.axis('off')
    
    ax_search = fig.add_subplot(gs[0, 1:3])
    ax_search.imshow(search_img, cmap='gray')
    ax_search.set_title(f"Search Image ({row['architecture']} | {row['difficulty']})")
    
    half_w = 45.0
    ax_search.add_patch(Rectangle((row['gt_x'] - half_w, row['gt_y'] - half_w), 90, 90, linewidth=2, edgecolor='g', facecolor='none', label='Ground Truth'))
    ax_search.add_patch(Rectangle((row['final_x'] - half_w, row['final_y'] - half_w), 90, 90, linewidth=2, edgecolor='r', facecolor='none', linestyle='--', label='Prediction'))
    ax_search.legend(loc='upper right')
    
    ax_info = fig.add_subplot(gs[0, 3])
    ax_info.axis('off')
    status_color = 'green' if row['success'] else 'red'
    info_text = (
        f"--- SPATIAL LOCALIZATION ---\n"
        f"Case: {row['case_id']}\n"
        f"Ground Truth: ({row['gt_x']:.1f}, {row['gt_y']:.1f})\n"
        f"Prediction: ({row['final_x']:.4f}, {row['final_y']:.4f})\n"
        f"Error: {row['final_error']:.4f} px\n"
        f"Status: {'PASS' if row['success'] else 'FAIL'}\n\n"
        f"--- PIPELINE INFO ---\n"
        f"GSPE: Hybrid ZNCC\n"
        f"Subpixel: 3x3 Parabolic\n"
        f"SRAE: BYPASSED\n"
        f"SNRN: BYPASSED\n"
    )
    ax_info.text(0.1, 0.5, info_text, fontsize=12, family='monospace', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor=status_color, lw=3))
    
    ax_raw = fig.add_subplot(gs[1, 0])
    im1 = ax_raw.imshow(raw_map, cmap='viridis')
    ax_raw.set_title(f"Raw NCC Response")
    plt.colorbar(im1, ax=ax_raw, fraction=0.046, pad=0.04)
    
    ax_lf = fig.add_subplot(gs[1, 1])
    im2 = ax_lf.imshow(lf_map, cmap='viridis')
    ax_lf.set_title(f"Low-Freq NCC Response")
    plt.colorbar(im2, ax=ax_lf, fraction=0.046, pad=0.04)
    
    ax_hyb = fig.add_subplot(gs[1, 2])
    im3 = ax_hyb.imshow(hyb_map, cmap='viridis')
    ax_hyb.set_title(f"Hybrid NCC Response")
    plt.colorbar(im3, ax=ax_hyb, fraction=0.046, pad=0.04)
    
    ax_zoom = fig.add_subplot(gs[1, 3])
    _, _, _, max_loc = cv2.minMaxLoc(hyb_map)
    ix, iy = max_loc
    z_size = 5
    y1, y2 = max(0, iy - z_size), min(hyb_map.shape[0], iy + z_size + 1)
    x1, x2 = max(0, ix - z_size), min(hyb_map.shape[1], ix + z_size + 1)
    ax_zoom.imshow(hyb_map[y1:y2, x1:x2], cmap='viridis', extent=[x1-0.5, x2-0.5, y2-0.5, y1-0.5])
    
    gt_map_x, gt_map_y = row['gt_x'] - 45.0, row['gt_y'] - 45.0
    pred_map_x, pred_map_y = row['final_x'] - 45.0, row['final_y'] - 45.0
    
    ax_zoom.plot(ix, iy, 'wo', markersize=10, label='Integer')
    ax_zoom.plot(pred_map_x, pred_map_y, 'r*', markersize=12, label='Subpx')
    ax_zoom.plot(gt_map_x, gt_map_y, 'gX', markersize=10, label='GT')
    ax_zoom.set_title(f"Subpixel Offset: ({pred_map_x-ix:+.4f}, {pred_map_y-iy:+.4f})")
    ax_zoom.legend(loc='lower right', fontsize=8)
    
    plt.suptitle(f"Drift-Sense V2 Spatial Benchmark | {row['case_id']} | Zone: {row['global_zone']}", fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def generate_spatial_maps(df, out_dir):
    # Map 1: Coverage Map
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1024)
    ax.set_ylim(1024, 0)
    
    colors = {'CORNER': 'red', 'EDGE': 'orange', 'NEAR_EDGE': 'yellow', 'INTERIOR': 'green', 'CENTER': 'blue'}
    
    for _, row in df.iterrows():
        ax.scatter(row['gt_x'], row['gt_y'], c=colors[row['global_zone']], s=100, edgecolors='black')
        ax.text(row['gt_x']+10, row['gt_y']-10, row['case_id'].replace('spatial_case_',''), fontsize=8)
        
    # Draw physical valid boundary (210 to 810 search pixels)
    ax.add_patch(Rectangle((210, 210), 600, 600, fill=False, edgecolor='black', linestyle='--', linewidth=2, label='Valid Substrate Region'))
    
    ax.set_title("Full-Wafer Spatial Coverage Map", fontsize=16)
    ax.legend()
    plt.savefig(os.path.join(out_dir, "spatial_coverage_map.png"), dpi=150)
    plt.close()
    
    # Map 2: Error Heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1024)
    ax.set_ylim(1024, 0)
    
    sc = ax.scatter(df['gt_x'], df['gt_y'], c=df['final_error'], cmap='plasma', s=200, edgecolors='black', vmin=0.0, vmax=0.2)
    plt.colorbar(sc, ax=ax, label="Euclidean Error (px)")
    
    ax.add_patch(Rectangle((210, 210), 600, 600, fill=False, edgecolor='black', linestyle='--', linewidth=2))
    ax.set_title("Spatial Error Heatmap (px)", fontsize=16)
    plt.savefig(os.path.join(out_dir, "spatial_error_heatmap.png"), dpi=150)
    plt.close()

def write_markdown_reports(df, summary, out_dir):
    # Analysis Report
    zone_stats = df.groupby('global_zone')['final_error'].agg(['count', 'mean', 'median', 'max', lambda x: np.sqrt(np.mean(x**2))])
    zone_stats.columns = ['Count', 'Mean', 'Median', 'Max', 'RMSE']
    zone_stats['Accuracy'] = df.groupby('global_zone')['success'].mean() * 100.0
    
    diff_stats = df.groupby('difficulty')['final_error'].agg(['count', 'mean', 'median', 'max', lambda x: np.sqrt(np.mean(x**2))])
    diff_stats.columns = ['Count', 'Mean', 'Median', 'Max', 'RMSE']
    diff_stats['Accuracy'] = df.groupby('difficulty')['success'].mean() * 100.0

    with open(os.path.join(out_dir, "full_wafer_spatial_analysis.md"), "w") as f:
        f.write("# Full-Wafer Spatial Analysis\n\n")
        f.write("## Performance by Spatial Zone\n\n")
        f.write(zone_stats.round(4).to_markdown())
        f.write("\n\n## Performance by Difficulty\n\n")
        f.write(diff_stats.round(4).to_markdown())
        
    # Main Benchmark Report
    with open(os.path.join(out_dir, "FULL_WAFER_BENCHMARK_REPORT.md"), "w") as f:
        f.write("# Full-Wafer Spatial Coverage Benchmark Report\n\n")
        f.write("## 1. Objective\nTo determine if the Drift-Sense V2 Frozen Champion localization remains uniformly robust across the entire valid substrate field, avoiding corner-case overfitting.\n\n")
        f.write("## 2. Sampling Strategy\nA deterministic 5x8 Cartesian grid across the valid structural boundary (2100-7200 px). Architectures and noise difficulties were uniformly mixed across all zones.\n\n")
        f.write("## 3. Overall Performance\n")
        f.write(f"- Total Cases: {summary['Total Cases']}\n")
        f.write(f"- Successful: {summary['Successful']}\n")
        f.write(f"- Accuracy @ ±0.5 px: {summary['Accuracy']}%\n")
        f.write(f"- Mean Error: {summary['Mean Error']:.4f} px\n")
        f.write(f"- Median Error: {summary['Median Error']:.4f} px\n")
        f.write(f"- RMSE: {summary['RMSE']:.4f} px\n")
        f.write(f"- Maximum Error: {summary['Max Error']:.4f} px\n\n")
        f.write("## 4. Observations\n")
        if summary['Successful'] == 40:
            f.write("The Frozen Champion completely succeeds across the entire wafer with 100% accuracy. There are no localized failure clusters, proving the GSPE Hybrid algorithm is globally invariant and robust to position.\n")
        else:
            f.write("Failures were detected. Refer to `spatial_error_heatmap.png` to identify the structural breakdown locations.\n")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = os.path.join(base_dir, 'benchmark', 'results', 'full_wafer')
    vis_dir = os.path.join(out_dir, 'visualizations')
    ensure_dir(vis_dir)
    
    pipeline = HybridNavigationPipeline()
    
    x_offsets = np.linspace(2100, 7200, 8)
    y_offsets = np.linspace(2100, 7200, 5)
    
    archs = ["DRAM", "FinFET"]
    diffs = ["easy", "moderate", "hard"]
    
    results = []
    case_idx = 1
    
    print("--- STARTING FULL-WAFER SPATIAL BENCHMARK ---")
    
    for r, y_off in enumerate(y_offsets):
        for c, x_off in enumerate(x_offsets):
            case_id = f"spatial_case_{case_idx:03d}"
            seed = 2000 + case_idx
            arch = archs[case_idx % len(archs)]
            diff = diffs[case_idx % len(diffs)]
            y_label, x_label, global_zone = get_zone_labels(r, c)
            
            print(f"[{case_id}] {global_zone} ({x_off:.0f}, {y_off:.0f}) | {arch} {diff}")
            
            ref_img, search_img, gt_x, gt_y = generate_spatial_benchmark_case(seed, arch, diff, x_off, y_off)
            gt = np.array([gt_x, gt_y], dtype=np.float32)
            
            t0 = time.time()
            state = pipeline.run(ref_img, search_img)
            runtime = time.time() - t0
            
            f_coord = state['final_coord']
            f_err = float(np.linalg.norm(f_coord - gt))
            success = f_err <= 0.5
            
            res_dict = {
                'case_id': case_id, 'seed': seed, 'architecture': arch, 'difficulty': diff,
                'x_zone': x_label, 'y_zone': y_label, 'global_zone': global_zone,
                'gt_x': float(gt_x), 'gt_y': float(gt_y),
                'final_x': float(f_coord[0]), 'final_y': float(f_coord[1]),
                'final_error': f_err, 'success': success, 'runtime_seconds': runtime
            }
            results.append(res_dict)
            
            # Generate Visualization
            render_case_visualization(res_dict, ref_img, search_img, os.path.join(vis_dir, f"{case_id}_result.png"))
            
            case_idx += 1
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "full_wafer_results.csv"), index=False)
    
    successful = df['success'].sum()
    summary = {
        'Total Cases': len(df),
        'Successful': int(successful),
        'Accuracy': float(successful / len(df) * 100),
        'Mean Error': float(df['final_error'].mean()),
        'Median Error': float(df['final_error'].median()),
        'RMSE': float(np.sqrt((df['final_error']**2).mean())),
        'Max Error': float(df['final_error'].max())
    }
    
    with open(os.path.join(out_dir, "full_wafer_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    generate_spatial_maps(df, out_dir)
    write_markdown_reports(df, summary, out_dir)
    
    print("\nBENCHMARK COMPLETE.")
    print(f"Accuracy: {summary['Accuracy']}% | RMSE: {summary['RMSE']:.4f} px")

if __name__ == '__main__':
    main()
