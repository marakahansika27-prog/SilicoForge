import os
import sys
import glob
import json
import csv
import math
import argparse
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.integration.pipeline import HybridNavigationPipeline

def validate_dataset_preflight(base_dir):
    print("Performing strict dataset pre-flight check...")
    
    if not os.path.exists(base_dir):
        print(f"FAIL: Dataset directory {base_dir} does not exist.")
        sys.exit(1)
        
    case_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'metadata.json' in files:
            case_dirs.append(root)
            
    if len(case_dirs) != 60:
        print(f"FAIL: Expected exactly 60 cases, found {len(case_dirs)}.")
        sys.exit(1)
        
    dram_count = 0
    finfet_count = 0
    case_ids = set()
    seeds = set()
    
    cases = []

    for cdir in case_dirs:
        case_id = os.path.basename(cdir)
        meta_path = os.path.join(cdir, 'metadata.json')
        ref_path = os.path.join(cdir, 'reference.png')
        search_path = os.path.join(cdir, 'search.png')
        
        if not os.path.exists(ref_path):
            print(f"FAIL: {case_id} is missing reference.png")
            sys.exit(1)
        if not os.path.exists(search_path):
            print(f"FAIL: {case_id} is missing search.png")
            sys.exit(1)
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        if meta.get('case_id') != case_id:
            print(f"FAIL: Metadata case_id {meta.get('case_id')} does not match directory {case_id}")
            sys.exit(1)
            
        if case_id in case_ids:
            print(f"FAIL: Duplicate case_id {case_id}")
            sys.exit(1)
        case_ids.add(case_id)
        
        arch = meta.get('architecture')
        if arch == 'DRAM': dram_count += 1
        elif arch == 'FinFET': finfet_count += 1
        else:
            print(f"FAIL: Unknown architecture {arch} in {case_id}")
            sys.exit(1)
            
        gt_x = meta.get('gt_x')
        gt_y = meta.get('gt_y')
        
        if gt_x is None or gt_y is None or not math.isfinite(gt_x) or not math.isfinite(gt_y):
            print(f"FAIL: Invalid ground truth in {case_id}")
            sys.exit(1)
            
        if not (0 <= gt_x <= 1000 and 0 <= gt_y <= 1000):
            print(f"FAIL: Ground truth {gt_x}, {gt_y} out of bounds in {case_id}")
            sys.exit(1)
            
        # Optional: check dimensions (we just read headers to save time, or trust the previous validation)
        # We will trust the previous validator script for exact dimensions to keep preflight fast,
        # but the spec says "search image is exactly 1000 x 1000, reference dimensions match canonical".
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        if search_img is None or search_img.shape != (1000, 1000):
            print(f"FAIL: search_img invalid or not 1000x1000 in {case_id}")
            sys.exit(1)
            
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None or ref_img.shape != (1000, 1000):
            print(f"FAIL: ref_img invalid or not 1000x1000 in {case_id}")
            sys.exit(1)
            
        seed = meta.get('seed')
        if seed in seeds:
            print(f"FAIL: Duplicate seed {seed} in {case_id}")
            sys.exit(1)
        seeds.add(seed)
        
        cases.append({
            'case_id': case_id,
            'arch': arch,
            'difficulty': meta.get('difficulty'),
            'seed': seed,
            'gt_x': gt_x,
            'gt_y': gt_y,
            'ref_path': ref_path,
            'search_path': search_path,
            'meta': meta
        })
        
    if dram_count != 30 or finfet_count != 30:
        print(f"FAIL: Architecture split not 30/30 (DRAM: {dram_count}, FinFET: {finfet_count})")
        sys.exit(1)
        
    print("Pre-flight check PASS. Dataset is structurally valid.\n")
    return sorted(cases, key=lambda x: x['case_id'])

def create_result_image(search_img, gt_x, gt_y, pred_x, pred_y, case_id, arch, diff, err_px, out_path):
    if len(search_img.shape) == 2:
        vis = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = search_img.copy()
        
    # Draw GT (Green circle, size 10, thickness 2)
    cv2.circle(vis, (int(gt_x), int(gt_y)), 10, (0, 255, 0), 2)
    cv2.putText(vis, "GT", (int(gt_x)+15, int(gt_y)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw Pred (Red X, size 10, thickness 2)
    cv2.drawMarker(vis, (int(pred_x), int(pred_y)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.putText(vis, "Pred", (int(pred_x)+15, int(pred_y)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw line between them
    cv2.line(vis, (int(gt_x), int(gt_y)), (int(pred_x), int(pred_y)), (255, 0, 0), 2)
    
    # Add labels
    cv2.putText(vis, f"Case: {case_id} | Arch: {arch} | Diff: {diff}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"Error: {err_px:.2f} px", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if err_px <= 10 else (0, 0, 255), 2)
    
    cv2.imwrite(out_path, vis)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run a SINGLE CASE DRY RUN using case_0001')
    parser.add_argument('--diagnostic-case', type=str, help='Run READ-ONLY diagnostic for a specific case')
    parser.add_argument('--dataset-dir', type=str, default=os.path.join('dataset', 'hackathon'), help='Path to dataset directory')
    args = parser.parse_args()
    
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.dataset_dir))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', os.path.basename(args.dataset_dir)))
    
    cases = validate_dataset_preflight(dataset_dir)
    
    if args.diagnostic_case:
        print("Running in DIAGNOSTIC mode.")
        cases = [c for c in cases if c['case_id'] == args.diagnostic_case]
        if not cases:
            print(f"FAIL: {args.diagnostic_case} not found.")
            sys.exit(1)
        
        c = cases[0]
        # Evaluate top 10 candidates for geometry diagnostic
        pipeline = HybridNavigationPipeline(top_k=10)
        ref_img = cv2.imread(c['ref_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(c['search_path'], cv2.IMREAD_GRAYSCALE)
        
        state = pipeline.run(ref_img, search_img)
        
        print("\n========================================")
        print("\n========================================")
        print("CASE-SPECIFIC LOCALIZATION DIAGNOSTIC")
        print("========================================")
        print(f"Case: {c['case_id']}")
        
        gspe = state.get('gspe_diagnostics', {})
        if not gspe:
            print("GSPE diagnostics missing!")
            sys.exit(1)
            
        res_raw = gspe['res_raw']
        res_lowfreq = gspe['res_lowfreq']
        res_hybrid = gspe['res_hybrid']
        best_scale_map = gspe['best_scale_map']
        best_rot_map = gspe['best_rot_map']
        best_w_map = gspe['best_w_map']
        best_h_map = gspe['best_h_map']
        
        # 1. Dimensions at GT
        gt_x, gt_y = c['gt_x'], c['gt_y']
        
        # To find GT rank, we must find the closest response peak to GT center.
        # But wait! Because templates can vary in size (and we padded the response map 
        # so the coordinate corresponds EXACTLY to the template center), the GT response
        # coordinate is exactly the GT center!
        gt_resp_x = gt_x
        gt_resp_y = gt_y
        
        ix = max(0, min(int(round(gt_resp_x)), res_hybrid.shape[1] - 1))
        iy = max(0, min(int(round(gt_resp_y)), res_hybrid.shape[0] - 1))
        
        ref_w = best_w_map[iy, ix]
        ref_h = best_h_map[iy, ix]
        gt_scale = best_scale_map[iy, ix]
        gt_rot = best_rot_map[iy, ix]
        
        print("\n--- BASELINE (GT vs Top-1) ---")
        _, max_val, _, max_loc = cv2.minMaxLoc(res_hybrid)
        print(f"GT score: {res_hybrid[iy, ix]:.6f}")
        
        shifted_up = np.roll(res_hybrid, 1, axis=0)
        shifted_down = np.roll(res_hybrid, -1, axis=0)
        shifted_left = np.roll(res_hybrid, 1, axis=1)
        shifted_right = np.roll(res_hybrid, -1, axis=1)
        local_max = (res_hybrid > shifted_up) & (res_hybrid > shifted_down) & \
                    (res_hybrid > shifted_left) & (res_hybrid > shifted_right)
        peaks_y, peaks_x = np.where(local_max)
        peak_scores = res_hybrid[peaks_y, peaks_x]
        sorted_indices = np.argsort(peak_scores)[::-1]
        sorted_peaks_x = peaks_x[sorted_indices]
        sorted_peaks_y = peaks_y[sorted_indices]
        
        y0, y1 = max(0, iy-2), min(res_hybrid.shape[0], iy+3)
        x0, x1 = max(0, ix-2), min(res_hybrid.shape[1], ix+3)
        window = res_hybrid[y0:y1, x0:x1]
        _, max_val_local, _, max_loc_local = cv2.minMaxLoc(window)
        local_peak_x = x0 + max_loc_local[0]
        local_peak_y = y0 + max_loc_local[1]
        
        gt_rank = "GT IS NOT A RESPONSE PEAK"
        for rank, (px, py) in enumerate(zip(sorted_peaks_x, sorted_peaks_y)):
            if px == local_peak_x and py == local_peak_y:
                gt_rank = rank + 1
                break
                
        print(f"GT rank: {gt_rank}")
        print(f"Top-1 score: {max_val:.6f}")
        print(f"Top-1 center coordinate: {max_loc}")
        
        print("\n--- GEOMETRY SEARCH ---")
        # pipeline initialized with 5 scales x 5 rotations = 25 hypotheses
        print("Number of hypotheses: 25")
        top1_scale = best_scale_map[max_loc[1], max_loc[0]]
        top1_rot = best_rot_map[max_loc[1], max_loc[0]]
        print(f"Best scale overall: {top1_scale}")
        print(f"Best rotation overall: {top1_rot}")
        print(f"Best hybrid score overall: {max_val:.6f}")
        
        print(f"GT score under best local hypothesis: {res_hybrid[iy, ix]:.6f}")
        print(f"GT rank under best local hypothesis: {gt_rank}")
        print(f"GT local geometry: Scale {gt_scale}, Rot {gt_rot}")
        
        print("\n--- TOP CANDIDATES ---")
        print("Rank |   X      |   Y      | Score    | Scale | Rotation")
        for i, (box, score) in enumerate(zip(gspe['boxes'], gspe['scores'])):
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            scale = box[4]
            rot = box[5]
            print(f"{i+1:<4} | {cx:<8.2f} | {cy:<8.2f} | {score:<8.6f} | {scale:<5.2f} | {rot:<8.2f}")
            
        gt_survives = False
        suppressor = None
        suppress_dist = None
        # V2: NMS radius is fixed at 10px to allow periodic candidates to survive
        nms_radius = 10
        
        for i, box in enumerate(gspe['boxes']):
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            dist = math.sqrt((cx - gt_resp_x)**2 + (cy - gt_resp_y)**2)
            if dist <= 2.0:
                gt_survives = True
                break
            elif dist <= nms_radius and suppressor is None:
                suppressor = i + 1
                suppress_dist = dist
                
        print(f"\nGT candidate before NMS: {'YES (Rank ' + str(gt_rank) + ')' if isinstance(gt_rank, int) else 'NO'}")
        if gt_survives:
            print("GT candidate after NMS: YES")
        else:
            print(f"GT candidate after NMS: NO (Suppressed by Rank {suppressor} at dist {suppress_dist:.2f}px)" if suppressor else "NO (Ranked below top_k)")
        
        print("\n========================================\n")
        sys.exit(0)
        
    if args.dry_run:
        print("Running in DRY-RUN mode. Limiting to case_0001.")
        cases = [c for c in cases if c['case_id'] == 'case_0001']
        if not cases:
            print("FAIL: case_0001 not found.")
            sys.exit(1)
            
    os.makedirs(os.path.join(out_dir, 'cases'), exist_ok=True)
    
    pipeline = HybridNavigationPipeline(top_k=10)
    
    results = []
    
    print(f"Starting batch evaluation for {len(cases)} cases...")
    
    for c in cases:
        case_id = c['case_id']
        print(f"Processing {case_id} (Seed: {c['seed']})...")
        
        case_out_dir = os.path.join(out_dir, 'cases', case_id)
        os.makedirs(case_out_dir, exist_ok=True)
        
        ref_img = cv2.imread(c['ref_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(c['search_path'], cv2.IMREAD_GRAYSCALE)
        
        state = pipeline.run(ref_img, search_img)
        
        res = {
            'case_id': case_id,
            'architecture': c['arch'],
            'difficulty': c['difficulty'],
            'spatial_region': c['meta'].get('spatial_region', 'unknown'),
            'seed': c['seed'],
            'gt_x': c['gt_x'],
            'gt_y': c['gt_y']
        }
        
        pred_json_path = os.path.join(case_out_dir, 'prediction.json')
        
        if 'error' in state:
            res['status'] = 'FAILED'
            res['error'] = state['error']
            res['runtime_seconds'] = state.get('runtime', 0.0)
            
            with open(pred_json_path, 'w') as f:
                json.dump(res, f, indent=4)
                
            results.append(res)
            print(f"  -> FAILED: {state['error']}")
            continue
            
        # Extract predictions
        final_coord = state.get('final_coord', [0, 0])
        pred_x, pred_y = float(final_coord[0]), float(final_coord[1])
        
        dx = pred_x - c['gt_x']
        dy = pred_y - c['gt_y']
        err_px = float(math.sqrt(dx**2 + dy**2))
        
        res['prediction'] = {'x': pred_x, 'y': pred_y}
        res['pred_x'] = pred_x
        res['pred_y'] = pred_y
        res['dx'] = dx
        res['dy'] = dy
        res['localization_error_px'] = err_px
        res['success_at_1px'] = bool(err_px <= 1.0)
        res['success_at_5px'] = bool(err_px <= 5.0)
        res['success_at_10px'] = bool(err_px <= 10.0)
        res['runtime_seconds'] = state.get('runtime', 0.0)
        res['status'] = 'SUCCESS'
        
        res['absolute_x_error'] = abs(dx)
        res['absolute_y_error'] = abs(dy)
        
        # Determine GSPE Recall
        # Check if GT is within 10px of any candidate in top-K
        gspe_diagnostics = state.get('gspe_diagnostics', {})
        res['gspe_gt_rank'] = -1
        if 'boxes' in gspe_diagnostics:
            for i, box in enumerate(gspe_diagnostics['boxes']):
                cx = box[0] + box[2] / 2.0
                cy = box[1] + box[3] / 2.0
                if math.sqrt((cx - c['gt_x'])**2 + (cy - c['gt_y'])**2) <= 10.0:
                    res['gspe_gt_rank'] = i + 1
                    break
        
        # Optional fields from pipeline state
        if 'classical_coord' in state:
            res['classical_coord'] = [float(state['classical_coord'][0]), float(state['classical_coord'][1])]
        else:
            res['classical_coord'] = None
            
        if 'ai_coord' in state:
            res['ai_coord'] = [float(state['ai_coord'][0]), float(state['ai_coord'][1])]
        else:
            res['ai_coord'] = None
            
        res['confidence'] = state.get('confidence', None)
        res['decision'] = state.get('decision', None)
        res['module_status'] = state.get('modules', {})
        
        with open(pred_json_path, 'w') as f:
            json.dump(res, f, indent=4)
            
        img_out_path = os.path.join(case_out_dir, 'result.png')
        create_result_image(search_img, c['gt_x'], c['gt_y'], pred_x, pred_y, case_id, c['arch'], c['difficulty'], err_px, img_out_path)
        
        results.append(res)
        print(f"  -> Success: err={err_px:.2f}px | time={res['runtime_seconds']:.2f}s")
        
    # Write Master CSV
    csv_path = os.path.join(out_dir, 'hackathon_results.csv')
    fieldnames = [
        'case_id', 'architecture', 'difficulty', 'spatial_region', 'seed', 
        'gt_x', 'gt_y', 'pred_x', 'pred_y', 
        'dx', 'dy', 'localization_error_px', 
        'success_at_1px', 'success_at_5px', 'success_at_10px', 
        'gspe_gt_rank', 'runtime_seconds', 'status'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Calculate Summary Metrics
    total_cases = len(results)
    successful_executions = [r for r in results if r['status'] == 'SUCCESS']
    failed_executions = [r for r in results if r['status'] == 'FAILED']
    
    dram_cases = [r for r in successful_executions if r['architecture'] == 'DRAM']
    finfet_cases = [r for r in successful_executions if r['architecture'] == 'FinFET']
    
    easy_cases = [r for r in successful_executions if r['difficulty'] == 'easy']
    mod_cases = [r for r in successful_executions if r['difficulty'] == 'moderate']
    hard_cases = [r for r in successful_executions if r['difficulty'] == 'hard']
    
    def calc_metrics(subset):
        if not subset:
            return {'mean': 0.0, 'median': 0.0, 'rmse': 0.0, 'p90': 0.0, 'p95': 0.0, 'success_rate': 0.0, 'count': 0}
        errs = [r['localization_error_px'] for r in subset]
        successes = sum(1 for r in subset if r['success_at_10px'])
        return {
            'count': len(subset),
            'mean': float(np.mean(errs)),
            'median': float(np.median(errs)),
            'rmse': float(np.sqrt(np.mean(np.array(errs)**2))),
            'p90': float(np.percentile(errs, 90)),
            'p95': float(np.percentile(errs, 95)),
            'success_rate': (successes / len(subset)) * 100.0
        }
        
    global_metrics = calc_metrics(successful_executions)
    mean_runtime = np.mean([r.get('runtime_seconds', 0.0) for r in results]) if results else 0.0
    
    summary = {
        'total_cases': total_cases,
        'pipeline_executions_successful': len(successful_executions),
        'pipeline_executions_failed': len(failed_executions),
        'dram_count': len([r for r in results if r['architecture'] == 'DRAM']),
        'finfet_count': len([r for r in results if r['architecture'] == 'FinFET']),
        'easy_count': len([r for r in results if r['difficulty'] == 'easy']),
        'moderate_count': len([r for r in results if r['difficulty'] == 'moderate']),
        'hard_count': len([r for r in results if r['difficulty'] == 'hard']),
        'mean_runtime_seconds': float(mean_runtime),
        'global_metrics': global_metrics,
        'by_architecture': {
            'DRAM': calc_metrics(dram_cases),
            'FinFET': calc_metrics(finfet_cases)
        },
        'by_difficulty': {
            'Easy': calc_metrics(easy_cases),
            'Moderate': calc_metrics(mod_cases),
            'Hard': calc_metrics(hard_cases)
        }
    }
    
    # Write JSON Summary
    with open(os.path.join(out_dir, 'hackathon_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
        
    # Write MD Summary
    with open(os.path.join(out_dir, 'hackathon_summary.md'), 'w') as f:
        f.write("# Hackathon Batch Evaluation Summary\n\n")
        f.write(f"- Total Cases: {summary['total_cases']}\n")
        f.write(f"- Pipeline Successful: {summary['pipeline_executions_successful']}\n")
        f.write(f"- Pipeline Failed: {summary['pipeline_executions_failed']}\n\n")
        f.write("## Global Metrics\n")
        f.write(f"- Mean Error: {global_metrics['mean']:.4f} px\n")
        f.write(f"- Median Error: {global_metrics['median']:.4f} px\n")
        f.write(f"- RMSE: {global_metrics['rmse']:.4f} px\n")
        f.write(f"- Success @ 10px: {global_metrics['success_rate']:.2f}%\n")
        f.write(f"- Mean Runtime: {summary['mean_runtime_seconds']:.4f} s\n")
        
    print("\n========================================")
    print("HACKATHON BATCH EVALUATION COMPLETE")
    print("========================================")
    print(f"\nCases discovered: {total_cases}")
    print(f"Cases processed: {total_cases}")
    print(f"DRAM: {summary['dram_count']}")
    print(f"FinFET: {summary['finfet_count']}\n")
    print(f"Pipeline executions successful: {summary['pipeline_executions_successful']}")
    print(f"Pipeline executions failed: {summary['pipeline_executions_failed']}\n")
    print(f"Localization success @ 10 px: {global_metrics['success_rate']:.2f}%\n")
    print(f"Mean Error: {global_metrics['mean']:.2f} px")
    print(f"Median Error: {global_metrics['median']:.2f} px")
    print(f"RMSE: {global_metrics['rmse']:.2f} px")
    print(f"P90 Error: {global_metrics['p90']:.2f} px")
    print(f"P95 Error: {global_metrics['p95']:.2f} px\n")
    print(f"Mean Runtime: {mean_runtime:.2f} s\n")
    print("Results:\noutputs/hackathon/hackathon_results.csv\n")
    print("Summary:\noutputs/hackathon/hackathon_summary.md")
    print("========================================")

    # FINAL VALIDATION
    valid = True
    if not os.path.exists(csv_path): valid = False
    if not os.path.exists(os.path.join(out_dir, 'hackathon_summary.md')): valid = False
    if not os.path.exists(os.path.join(out_dir, 'hackathon_summary.json')): valid = False
    
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            rows = list(csv.DictReader(f))
            if len(rows) != total_cases:
                valid = False
            ids = set(r['case_id'] for r in rows)
            if len(ids) != total_cases:
                valid = False
                
    for c in results:
        p = os.path.join(out_dir, 'cases', c['case_id'], 'prediction.json')
        if not os.path.exists(p):
            valid = False
        if c['status'] == 'SUCCESS':
            im = os.path.join(out_dir, 'cases', c['case_id'], 'result.png')
            if not os.path.exists(im):
                valid = False

    if valid:
        print("\nFINAL BATCH VALIDATION: PASS")
    else:
        print("\nFINAL BATCH VALIDATION: FAIL")

if __name__ == '__main__':
    main()
