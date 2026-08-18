import os
import sys
import cv2
import json
import csv
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.integration.pipeline import HybridNavigationPipeline

def generate_benchmark_case(seed, architecture, difficulty):
    rs = np.random.RandomState(seed)
    
    base_size = 10240
    base_img = np.zeros((base_size, base_size), dtype=np.float32)
    
    # Finite array (bounded by scribe lines / peripheral blocks)
    # Array bounds: 2100 to 8100
    start_x, end_x = 2100, 8100
    start_y, end_y = 2100, 8100
    
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
            
    # Draw a simple peripheral circuit outline in the boundary region (physically plausible)
    cv2.rectangle(base_img, (start_x-200, start_y-200), (end_x+200, end_y+200), 50, 40)
            
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    gradient = cv2.morphologyEx(base_img, cv2.MORPH_GRADIENT, kernel)
    sem_img = np.clip(base_img + gradient * 1.5, 0, 255).astype(np.float32)
    
    noise_level = 50
    blur = 0
    if difficulty == "easy":
        noise_level = 20
    elif difficulty == "moderate":
        noise_level = 50
        blur = 3
    elif difficulty == "hard":
        noise_level = 90
        blur = 5
        
    # Controlled Target Sampling: Visible Boundary (Condition C)
    # Extract near the edge of the array (2100 or 8100)
    offset_x = 1800 if rs.rand() > 0.5 else 7400
    offset_y = 1800 if rs.rand() > 0.5 else 7400
    offset_x += rs.randint(-150, 150)
    offset_y += rs.randint(-150, 150)
    
    ref_float = sem_img[offset_y:offset_y+900, offset_x:offset_x+900].copy()
    
    # Check boundary visibility mathematically (for diagnostic output)
    dist_to_left = offset_x + 450 - 2100
    dist_to_right = 8100 - (offset_x + 450)
    dist_to_top = offset_y + 450 - 2100
    dist_to_bottom = 8100 - (offset_y + 450)
    min_dist_to_boundary = min(abs(dist_to_left), abs(dist_to_right), abs(dist_to_top), abs(dist_to_bottom))
    is_visible = (offset_x < 2100 or offset_x + 900 > 8100 or offset_y < 2100 or offset_y + 900 > 8100)
    assert is_visible, f"Target sampling failed to make boundary visible! offset=({offset_x}, {offset_y})"
    
    gt_x = (offset_x + 450.0) / 10.0
    gt_y = (offset_y + 450.0) / 10.0
    
    search_float = cv2.resize(sem_img, (1024, 1024), interpolation=cv2.INTER_AREA)
    if blur > 0: search_float = cv2.GaussianBlur(search_float, (blur, blur), 0)
        
    ref_noise = rs.poisson(ref_float / 255.0 * 20) / 20 * 255
    ref_img = np.clip(ref_float + ref_noise - 128, 0, 255).astype(np.uint8)
    
    search_noise = rs.poisson(search_float / 255.0 * noise_level) / noise_level * 255
    search_img = np.clip(search_float + search_noise - 128, 0, 255).astype(np.uint8)
    
    return ref_img, search_img, gt_x, gt_y

def classify_failure(state, error):
    if 'error' in state:
        return "PIPELINE_CRASH"
    if not state['modules'].get('SRAE', False):
        return "SRAE_FAILURE"
    
    # Basic heuristic
    classical_coord = state.get('classical_coord', [0,0])
    ai_coord = state.get('ai_coord', [0,0])
    final_coord = state.get('final_coord', [0,0])
    
    c_err = np.linalg.norm(classical_coord - state['ground_truth'])
    if c_err > 50.0:
        return "PERIODIC_AMBIGUITY"
    if c_err > 5.0:
        return "CLASSICAL_LOCALIZATION_FAILURE"
        
    if state.get('decision') == 'AI_REFINED' and error > c_err:
        return "AI_REFINEMENT_FAILURE"
        
    if state.get('confidence', 1.0) < 0.90:
        return "LOW_CONFIDENCE"
        
    return "OTHER"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tolerance', type=float, default=0.5, help='Success threshold')
    parser.add_argument('--smoke-test', action='store_true', help='Run only 5 cases')
    args = parser.parse_args()
    
    cases = []
    
    # Define Distribution
    # DRAM: 8 easy, 8 moderate, 6 hard
    for i in range(8): cases.append(("DRAM", "easy"))
    for i in range(8): cases.append(("DRAM", "moderate"))
    for i in range(6): cases.append(("DRAM", "hard"))
    
    # FinFET: 8 easy, 6 moderate, 4 hard
    for i in range(8): cases.append(("FinFET", "easy"))
    for i in range(6): cases.append(("FinFET", "moderate"))
    for i in range(4): cases.append(("FinFET", "hard"))
    
    if args.smoke_test:
        cases = cases[:5]
        print(f"Running SMOKE TEST with {len(cases)} cases.")
        
    os.makedirs("benchmark/cases", exist_ok=True)
    os.makedirs("benchmark/results/visualizations", exist_ok=True)
    
    pipeline = HybridNavigationPipeline()
    
    results = []
    
    for idx, (arch, diff) in enumerate(cases):
        case_id = f"case_{idx+1:03d}"
        seed = 1000 + idx + 1
        print(f"[{case_id}] Generating {arch} {diff} (Seed: {seed})...")
        
        ref_img, search_img, gt_x, gt_y = generate_benchmark_case(seed, arch, diff)
        gt = np.array([gt_x, gt_y], dtype=np.float32)
        
        case_dir = f"benchmark/cases/{case_id}"
        os.makedirs(case_dir, exist_ok=True)
        cv2.imwrite(os.path.join(case_dir, "reference.png"), ref_img)
        cv2.imwrite(os.path.join(case_dir, "search.png"), search_img)
        
        # Run Pipeline
        t0 = time.time()
        state = pipeline.run(ref_img, search_img)
        runtime = time.time() - t0
        
        state['ground_truth'] = gt
        
        if 'error' in state:
            res = {
                'case_id': case_id, 'architecture': arch, 'difficulty': diff, 'seed': seed,
                'gt_x': gt_x, 'gt_y': gt_y, 'success': False, 'failure_reason': 'PIPELINE_CRASH'
            }
            results.append(res)
            print(f"  [ERROR] {state['error']}")
            continue
            
        c_coord = state['classical_coord']
        a_coord = state['ai_coord']
        f_coord = state['final_coord']
        
        c_err = float(np.linalg.norm(c_coord - gt))
        a_err = float(np.linalg.norm(a_coord - gt))
        f_err = float(np.linalg.norm(f_coord - gt))
        
        if c_err > 1e-6:
            imp = ((c_err - f_err) / c_err) * 100.0
        else:
            imp = 0.0
            
        success = f_err <= args.tolerance
        fail_reason = "" if success else classify_failure(state, f_err)
        
        # AI Specifics
        ai_res_vector = a_coord - c_coord
        ai_mag = float(np.linalg.norm(ai_res_vector))
        
        res = {
            'case_id': case_id,
            'architecture': arch,
            'difficulty': diff,
            'seed': seed,
            'gt_x': float(gt_x),
            'gt_y': float(gt_y),
            'classical_x': float(c_coord[0]),
            'classical_y': float(c_coord[1]),
            'classical_error': c_err,
            'ai_dx': float(ai_res_vector[0]),
            'ai_dy': float(ai_res_vector[1]),
            'ai_residual_magnitude': ai_mag,
            'ai_confidence': float(state['confidence']),
            'ai_x': float(a_coord[0]),
            'ai_y': float(a_coord[1]),
            'ai_error': a_err,
            'final_x': float(f_coord[0]),
            'final_y': float(f_coord[1]),
            'final_error': f_err,
            'improvement_percent': imp,
            'decision': state['decision'],
            'runtime_seconds': runtime,
            'success': success,
            'failure_reason': fail_reason
        }
        results.append(res)
        
        print(f"  Class Err: {c_err:.4f} | Final Err: {f_err:.4f} | Dec: {state['decision']} | Pass: {success}")
        
        # Viz for 1st success or 1st fail
        is_first_success = success and not any(r.get('success', False) for r in results[:-1])
        is_first_fail = not success and not any(not r.get('success', True) for r in results[:-1])
        
        if is_first_success or is_first_fail:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(ref_img, cmap='gray')
            ax[0].set_title(f"Reference ({arch} {diff})")
            
            ax[1].imshow(search_img, cmap='gray')
            # Zoom into 100x100 region around ground truth for clarity
            half_w = 100
            ax[1].set_xlim(gt_x - half_w, gt_x + half_w)
            ax[1].set_ylim(gt_y + half_w, gt_y - half_w) # Invert Y for image
            
            ax[1].plot(gt_x, gt_y, 'g+', markersize=15, label='Ground Truth')
            ax[1].plot(c_coord[0], c_coord[1], 'rx', markersize=12, label='Classical')
            ax[1].plot(a_coord[0], a_coord[1], 'bo', markersize=8, fillstyle='none', label='AI')
            ax[1].set_title(f"Target Region\nErr: {f_err:.3f} px")
            ax[1].legend()
            
            name = "success" if success else "failure"
            plt.savefig(f"benchmark/results/visualizations/{case_id}_{name}.png")
            plt.close()
            
    # Save CSV
    keys = results[0].keys() if results else []
    with open("benchmark/results/benchmark_results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    total_valid = len([r for r in results if 'classical_error' in r])
    
    if total_valid > 0:
        c_errs = [r['classical_error'] for r in results if 'classical_error' in r]
        a_errs = [r['ai_error'] for r in results if 'ai_error' in r]
        f_errs = [r['final_error'] for r in results if 'final_error' in r]
        imps = [r['improvement_percent'] for r in results if 'improvement_percent' in r]
        runtimes = [r['runtime_seconds'] for r in results if 'runtime_seconds' in r]
        
        ai_refined = sum(1 for r in results if r.get('decision') == 'AI_REFINED')
        class_fall = sum(1 for r in results if r.get('decision') == 'CLASSICAL_FALLBACK')
        
        c_wins = sum(1 for r in results if r.get('classical_error', 0) < r.get('ai_error', 0))
        a_wins = sum(1 for r in results if r.get('ai_error', 0) < r.get('classical_error', 0))
        f_wins = sum(1 for r in results if r.get('final_error', 0) < r.get('classical_error', 0))
        
        summary = {
            'Total Cases': len(results),
            'Successful': successful,
            'Accuracy @ ±0.5 px': f"{(successful/len(results))*100:.2f}%",
            'Classical Mean Error': np.mean(c_errs),
            'AI Mean Error': np.mean(a_errs),
            'Fusion Mean Error': np.mean(f_errs),
            'Classical Median Error': np.median(c_errs),
            'AI Median Error': np.median(a_errs),
            'Fusion Median Error': np.median(f_errs),
            'Classical RMSE': np.sqrt(np.mean(np.array(c_errs)**2)),
            'AI RMSE': np.sqrt(np.mean(np.array(a_errs)**2)),
            'Fusion RMSE': np.sqrt(np.mean(np.array(f_errs)**2)),
            'AI_REFINED': ai_refined,
            'CLASSICAL_FALLBACK': class_fall,
            'Classical Wins': c_wins,
            'AI Wins': a_wins,
            'Fusion Wins': f_wins,
            'Mean Improvement': np.mean(imps),
            'Median Improvement': np.median(imps),
            'Best Improvement': np.max(imps),
            'Worst Improvement': np.min(imps),
            'Mean Runtime': np.mean(runtimes),
            'Median Runtime': np.median(runtimes),
            'Maximum Runtime': np.max(runtimes),
        }
        
        with open("benchmark/results/benchmark_summary.json", "w") as f:
            json.dump(summary, f, indent=4)
            
        print("\n========================================")
        print("DRIFT-SENSE 40 CASE BENCHMARK")
        print("========================================")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
                
        if successful == len(results):
            print("\nNOTE: No genuine failures occurred, so no failure visualization was generated.")
            
if __name__ == "__main__":
    main()
