import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from glob import glob

# Import production Frozen Champion mathematics directly
# DO NOT modify src/coarse_search/gspe.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.coarse_search.gspe import GlobalSearchProposalEngine

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

def main():
    print("JSON serialization hardening: ENABLED")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    source_dir = os.path.join(base_dir, 'benchmark', 'results', 'deep_interior_context')
    out_dir = os.path.join(base_dir, 'benchmark', 'results', 'global_disambiguation')
    ensure_dir(out_dir)
    
    vis_dir = os.path.join(out_dir, 'top5_candidate_visualizations')
    ensure_dir(vis_dir)
    
    # Selected dimensions
    target_D = [1000, 1250, 1500, 2000]
    target_ctx = [900, 1350, 1750, 2000]
    
    # We will search the source_dir for all metadata.json files that match these criteria
    search_pattern = os.path.join(source_dir, '**', 'metadata.json')
    all_metadata_files = glob(search_pattern, recursive=True)
    
    selected_cases = []
    for f in all_metadata_files:
        with open(f, 'r') as meta_file:
            meta = json.load(meta_file)
            
        if meta.get('D') in target_D and meta.get('context_size') in target_ctx:
            # We found a target case
            case_dir = os.path.dirname(f)
            # Verify source artifacts exist
            if os.path.exists(os.path.join(case_dir, 'reference.png')) and os.path.exists(os.path.join(case_dir, 'search.png')):
                selected_cases.append((meta, case_dir))
    
    # Structural validation
    expected_count = len(target_D) * len(target_ctx) * 2 * 3 # 4 * 4 * 2 * 3 = 96
    print(f"--- STRUCTURAL VALIDATION ---")
    print(f"Target D values: {target_D}")
    print(f"Target Contexts: {target_ctx}")
    print(f"Expected Cases : {expected_count}")
    print(f"Found Cases    : {len(selected_cases)}")
    assert len(selected_cases) == expected_count, f"Missing cases from deep_interior_context. Expected {expected_count}, found {len(selected_cases)}"
    print(f"All selected cases have valid source artifacts.")
    print(f"--- END VALIDATION ---")
    
    # Initialize GSPE with Top-K extraction
    gspe = GlobalSearchProposalEngine(top_k=5)
    
    results = []
    
    for meta, case_dir in selected_cases:
        case_id = meta['case_id']
        ctx = meta['context_size']
        d = meta['D']
        arch = meta['architecture']
        diff = meta['difficulty']
        
        gt_x = meta['gt_x']
        gt_y = meta['gt_y']
        
        # Load artifacts
        ref_img = cv2.imread(os.path.join(case_dir, 'reference.png'), cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(os.path.join(case_dir, 'search.png'), cv2.IMREAD_GRAYSCALE)
        
        if ref_img is None or search_img is None:
            print(f"WARNING: missing image files for {case_id}")
            continue
            
        # Run GSPE exactly identically to production
        # This yields the top-5 candidates from the identical response surface.
        inputs = {'reference': ref_img, 'search': search_img}
        out = gspe.run(inputs)
        
        boxes = out['boxes'] # List of (x, y, w, h)
        scores = out['scores']
        
        # We need final_x, final_y exactly as in pipeline
        new_w = boxes[0][2]
        new_h = boxes[0][3]
        
        candidates = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            # Box is subpixel integer peak + delta
            # The coordinate is the box center + subpixel delta, plus half width
            c_x = float(box[0]) + new_w / 2.0
            c_y = float(box[1]) + new_h / 2.0
            
            err = np.linalg.norm(np.array([c_x, c_y]) - np.array([gt_x, gt_y]))
            dx = c_x - gt_x
            dy = c_y - gt_y
            
            nearest_shift_x = float(round(dx / 3.0) * 3.0)
            periodic_residual_x = float(abs(dx - nearest_shift_x))
            nearest_shift_y = float(round(dy / 3.0) * 3.0)
            periodic_residual_y = float(abs(dy - nearest_shift_y))
            
            is_periodic = bool((periodic_residual_x < 0.1) and (periodic_residual_y < 0.1) and (err > 1.0))
            
            candidates.append({
                'rank': i + 1,
                'x': float(c_x),
                'y': float(c_y),
                'score': float(score),
                'distance_to_gt': float(err),
                'dx': float(dx),
                'dy': float(dy),
                'nearest_shift_x': nearest_shift_x,
                'periodic_residual_x': periodic_residual_x,
                'nearest_shift_y': nearest_shift_y,
                'periodic_residual_y': periodic_residual_y,
                'is_periodic': is_periodic
            })
            
        # Oracle candidate: candidate with minimum distance to GT
        oracle_idx = np.argmin([c['distance_to_gt'] for c in candidates])
        oracle_cand = candidates[oracle_idx]
        
        top1 = candidates[0]
        top1_is_correct = bool(top1['distance_to_gt'] <= 0.5)
        
        gt_in_top3 = any(c['distance_to_gt'] <= 0.5 for c in candidates[:3])
        gt_in_top5 = any(c['distance_to_gt'] <= 0.5 for c in candidates[:5])
        
        res_dict = {
            'case_id': case_id,
            'context_size': ctx,
            'D': d,
            'architecture': arch,
            'difficulty': diff,
            'top1_score': top1['score'],
            'top2_score': candidates[1]['score'] if len(candidates) > 1 else None,
            'top3_score': candidates[2]['score'] if len(candidates) > 2 else None,
            'top1_minus_top2': top1['score'] - candidates[1]['score'] if len(candidates) > 1 else None,
            'top1_error': top1['distance_to_gt'],
            'top1_is_correct': top1_is_correct,
            'top1_periodic_ambiguity': top1['is_periodic'],
            'oracle_best_of_5_error': oracle_cand['distance_to_gt'],
            'gt_in_top3': gt_in_top3,
            'gt_in_top5': gt_in_top5,
            'candidates': candidates
        }
        results.append(res_dict)
        
        # Visualizations
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(search_img, cmap='gray')
        ax[0].set_title(f"Top-5 Candidates Overlay\n{arch} {diff} (Ctx:{ctx}, D:{d})")
        # GT Box (Green)
        gt_rect = Rectangle((gt_x - new_w/2, gt_y - new_h/2), new_w, new_h, linewidth=2, edgecolor='g', facecolor='none', label='GT')
        ax[0].add_patch(gt_rect)
        
        # Candidate Boxes
        colors = ['r', 'orange', 'yellow', 'm', 'c']
        for i, c in enumerate(candidates):
            rect = Rectangle((c['x'] - new_w/2, c['y'] - new_h/2), new_w, new_h, linewidth=2, edgecolor=colors[i], linestyle='--', facecolor='none', label=f"Top-{i+1} (s:{c['score']:.3f})")
            ax[0].add_patch(rect)
        ax[0].legend()
        
        # Text summary
        ax[1].axis('off')
        info = f"--- ORACLE DIAGNOSTIC ---\n"
        info += f"Case: {case_id}\n\n"
        info += f"Top-1 Correct: {top1_is_correct}\n"
        info += f"GT in Top-5  : {gt_in_top5}\n\n"
        for c in candidates:
            info += f"Rank {c['rank']}: Score={c['score']:.4f} Err={c['distance_to_gt']:.2f}px\n"
        
        ax[1].text(0.05, 0.5, info, fontsize=12, family='monospace', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{case_id}_top5.png"), dpi=100)
        plt.close()
        
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'candidates'} for r in results])
    df.to_csv(os.path.join(out_dir, "global_disambiguation_results.csv"), index=False)
    
    summary = {
        'total_cases': len(df),
        'top1_accuracy': float(df['top1_is_correct'].mean() * 100),
        'gt_in_top3_rate': float(df['gt_in_top3'].mean() * 100),
        'gt_in_top5_rate': float(df['gt_in_top5'].mean() * 100),
        'mean_top1_error': float(df['top1_error'].mean()),
        'mean_oracle_best_of_5_error': float(df['oracle_best_of_5_error'].mean()),
        'top1_rmse': float(np.sqrt((df['top1_error']**2).mean())),
        'periodic_ambiguity_rate': float(df['top1_periodic_ambiguity'].mean() * 100),
        'mean_top1_minus_top2_margin': float(df['top1_minus_top2'].dropna().mean())
    }
    with open(os.path.join(out_dir, "global_disambiguation_summary.json"), "w") as f:
        json.dump(make_json_safe(summary), f, indent=4)
        
    # Analysis Plots
    # 1. Candidate rank distribution (where does the GT actually fall?)
    rank_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 'Not in Top 5': 0}
    for r in results:
        found_rank = None
        for c in r['candidates']:
            if c['distance_to_gt'] <= 0.5:
                found_rank = c['rank']
                break
        if found_rank:
            rank_counts[found_rank] += 1
        else:
            rank_counts['Not in Top 5'] += 1
            
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([str(k) for k in rank_counts.keys()], rank_counts.values(), color=['blue', 'green', 'yellow', 'orange', 'red', 'gray'])
    ax.set_title('True Ground Truth Rank Distribution (Oracle)', fontweight='bold')
    ax.set_ylabel('Count of Cases')
    ax.set_xlabel('Rank (1 is Top-1)')
    plt.savefig(os.path.join(out_dir, "candidate_rank_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Score Margin
    fig, ax = plt.subplots(figsize=(8, 6))
    correct_margins = df[df['top1_is_correct']]['top1_minus_top2'].dropna()
    incorrect_margins = df[~df['top1_is_correct']]['top1_minus_top2'].dropna()
    ax.hist(correct_margins, bins=20, alpha=0.5, label='Top-1 Correct', color='g')
    ax.hist(incorrect_margins, bins=20, alpha=0.5, label='Top-1 Wrong', color='r')
    ax.set_xlabel('Score Margin (Top-1 - Top-2)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_title('Top-1 vs Top-2 Margin by Outcome', fontweight='bold')
    plt.savefig(os.path.join(out_dir, "candidate_score_margin.png"), dpi=150)
    plt.close()
    
    # Generate Report
    report = f"""# Global Disambiguation Ablation Report

## 1. Experimental Objective
Determine whether deep-interior failures are caused by GSPE selecting an incorrect periodic candidate when multiple structurally similar candidates exist (candidate-selection/disambiguation problem) or if the Ground Truth simply does not yield a high NCC response.

## 2. Oracle Top-K Diagnostic
*NOTE: The Oracle Top-K explicitly uses the known Ground Truth to select the "best" candidate from the Top-5 pool. This is a purely mathematical diagnostic to measure the theoretical upper-bound of perfect disambiguation. It is NOT a deployable algorithm.*

### Summary Metrics
- **Total Diagnostic Cases:** {summary['total_cases']}
- **Frozen Champion Top-1 Accuracy:** {summary['top1_accuracy']:.2f}%
- **Oracle GT-in-Top-3 Rate:** {summary['gt_in_top3_rate']:.2f}%
- **Oracle GT-in-Top-5 Rate:** {summary['gt_in_top5_rate']:.2f}%

### Error Comparison
- **Mean Top-1 Error:** {summary['mean_top1_error']:.2f} px
- **Mean Oracle Best-of-5 Error:** {summary['mean_oracle_best_of_5_error']:.2f} px
- **Top-1 Periodic Ambiguity Rate:** {summary['periodic_ambiguity_rate']:.2f}%

### Score Margins
- **Mean Margin (Top 1 - Top 2):** {summary['mean_top1_minus_top2_margin']:.5f}

## 3. Conclusion
"""
    if summary['gt_in_top5_rate'] > summary['top1_accuracy'] + 20:
        report += "The Ground Truth is frequently present in the Top-5 candidates even when the Frozen Champion selects an incorrect Top-1 peak. This confirms that the periodic failure is primarily a **disambiguation problem**. The response surface is generating a valid peak at the GT location, but other lattice cells are producing higher raw scores due to noise or insufficient boundary inclusion.\n"
    else:
        report += "The Ground Truth is generally absent from the Top-5 candidates when the Frozen Champion fails. This suggests a failure of the response surface itself, where the true location is suppressed entirely, rather than a candidate ambiguity problem.\n"
        
    with open(os.path.join(out_dir, "GLOBAL_DISAMBIGUATION_REPORT.md"), "w") as f:
        f.write(report)
        
    print("\nGLOBAL DISAMBIGUATION ABLATION COMPLETE.")

if __name__ == '__main__':
    main()
