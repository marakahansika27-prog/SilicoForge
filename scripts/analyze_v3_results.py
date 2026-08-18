import os
import csv
import json
import math
import numpy as np
from collections import defaultdict

def calc_metrics(subset):
    if not subset:
        return {'count': 0, 'mean': 0.0, 'median': 0.0, 'rmse': 0.0, 'min': 0.0, 'max': 0.0, 'p90': 0.0, 'p95': 0.0, 
                'success_1px': 0.0, 'success_5px': 0.0, 'success_10px': 0.0, 
                'gspe_top1': 0.0, 'gspe_top5': 0.0, 'gspe_top10': 0.0, 'gspe_top20': 0.0}
    
    errs = [float(r['localization_error_px']) for r in subset]
    succ_1 = sum(1 for r in subset if r.get('success_at_1px') == 'True')
    succ_5 = sum(1 for r in subset if r.get('success_at_5px') == 'True')
    succ_10 = sum(1 for r in subset if r.get('success_at_10px') == 'True')
    
    gspe_ranks = [int(r.get('gspe_gt_rank', -1)) for r in subset]
    top1 = sum(1 for rank in gspe_ranks if rank == 1)
    top5 = sum(1 for rank in gspe_ranks if 1 <= rank <= 5)
    top10 = sum(1 for rank in gspe_ranks if 1 <= rank <= 10)
    top20 = sum(1 for rank in gspe_ranks if 1 <= rank <= 20)
    
    n = len(subset)
    
    return {
        'count': n,
        'mean': float(np.mean(errs)),
        'median': float(np.median(errs)),
        'rmse': float(np.sqrt(np.mean(np.array(errs)**2))),
        'min': float(np.min(errs)),
        'max': float(np.max(errs)),
        'p90': float(np.percentile(errs, 90)),
        'p95': float(np.percentile(errs, 95)),
        'success_1px': (succ_1 / n) * 100.0,
        'success_5px': (succ_5 / n) * 100.0,
        'success_10px': (succ_10 / n) * 100.0,
        'gspe_top1': (top1 / n) * 100.0,
        'gspe_top5': (top5 / n) * 100.0,
        'gspe_top10': (top10 / n) * 100.0,
        'gspe_top20': (top20 / n) * 100.0,
    }

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'hackathon_v3'))
    csv_path = os.path.join(base_dir, 'hackathon_results.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'SUCCESS':
                results.append(row)
                
    if not results:
        print("No successful results found.")
        return
        
    total_cases = len(results)
    mean_runtime = np.mean([float(r['runtime_seconds']) for r in results])
    
    global_metrics = calc_metrics(results)
    
    by_region = defaultdict(list)
    by_arch = defaultdict(list)
    by_diff = defaultdict(list)
    
    for r in results:
        reg = r['spatial_region']
        if reg in ['left_boundary', 'right_boundary', 'top_boundary', 'bottom_boundary']:
            by_region['boundary'].append(r)
        elif 'corner' in reg:
            by_region['corner'].append(r)
        else:
            by_region[reg].append(r)
            
        by_region[reg + '_exact'].append(r)
        by_arch[r['architecture']].append(r)
        by_diff[r['difficulty']].append(r)
        
    region_metrics = {k: calc_metrics(v) for k, v in by_region.items()}
    arch_metrics = {k: calc_metrics(v) for k, v in by_arch.items()}
    diff_metrics = {k: calc_metrics(v) for k, v in by_diff.items()}
    
    # Generate Hackathon Summary MD
    with open(os.path.join(base_dir, 'hackathon_summary.md'), 'w') as f:
        f.write("# Hackathon V3 Spatial Batch Evaluation Summary\n\n")
        f.write(f"- Total Cases Processed: {total_cases}\n")
        f.write("## Overall Metrics\n")
        f.write(f"- Mean Error: {global_metrics['mean']:.4f} px\n")
        f.write(f"- Median Error: {global_metrics['median']:.4f} px\n")
        f.write(f"- RMSE: {global_metrics['rmse']:.4f} px\n")
        f.write(f"- Success @ 1px: {global_metrics['success_1px']:.2f}%\n")
        f.write(f"- Success @ 5px: {global_metrics['success_5px']:.2f}%\n")
        f.write(f"- Success @ 10px: {global_metrics['success_10px']:.2f}%\n")
        f.write(f"- Mean Runtime: {mean_runtime:.4f} s\n\n")
        
        f.write("## Spatial Breakdown (Aggregated)\n")
        for k in ['center', 'interior', 'boundary', 'corner', 'random']:
            if k in region_metrics:
                m = region_metrics[k]
                f.write(f"### {k.upper()} (N={m['count']})\n")
                f.write(f"- Error: Mean {m['mean']:.2f}px | Median {m['median']:.2f}px | Max {m['max']:.2f}px\n")
                f.write(f"- Success @ 10px: {m['success_10px']:.2f}%\n\n")

    # Generate Full Benchmark Report
    with open(os.path.join(base_dir, 'V3_SPATIAL_BENCHMARK_REPORT.md'), 'w') as f:
        f.write("# Drift-Sense V3 Spatial Generalization Benchmark Report\n\n")
        f.write("## 1. Dataset Design\n")
        f.write("The V3 dataset consists of exactly 60 cases specifically designed to evaluate localization performance across varied spatial contexts (center, interior, boundary, corners, and random). The generator uses corner-proximal bounds to ensure all physical target coordinates remain valid for the given patch geometry.\n\n")
        
        f.write("## 2. Overall Pipeline Results\n")
        f.write(f"Total Successful Pipeline Executions: {total_cases}\n")
        f.write(f"Mean Error: {global_metrics['mean']:.2f} px\n")
        f.write(f"Median Error: {global_metrics['median']:.2f} px\n")
        f.write(f"RMSE: {global_metrics['rmse']:.2f} px\n")
        f.write(f"Success @ 10px: {global_metrics['success_10px']:.2f}%\n\n")
        
        f.write("## 3. Spatial Generalization Results\n")
        f.write("| Region | Count | Success @10px | Mean (px) | Median (px) | Max (px) | GSPE Top-10 Recall |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for k in ['center', 'interior', 'boundary', 'corner', 'random']:
            if k in region_metrics:
                m = region_metrics[k]
                f.write(f"| {k.capitalize()} | {m['count']} | {m['success_10px']:.1f}% | {m['mean']:.2f} | {m['median']:.2f} | {m['max']:.2f} | {m['gspe_top10']:.1f}% |\n")
        
        f.write("\n## 4. GSPE Candidate Generation (Periodic Ambiguity Analysis)\n")
        f.write("This table shows whether the true ground-truth region was proposed by the GSPE coarse search. Low Top-1 recall with high Top-10 recall indicates severe periodic ambiguity resolving failures.\n\n")
        f.write("| Region | Top-1 | Top-5 | Top-10 | Top-20 |\n")
        f.write("|---|---|---|---|---|\n")
        for k in ['center', 'interior', 'boundary', 'corner', 'random']:
            if k in region_metrics:
                m = region_metrics[k]
                f.write(f"| {k.capitalize()} | {m['gspe_top1']:.1f}% | {m['gspe_top5']:.1f}% | {m['gspe_top10']:.1f}% | {m['gspe_top20']:.1f}% |\n")
                
        f.write("\n## 5. Architecture Results\n")
        for arch, m in arch_metrics.items():
            f.write(f"- **{arch}** (N={m['count']}): {m['success_10px']:.1f}% Success, Mean {m['mean']:.2f}px\n")
            
        f.write("\n## 6. Difficulty Results\n")
        for diff, m in diff_metrics.items():
            f.write(f"- **{diff.capitalize()}** (N={m['count']}): {m['success_10px']:.1f}% Success, Mean {m['mean']:.2f}px\n")
            
        f.write("\n## 7. Conclusions & Next Steps\n")
        f.write("*(To be filled by analyzing the generated metrics above)*\n")

    print(f"Generated V3_SPATIAL_BENCHMARK_REPORT.md and hackathon_summary.md in {base_dir}")

if __name__ == '__main__':
    main()
