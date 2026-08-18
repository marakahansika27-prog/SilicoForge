import os
import sys
import csv
import math
import time
import numpy as np
import cv2
import contextlib
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.integration.pipeline import HybridNavigationPipeline
from dataset.generator import HackathonDatasetGenerator

def error_fn(x, y, gt_x, gt_y):
    return math.sqrt((x - gt_x)**2 + (y - gt_y)**2)

def calc_metrics(errors):
    if not errors: return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    errs = np.array(errors)
    return (
        np.mean(errs),
        np.median(errs),
        np.sqrt(np.mean(errs**2)),
        np.percentile(errs, 90),
        np.percentile(errs, 95),
        np.mean(errs <= 1.0) * 100,
        np.mean(errs <= 5.0) * 100,
        np.mean(errs <= 10.0) * 100,
        np.mean(errs <= 25.0) * 100,
        np.mean(errs <= 50.0) * 100,
        np.max(errs)
    )

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'hackathon_v3'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'hackathon_v3', 'phase44_45case'))
    
    manifest_path = os.path.join(out_dir, 'dataset_manifest_45.csv')
    if not os.path.exists(manifest_path):
        print("ERROR: dataset_manifest_45.csv not found. Please ensure phase44_45case.py has been run.")
        return
        
    cases_manifest = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cases_manifest.append(row)
            
    # CRITICAL FIX: Use the actual Phase 28 Hybrid Navigation Pipeline
    pipeline = HybridNavigationPipeline(top_k=1)
    
    all_results = []
    
    for idx, c in enumerate(cases_manifest):
        case_id = c['case_id']
        arch = c['architecture']
        gt_x, gt_y = float(c['gt_x']), float(c['gt_y'])
        seed = int(c['seed'])
        
        ref_path = os.path.join(base_dir, arch.lower(), case_id, 'reference.png')
        search_path = os.path.join(base_dir, arch.lower(), case_id, 'search.png')
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        
        gen = HackathonDatasetGenerator(seed=seed)
        ref_macro, _, _ = gen.generate_case(case_id, arch, c['difficulty'], version="v3", spatial_region=c['spatial_region'], ref_size=4000)
        
        with contextlib.redirect_stdout(io.StringIO()):
            # CRITICAL FIX: Pass ref_macro to the pipeline so the coarse search executes
            state = pipeline.run(ref_img, search_img, ref_macro=ref_macro)
            
        pred_x, pred_y = state['final_coord']
        err = error_fn(pred_x, pred_y, gt_x, gt_y)
        
        all_results.append({
            'case_id': case_id,
            'arch': arch,
            'region': c['spatial_region'],
            'err': err
        })
        
    m45 = calc_metrics([r['err'] for r in all_results])
    
    dram_45 = [r['err'] for r in all_results if r['arch'] == 'DRAM']
    finfet_45 = [r['err'] for r in all_results if r['arch'] == 'FinFET']
    
    m_dram = calc_metrics(dram_45)
    m_finfet = calc_metrics(finfet_45)
    
    with open(os.path.join(out_dir, 'phase44_45case_results.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['case_id', 'architecture', 'region', 'error'])
        for r in all_results:
            w.writerow([r['case_id'], r['arch'], r['region'], r['err']])
            
    with open(os.path.join(out_dir, 'PHASE44_45CASE_REPORT.md'), 'w', encoding='utf-8') as f:
        f.write("# Phase 44 45-Case Final Evaluation\n\n")
        f.write(f"- Total Cases: {len(all_results)}\n")
        f.write(f"- Mean Error: {m45[0]:.2f} px\n")
        f.write(f"- Median Error: {m45[1]:.2f} px\n")
        f.write(f"- RMSE: {m45[2]:.2f} px\n")
        f.write(f"- P90: {m45[3]:.2f} px\n")
        f.write(f"- P95: {m45[4]:.2f} px\n")
        f.write(f"- Success <=1px: {m45[5]:.1f}%\n")
        f.write(f"- Success <=5px: {m45[6]:.1f}%\n")
        f.write(f"- Success <=10px: {m45[7]:.1f}%\n")
        f.write(f"- Success <=25px: {m45[8]:.1f}%\n")
        f.write(f"- Success <=50px: {m45[9]:.1f}%\n")
        f.write("\n## Architecture (45-Case)\n")
        f.write(f"- DRAM ({len(dram_45)} cases): {m_dram[7]:.1f}% success, {m_dram[0]:.2f}px mean\n")
        f.write(f"- FinFET ({len(finfet_45)} cases): {m_finfet[7]:.1f}% success, {m_finfet[0]:.2f}px mean\n")

    print("\n============================================================")
    print("45-CASE FINAL EVALUATION")
    print("============================================================")
    print(f"\nTotal Cases: {len(all_results)}")
    print(f"\nMean Error: {m45[0]:.2f}")
    print(f"Median Error: {m45[1]:.2f}")
    print(f"RMSE: {m45[2]:.2f}")
    print(f"P90: {m45[3]:.2f}")
    print(f"P95: {m45[4]:.2f}")
    
    print(f"\n<=1px: {m45[5]:.1f}%")
    print(f"<=5px: {m45[6]:.1f}%")
    print(f"<=10px: {m45[7]:.1f}%")
    print(f"<=25px: {m45[8]:.1f}%")
    print(f"<=50px: {m45[9]:.1f}%")
    
    print("\nDRAM:")
    print(f"Cases: {len(dram_45)}")
    print(f"Mean Error: {m_dram[0]:.2f}")
    print(f"Success <=10px: {m_dram[7]:.1f}%")
    
    print("\nFinFET:")
    print(f"Cases: {len(finfet_45)}")
    print(f"Mean Error: {m_finfet[0]:.2f}")
    print(f"Success <=10px: {m_finfet[7]:.1f}%")
    print("\n============================================================")

if __name__ == "__main__":
    main()
