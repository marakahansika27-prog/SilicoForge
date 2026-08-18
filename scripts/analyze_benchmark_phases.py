import os
import sys
import cv2
import csv
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine
from src.feature_matching.gfee import GeometricFeatureExtractionEngine
from src.registration.srae import SpatialRegistrationAlignmentEngine
from src.localization.localization import ClassicalLocalization

def main():
    results_path = "benchmark/results/benchmark_results.csv"
    if not os.path.exists(results_path):
        print("Results CSV not found.")
        return

    cases = []
    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)

    ice = ImageConditioningEngine()
    gspe = GlobalSearchProposalEngine(top_k=1)
    gfee = GeometricFeatureExtractionEngine()
    srae = SpatialRegistrationAlignmentEngine()

    group_a_stats = {'gspe_success': 0, 'gfee_success': 0, 'srae_success': 0, 'class_05': 0, 'c_errs': []}
    group_b_stats = {'c_errs': [], 'a_errs': [], 'f_errs': [], 'conf': [], 'ai_wins': 0, 'class_wins': 0, 'ai_refined': 0, 'fallback': 0}

    print("==================================================")
    print("PHASE ANALYSIS")
    print("==================================================")

    for case in cases:
        case_id = case['case_id']
        arch = case['architecture']
        diff = case['difficulty']
        gt = np.array([float(case['gt_x']), float(case['gt_y'])])
        
        c_err = float(case['classical_error'])
        a_err = float(case['ai_error'])
        f_err = float(case['final_error'])
        conf = float(case['ai_confidence'])
        ai_mag = float(case['ai_residual_magnitude'])
        decision = case['decision']

        # Run Phase 1 manually to get intermediate metrics
        case_dir = f"benchmark/cases/{case_id}"
        ref_img = cv2.imread(os.path.join(case_dir, "reference.png"), cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(os.path.join(case_dir, "search.png"), cv2.IMREAD_GRAYSCALE)
        
        cond = ice.run({'reference': ref_img, 'search': search_img})
        gspe_res = gspe.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
        
        if gspe_res['boxes']:
            x, y, w, h = gspe_res['boxes'][0]
            cand_crop = cond['search_cond'][y:y+h, x:x+w]
            gspe_coord = np.array([x + w/2.0, y + h/2.0])
            gspe_error = np.linalg.norm(gspe_coord - gt)
            
            gfee_res = gfee.run({'reference': cond['reference_cond'], 'candidate': cand_crop})
            kp_count = len(gfee_res.get('kp1', []))
            match_count = len(gfee_res.get('good_matches', []))
            
            srae_res = srae.run({
                'reference': cond['reference_cond'],
                'candidate': cand_crop,
                'kp1': gfee_res.get('kp1', []),
                'kp2': gfee_res.get('kp2', []),
                'matches': gfee_res.get('good_matches', [])
            })
            inliers = srae_res['stats'].get('inliers', 0)
        else:
            gspe_coord = np.array([0, 0])
            gspe_error = float('inf')
            kp_count = 0
            match_count = 0
            inliers = 0

        # Classification
        if c_err <= 5.0:
            category = "LOCAL_REFINEMENT_CASE"
        elif inliers < 3:
            category = "SRAE_FAILURE"
        elif gspe_error > 50.0:
            category = "PERIODIC_AMBIGUITY"
        else:
            category = "GLOBAL_SEARCH_FAILURE"

        print(f"{case_id} | {arch} | {diff}")
        print(f"GT: {gt} | GSPE: {gspe_coord} (Err: {gspe_error:.2f})")
        print(f"GFEE KP: {kp_count} | Matches: {match_count} | SRAE Inliers: {inliers}")
        print(f"Class Err: {c_err:.2f} | AI Err: {a_err:.2f} | Category: {category}\n")

        # Accumulate A
        group_a_stats['c_errs'].append(c_err)
        if gspe_error <= 50.0: group_a_stats['gspe_success'] += 1
        if match_count > 0: group_a_stats['gfee_success'] += 1
        if inliers >= 3: group_a_stats['srae_success'] += 1
        if c_err <= 0.5: group_a_stats['class_05'] += 1

        # Accumulate B
        if c_err <= 5.0:
            group_b_stats['c_errs'].append(c_err)
            group_b_stats['a_errs'].append(a_err)
            group_b_stats['f_errs'].append(f_err)
            group_b_stats['conf'].append(conf)
            if a_err < c_err: group_b_stats['ai_wins'] += 1
            else: group_b_stats['class_wins'] += 1
            if decision == 'AI_REFINED': group_b_stats['ai_refined'] += 1
            else: group_b_stats['fallback'] += 1

    print("==================================================")
    print("GROUP A — GLOBAL LOCALIZATION")
    print("==================================================")
    print(f"Number of cases: {len(cases)}")
    print(f"GSPE success rate: {group_a_stats['gspe_success']/len(cases)*100:.1f}%")
    print(f"GFEE success rate: {group_a_stats['gfee_success']/len(cases)*100:.1f}%")
    print(f"SRAE success rate: {group_a_stats['srae_success']/len(cases)*100:.1f}%")
    print(f"Classical accuracy @ 0.5 px: {group_a_stats['class_05']/len(cases)*100:.1f}%")
    print(f"Classical mean error: {np.mean(group_a_stats['c_errs']):.2f} px")
    print(f"Classical median error: {np.median(group_a_stats['c_errs']):.2f} px")

    print("\n==================================================")
    print("GROUP B — LOCAL REFINEMENT")
    print("==================================================")
    num_b = len(group_b_stats['c_errs'])
    print(f"Number of cases: {num_b}")
    if num_b > 0:
        c_errs = np.array(group_b_stats['c_errs'])
        a_errs = np.array(group_b_stats['a_errs'])
        f_errs = np.array(group_b_stats['f_errs'])
        
        print(f"Classical mean error: {np.mean(c_errs):.2f} px")
        print(f"AI mean error: {np.mean(a_errs):.2f} px")
        print(f"Final mean error: {np.mean(f_errs):.2f} px")
        print(f"Classical median: {np.median(c_errs):.2f} px")
        print(f"AI median: {np.median(a_errs):.2f} px")
        print(f"Final median: {np.median(f_errs):.2f} px")
        
        print(f"AI wins: {group_b_stats['ai_wins']}")
        print(f"Classical wins: {group_b_stats['class_wins']}")
        
        ai_imps = ((c_errs - a_errs) / c_errs) * 100
        f_imps = ((c_errs - f_errs) / c_errs) * 100
        print(f"Mean AI improvement: {np.mean(ai_imps):.2f}%")
        print(f"Mean final improvement: {np.mean(f_imps):.2f}%")
        
        print(f"AI_REFINED: {group_b_stats['ai_refined']}")
        print(f"CLASSICAL_FALLBACK: {group_b_stats['fallback']}")
        
        print(f"Confidence Mean: {np.mean(group_b_stats['conf']):.3f}")
        print(f"Confidence Median: {np.median(group_b_stats['conf']):.3f}")

if __name__ == "__main__":
    main()
