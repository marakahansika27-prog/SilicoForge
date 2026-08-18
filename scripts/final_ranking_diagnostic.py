import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.integration.pipeline import HybridNavigationPipeline
from scripts.benchmark_40_cases import generate_benchmark_case

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def main():
    print("========================================")
    print("FINAL RANKING & PERIODICITY DIAGNOSTIC")
    print("========================================")
    
    # We use Top-K=10 and NMS=50 to capture enough candidates
    pipeline = HybridNavigationPipeline(top_k=10, nms_radius=50)
    
    cases_def = []
    for i in range(8): cases_def.append(("DRAM", "easy"))
    for i in range(8): cases_def.append(("DRAM", "moderate"))
    for i in range(6): cases_def.append(("DRAM", "hard"))
    for i in range(8): cases_def.append(("FinFET", "easy"))
    for i in range(6): cases_def.append(("FinFET", "moderate"))
    for i in range(4): cases_def.append(("FinFET", "hard"))
    
    ncc_wins = 0
    srae_wins = 0
    top3_ncc_hits = 0
    top5_ncc_hits = 0
    total_valid = 0
    
    print("\n| Case | Rank | X | Y | NCC Score | SRAE Inliers | Dist to GT | Selected |")
    print("|------|------|---|---|-----------|--------------|------------|----------|")
    
    for idx, (arch, diff) in enumerate(cases_def):
        seed = 1000 + idx + 1
        ref_img, search_img, gt_x, gt_y = generate_benchmark_case(seed, arch, diff)
        
        cond = pipeline.ice.run({'reference': ref_img, 'search': search_img})
        gspe_res = pipeline.gspe.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
        
        boxes = gspe_res['boxes']
        scores = gspe_res['scores']
        
        if not boxes:
            continue
            
        candidates = []
        best_inliers = -1
        srae_winner_idx = -1
        
        for rank_idx, ((x, y, w, h), score) in enumerate(zip(boxes, scores)):
            center_x = x + w/2.0
            center_y = y + h/2.0
            dist = calculate_distance(center_x, center_y, gt_x, gt_y)
            
            cand_crop = cond['search_cond'][y:y+h, x:x+w]
            gfee_res = pipeline.gfee.run({'reference': cond['reference_cond'], 'candidate': cand_crop})
            srae_res = pipeline.srae.run({
                'reference': cond['reference_cond'],
                'candidate': cand_crop,
                'kp1': gfee_res.get('kp1', []),
                'kp2': gfee_res.get('kp2', []),
                'matches': gfee_res.get('good_matches', [])
            })
            inliers = srae_res['stats'].get('inliers', 0)
            
            if inliers > best_inliers:
                best_inliers = inliers
                srae_winner_idx = rank_idx
                
            candidates.append({
                'idx': rank_idx,
                'x': center_x,
                'y': center_y,
                'ncc': score,
                'inliers': inliers,
                'dist': dist
            })
            
        # Determine the True Candidate (min distance)
        true_cand = min(candidates, key=lambda c: c['dist'])
        is_present = true_cand['dist'] <= 25.0
        
        if is_present:
            total_valid += 1
            # Sort by NCC
            ncc_sorted = sorted(candidates, key=lambda c: c['ncc'], reverse=True)
            ncc_rank = next(i for i, c in enumerate(ncc_sorted) if c['idx'] == true_cand['idx']) + 1
            
            # Sort by SRAE Inliers
            srae_sorted = sorted(candidates, key=lambda c: c['inliers'], reverse=True)
            srae_rank = next(i for i, c in enumerate(srae_sorted) if c['idx'] == true_cand['idx']) + 1
            
            if ncc_rank == 1: ncc_wins += 1
            if srae_rank == 1: srae_wins += 1
            if ncc_rank <= 3: top3_ncc_hits += 1
            if ncc_rank <= 5: top5_ncc_hits += 1
            
            # Detailed print for specific observed failure cases (simulated via indices 4, 11, 13, 15, 17)
            if idx in [4, 11, 13, 15, 17]:
                for c in candidates:
                    selected = "YES" if c['idx'] == srae_winner_idx else ""
                    print(f"| {idx+1:02d} | {c['idx']+1} | {c['x']:.0f} | {c['y']:.0f} | {c['ncc']:.4f} | {c['inliers']:3d} | {c['dist']:6.1f} | {selected:3s} |")
                
                # Periodicity
                if len(candidates) >= 2:
                    d01 = calculate_distance(candidates[0]['x'], candidates[0]['y'], candidates[1]['x'], candidates[1]['y'])
                    print(f"  -> PERIODICITY: Peak 1 and 2 separated by {d01:.1f} px. (Expected DRAM pitch: 100px)")
                    print(f"  -> True NCC vs Selected NCC difference: {true_cand['ncc'] - candidates[srae_winner_idx]['ncc']:.4f}")
                    print(f"  -> True SRAE vs Selected SRAE difference: {true_cand['inliers'] - candidates[srae_winner_idx]['inliers']}")
                    print("---------------------------------------------------------")
                    
    print("\n========================================")
    print("RANKING PERFORMANCE SUMMARY")
    print("========================================")
    print(f"Total Valid Cases (True Target in Top-10): {total_valid}/40")
    if total_valid > 0:
        print(f"NCC ranks True Candidate #1: {ncc_wins / total_valid * 100:.1f}%")
        print(f"SRAE ranks True Candidate #1: {srae_wins / total_valid * 100:.1f}%")
        print(f"Top-3 NCC Recall: {top3_ncc_hits / total_valid * 100:.1f}%")
        print(f"Top-5 NCC Recall: {top5_ncc_hits / total_valid * 100:.1f}%")

if __name__ == "__main__":
    main()
