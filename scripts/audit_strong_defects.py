import os
import sys
import cv2
import numpy as np
from prettytable import PrettyTable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.integration.pipeline import HybridNavigationPipeline

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def generate_strong_defect_audit(seed, architecture):
    rs = np.random.RandomState(seed)
    
    base_size = 10240
    base_img_clean = np.zeros((base_size, base_size), dtype=np.float32)
    base_img = np.zeros((base_size, base_size), dtype=np.float32)
    
    # Strong defect parameters (D3)
    prob_missing = 0.10
    prob_disp = 0.20
    disp_mag = 70
    
    if architecture == "DRAM":
        for i in range(150, base_size, 300):
            for j in range(150, base_size, 300):
                # Clean Scene
                cv2.circle(base_img_clean, (i, j), 60, 180, -1)
                cv2.rectangle(base_img_clean, (i-45, j-45), (i+45, j+45), 100, 6)
                
                # Defect Scene
                is_missing = (rs.rand() < prob_missing)
                if is_missing:
                    continue
                    
                is_disp = (rs.rand() < prob_disp)
                dx = rs.randint(-disp_mag, disp_mag) if is_disp else 0
                dy = rs.randint(-disp_mag, disp_mag) if is_disp else 0
                
                cv2.circle(base_img, (i+dx, j+dy), 60, 180, -1)
                cv2.rectangle(base_img, (i+dx-45, j+dy-45), (i+dx+45, j+dy+45), 100, 6)
                
    elif architecture == "FinFET":
        for i in range(150, base_size, 200):
            # Clean
            cv2.line(base_img_clean, (i, 0), (i, base_size), 180, 30)
            
            # Defect
            is_disp = (rs.rand() < prob_disp)
            dx = rs.randint(-disp_mag//2, disp_mag//2) if is_disp else 0
            cv2.line(base_img, (i+dx, 0), (i+dx, base_size), 180, 30)
            
        for j in range(150, base_size, 600):
            # Clean
            cv2.line(base_img_clean, (0, j), (base_size, j), 120, 15)
            
            # Defect
            is_missing = (rs.rand() < prob_missing)
            if is_missing:
                continue
            is_disp = (rs.rand() < prob_disp)
            dy = rs.randint(-disp_mag, disp_mag) if is_disp else 0
            cv2.line(base_img, (0, j+dy), (base_size, j+dy), 120, 15)
                
    # Shared Edge Brightening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    
    gradient_clean = cv2.morphologyEx(base_img_clean, cv2.MORPH_GRADIENT, kernel)
    sem_clean = np.clip(base_img_clean + gradient_clean * 1.5, 0, 255).astype(np.float32)
    
    gradient = cv2.morphologyEx(base_img, cv2.MORPH_GRADIENT, kernel)
    sem_img = np.clip(base_img + gradient * 1.5, 0, 255).astype(np.float32)
    
    noise_level = 50
    blur = 3
        
    valid = False
    attempts = 0
    while not valid and attempts < 100:
        attempts += 1
        offset_x = rs.randint(2000, 8000)
        offset_y = rs.randint(2000, 8000)
        ref_float = sem_img[offset_y:offset_y+900, offset_x:offset_x+900].copy()
        if np.std(ref_float) > 10.0:
            valid = True
            
    gt_x = (offset_x + 450.0) / 10.0
    gt_y = (offset_y + 450.0) / 10.0
    
    # 10x scale relationship
    search_float = cv2.resize(sem_img, (1024, 1024), interpolation=cv2.INTER_AREA)
    search_float = cv2.GaussianBlur(search_float, (blur, blur), 0)
        
    ref_noise = rs.poisson(ref_float / 255.0 * 20) / 20 * 255
    ref_img = np.clip(ref_float + ref_noise - 128, 0, 255).astype(np.uint8)
    
    search_noise = rs.poisson(search_float / 255.0 * noise_level) / noise_level * 255
    search_img = np.clip(search_float + search_noise - 128, 0, 255).astype(np.uint8)
    
    # Clean Search (for visual comparison only)
    search_clean = cv2.resize(sem_clean, (1024, 1024), interpolation=cv2.INTER_AREA)
    search_clean = np.clip(search_clean, 0, 255).astype(np.uint8)
    
    return ref_img, search_img, search_clean, gt_x, gt_y, sem_img, offset_x, offset_y

def main():
    print("========================================")
    print("STRONG DEFECT PLAUSIBILITY AUDIT")
    print("========================================")
    
    out_dir = "outputs/debug/strong_defect_plausibility"
    os.makedirs(out_dir, exist_ok=True)
    
    pipeline = HybridNavigationPipeline(top_k=5, nms_radius=50)
    
    cases = []
    for i in range(5): cases.append("DRAM")
    for i in range(5): cases.append("FinFET")
    
    for idx, arch in enumerate(cases):
        seed = 7000 + idx
        ref_img, search_img, search_clean, gt_x, gt_y, sem_img, ox, oy = generate_strong_defect_audit(seed, arch)
        
        # GSPE processing
        h_ref, w_ref = ref_img.shape
        new_h, new_w = h_ref // 10, w_ref // 10
        ref_scaled = cv2.resize(ref_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        top_left_x = int(round(gt_x - new_w/2.0))
        top_left_y = int(round(gt_y - new_h/2.0))
        gt_crop = search_img[top_left_y:top_left_y+new_h, top_left_x:top_left_x+new_w]
        
        # 3. VISUAL PLAUSIBILITY AUDIT
        case_dir = os.path.join(out_dir, f"{idx+1}_{arch}")
        os.makedirs(case_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(case_dir, "1_clean_periodic_scene.png"), search_clean)
        cv2.imwrite(os.path.join(case_dir, "2_strong_defect_scene.png"), search_img)
        cv2.imwrite(os.path.join(case_dir, "3_ref_900.png"), ref_img)
        cv2.imwrite(os.path.join(case_dir, "4_ref_scaled_90.png"), ref_scaled)
        cv2.imwrite(os.path.join(case_dir, "5_search_gt_crop.png"), gt_crop)
        
        # Quantitative
        res = cv2.matchTemplate(search_img, ref_scaled, cv2.TM_CCOEFF_NORMED)
        ncc_at_gt = res[top_left_y, top_left_x]
        
        gspe_res = pipeline.gspe.run({'reference': ref_img, 'search': search_img})
        boxes = gspe_res['boxes']
        scores = gspe_res['scores']
        
        gt_rank = -1
        distractor_idx = -1
        
        for r, (bx, by, bw, bh) in enumerate(boxes):
            cx = bx + bw/2.0
            cy = by + bh/2.0
            if calculate_distance(cx, cy, gt_x, gt_y) <= 5.0:
                gt_rank = r + 1
            else:
                if distractor_idx == -1:
                    distractor_idx = r
                    
        distractor_ncc = scores[distractor_idx] if distractor_idx != -1 else 0.0
        margin = ncc_at_gt - distractor_ncc
        
        print(f"[{idx+1:02d}] {arch:6s} | GT NCC: {ncc_at_gt:.4f} | Distractor: {distractor_ncc:.4f} | Margin: {margin:6.4f} | GT Rank: {gt_rank if gt_rank != -1 else '>5'}")

    print("\n--- DONE ---")
    print("Check outputs/debug/strong_defect_plausibility/ to visually assess if the defect density destroys the fundamental semiconductor lattice pattern, rendering it unrealistic.")

if __name__ == "__main__":
    main()
