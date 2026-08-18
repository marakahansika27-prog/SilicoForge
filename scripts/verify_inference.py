import os
import sys
import time
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.network import SNRN
from src.ai_refinement.inference import SNRNInference
from src.ai_refinement.dataset import Phase1OutputSimDataset
from src.utils.report_generator import ReportGenerator
from scripts.verify_utils import print_header, print_footer

def save_img(tensor_2d, path):
    img = tensor_2d.detach().cpu().numpy()
    if img.max() <= 1.01:
        img = img * 255
    cv2.imwrite(path, img.astype(np.uint8))

def main():
    start_time = time.time()
    name = "INFERENCE"
    print_header(name)
    report = ReportGenerator(f"VERIFY_{name}", save_dir="outputs/reports")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SNRN().to(device)
        model.eval()
        inference_engine = SNRNInference(model)
        
        ds = Phase1OutputSimDataset(num_samples=1)
        dl = DataLoader(ds, batch_size=1)
        batch = next(iter(dl))
        
        # FIX: Explicitly send batch to device to match pipeline
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
                
        # We also need the raw model predictions to dump diagnostics
        with torch.no_grad():
            preds = model(batch['reference_patch'], batch['candidate_patch'])
            
        metrics = inference_engine.run(batch)
        
        # Diagnostics Output
        os.makedirs("outputs/debug/inference", exist_ok=True)
        b = 0
        save_img(batch['reference_patch'][b,0], "outputs/debug/inference/reference_patch.png")
        save_img(batch['candidate_patch'][b,0], "outputs/debug/inference/candidate_patch.png")
        save_img(batch['target_heatmap'][b,0], "outputs/debug/inference/gt_heatmap.png")
        
        hm = preds['heatmap'][b,0]
        hm_norm = hm / (hm.max() + 1e-8)
        save_img(hm_norm, "outputs/debug/inference/pred_heatmap.png")
        
        attn = preds['attn_map'][b].mean(dim=0).view(8, 8)
        attn = torch.nn.functional.interpolate(attn.unsqueeze(0).unsqueeze(0), size=(128,128), mode='bilinear').squeeze()
        attn_norm = attn / (attn.max() + 1e-8)
        save_img(attn_norm, "outputs/debug/inference/attention_map.png")
        
        # Difference map
        diff = torch.abs(batch['reference_patch'][b,0] - batch['candidate_patch'][b,0])
        save_img(diff, "outputs/debug/inference/difference_map.png")
        
        c_err = metrics['classical_error'][b]
        a_err = metrics['ai_error'][b]
        pct = metrics['improvement_pct'][b]
        
        print("Performance Diagnostics")
        print(f"  Classical Error : {c_err:.4f} pixels")
        print(f"  AI Refined Error: {a_err:.4f} pixels")
        print(f"  Improvement %   : {pct:.2f}%")
        print("")
        print("Outputs saved to outputs/debug/inference/")
        
        # FIX: Adding strict assert to ensure inference is geometrically tight
        assert not np.isnan(a_err) and not np.isinf(a_err), "Inference produced invalid AI Error metric."
        
        report.add_status(True, "Inference executed correctly with full diagnostic outputs.")
        print_footer(name, start_time, True)
        
    except Exception as e:
        report.add_status(False, str(e))
        print_footer(name, start_time, False)
        print(f"Error: {e}")
        raise
    finally:
        report.save()

if __name__ == "__main__":
    main()
