import os
import sys
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.network import SNRN
from src.ai_refinement.loss import SNRNTotalLoss
from src.utils.report_generator import ReportGenerator
from scripts.verify_utils import print_header, print_footer, assert_tensor_valid

def main():
    start_time = time.time()
    name = "GRADIENT_FLOW"
    print_header(name)
    report = ReportGenerator(f"VERIFY_{name}", save_dir="outputs/reports")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SNRN().to(device)
        criterion = SNRNTotalLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        ref = torch.randn(2, 1, 128, 128).to(device)
        cand = torch.randn(2, 1, 128, 128).to(device)
        
        targets = {
            'target_delta': torch.randn(2, 2).to(device),
            'target_heatmap': (torch.ones(2, 1, 128, 128) / (128*128)).to(device),
            'confidence_label': torch.ones(2, 1).to(device)
        }
        
        optimizer.zero_grad()
        preds = model(ref, cand)
        loss, _, _, _ = criterion(preds, targets)
        loss.backward()
        
        print("Layer Gradients")
        print(f"{'Layer Name':<40} | {'Norm':<10} | {'Max':<10} | {'Min':<10} | {'Mean':<10}")
        print("-" * 90)
        
        all_grads = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Gradient is missing for {n}"
                grad = p.grad
                
                # Check NaNs/Infs
                assert_tensor_valid(grad, f"Gradient for {n}")
                
                norm = grad.norm().item()
                max_g = grad.max().item()
                min_g = grad.min().item()
                mean_g = grad.mean().item()
                
                assert norm > 0, f"Gradient norm is zero for {n}. Layer is not learning!"
                
                all_grads.extend(grad.view(-1).cpu().numpy())
                print(f"{n:<40} | {norm:<10.4f} | {max_g:<10.4f} | {min_g:<10.4f} | {mean_g:<10.4f}")
        
        os.makedirs("outputs/debug", exist_ok=True)
        plt.figure()
        plt.hist(all_grads, bins=100, log=True)
        plt.title("Gradient Histogram")
        plt.savefig("outputs/debug/gradient_histogram.png")
        plt.close()
        
        report.add_status(True, "Gradient flow verified successfully across all layers.")
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
