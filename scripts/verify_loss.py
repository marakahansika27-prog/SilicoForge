import os
import sys
import time
import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.network import SNRN
from src.ai_refinement.loss import SNRNTotalLoss
from src.utils.report_generator import ReportGenerator
from scripts.verify_utils import print_header, print_footer, assert_tensor_valid

def main():
    start_time = time.time()
    name = "LOSS"
    print_header(name)
    report = ReportGenerator(f"VERIFY_{name}", save_dir="outputs/reports")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SNRN().to(device)
        criterion = SNRNTotalLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Dummy batch
        heatmap_logits = torch.zeros(
            1, 1, 128, 128,
            device=device,
            requires_grad=True
        )
        
        heatmap = torch.softmax(
            heatmap_logits.view(1, -1),
            dim=1
        ).view(1, 1, 128, 128)
        
        preds = {
            'residual': torch.tensor([[1.0, -1.0]], device=device, requires_grad=True),
            'heatmap_logits': heatmap_logits,
            'heatmap': heatmap,
            'confidence': torch.tensor([[0.8]], device=device, requires_grad=True)
        }
        
        targets = {
            'target_delta': torch.tensor([[0.5, -0.5]], device=device),
            'target_heatmap': (torch.ones(1, 1, 128, 128, device=device) / (128*128)),
            'confidence_label': torch.tensor([[1.0]], device=device)
        }
        
        loss, l_res, l_hm, l_conf = criterion(preds, targets)
        loss.backward()
        
        print("Statistics")
        print(f"  Residual Loss   : {l_res:.6f}")
        print(f"  KL Loss         : {l_hm:.6f}")
        print(f"  Confidence BCE  : {l_conf:.6f}")
        print(f"  Total Loss      : {loss.item():.6f}")
        
        assert_tensor_valid(loss, "Total Loss")
        assert loss.item() > 0, "Loss should be > 0 for this dummy target"
        
        assert preds['residual'].grad is not None, "Gradients missing for residual"
        assert preds['heatmap_logits'].grad is not None, "Gradients missing for heatmap_logits"
        assert preds['confidence'].grad is not None, "Gradients missing for confidence"
        
        report.add_status(True, "Loss verification passed.")
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
