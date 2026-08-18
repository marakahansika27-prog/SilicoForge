import os
import sys
import time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.network import SNRN
from src.utils.report_generator import ReportGenerator
from scripts.verify_utils import print_header, print_footer, assert_tensor_valid, print_feature_stats

def main():
    start_time = time.time()
    name = "FORWARD"
    print_header(name)
    report = ReportGenerator(f"VERIFY_{name}", save_dir="outputs/reports")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SNRN().to(device)
        model.eval()
        
        ref = torch.randn(1, 1, 128, 128).to(device)
        cand = torch.randn(1, 1, 128, 128).to(device)
        
        with torch.no_grad():
            preds = model(ref, cand)
        
        res = preds['residual']
        hm_logits = preds['heatmap_logits']
        hm = preds['heatmap']
        conf = preds['confidence']
        attn = preds['attn_map']
        ref_feat = preds['ref_feat']
        cand_feat = preds['cand_feat']
        dec_feat = preds['dec_feat']
        
        print("Output Shapes")
        print(f"  Residual Shape         : {list(res.shape)}")
        print(f"  Heatmap Logits Shape   : {list(hm_logits.shape)}")
        print(f"  Heatmap Shape          : {list(hm.shape)}")
        print(f"  Confidence Shape       : {list(conf.shape)}")
        print(f"  Attention Shape        : {list(attn.shape)}")
        print(f"  Reference Feature Shape: {list(ref_feat.shape)}")
        print(f"  Candidate Feature Shape: {list(cand_feat.shape)}")
        print(f"  Decoder Feature Shape  : {list(dec_feat.shape)}")
        print("")
        
        # FIX: Enhanced Validations
        assert_tensor_valid(res, "Residual")
        assert_tensor_valid(hm_logits, "Heatmap Logits")
        assert_tensor_valid(hm, "Heatmap")
        assert_tensor_valid(conf, "Confidence")
        assert_tensor_valid(attn, "Attention")
        assert_tensor_valid(ref_feat, "Reference Feature")
        
        assert not torch.isnan(hm_logits).any() and not torch.isinf(hm_logits).any(), "NaNs/Infs in heatmap logits"
        assert hm.sum().item() > 0.99 and hm.sum().item() < 1.01, f"Heatmap sum {hm.sum().item():.4f} != 1"
        assert conf.item() >= 0.0 and conf.item() <= 1.0, f"Confidence {conf.item():.4f} not in [0,1]"
        
        print("Statistics")
        print_feature_stats("Reference Feature", ref_feat)
        print_feature_stats("Candidate Feature", cand_feat)
        print_feature_stats("Decoder Feature", dec_feat)
        
        # Attention Diagnostics
        print("Attention Diagnostics")
        print(f"  Attention Max      : {attn.max().item():.4f}")
        print(f"  Average Attention  : {attn.mean().item():.4f}")
        
        eps = 1e-8
        attn_flat = attn.view(-1)
        attn_entropy = -torch.sum(attn_flat * torch.log(attn_flat + eps)).item()
        print(f"  Attention Entropy  : {attn_entropy:.4f}")
        
        # Sparsity: fraction of elements close to 0
        sparsity = (attn < 1e-3).float().mean().item()
        print(f"  Attention Sparsity : {sparsity:.4f}")
        
        report.add_status(True, "Forward pass mathematically verified successfully.")
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
