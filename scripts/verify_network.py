import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.network import SNRN
from src.utils.report_generator import ReportGenerator

def main():
    report = ReportGenerator("VERIFY_NETWORK", save_dir="outputs/reports")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SNRN().to(device)
        
        # 1. Check modules exist
        assert hasattr(model, 'backbone'), "Missing module: backbone"
        assert hasattr(model, 'cross_attn'), "Missing module: cross_attention (cross_attn)"
        assert hasattr(model, 'residual_head'), "Missing module: residual_head"
        assert hasattr(model, 'heatmap_head'), "Missing module: heatmap_head"
        assert hasattr(model, 'confidence_head'), "Missing module: confidence_head"
        
        # 2. Parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 3. Dummy forward pass
        ref_patch = torch.randn(1, 1, 128, 128).to(device)
        cand_patch = torch.randn(1, 1, 128, 128).to(device)
        
        preds = model(ref_patch, cand_patch)
        
        res_shape = list(preds['residual'].shape)
        hm_shape = list(preds['heatmap'].shape)
        conf_shape = list(preds['confidence'].shape)
        
        assert res_shape == [1, 2], f"Expected Residual (1, 2), got {res_shape}"
        assert hm_shape == [1, 1, 128, 128], f"Expected Heatmap (1, 1, 128, 128), got {hm_shape}"
        assert conf_shape == [1, 1], f"Expected Confidence (1, 1), got {conf_shape}"
        
        # 4. Console output
        print("================================")
        print("VERIFY NETWORK")
        print("================================")
        print("Model name          : SNRN (Sub-pixel Navigation Refinement Network)")
        print(f"Device              : {device}")
        print(f"Total parameters    : {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("")
        print(f"Backbone type       : {type(model.backbone).__name__}")
        print(f"Cross Attention     : {type(model.cross_attn).__name__}")
        print(f"Residual Head       : {type(model.residual_head).__name__}")
        print(f"Heatmap Head        : {type(model.heatmap_head).__name__}")
        print(f"Confidence Head     : {type(model.confidence_head).__name__}")
        print("")
        print("Expected outputs:")
        print(f"Residual:           {tuple(res_shape)}")
        print(f"Heatmap:            {tuple(hm_shape)}")
        print(f"Confidence:         {tuple(conf_shape)}")
        print("")
        print("PASS")
        print("================================")
        
        obs = f"Model instantiated successfully.\nTotal Params: {total_params:,}\nTrainable Params: {trainable_params:,}\n" \
              f"Output Shapes -> Res: {res_shape}, HM: {hm_shape}, Conf: {conf_shape}"
        report.add_status(True, obs)
        
    except AssertionError as e:
        print("================================")
        print("FAIL")
        print(f"Error: {str(e)}")
        print("================================")
        report.add_status(False, str(e))
        raise
    except Exception as e:
        print("================================")
        print("FAIL")
        print(f"Exception: {str(e)}")
        print("================================")
        report.add_status(False, str(e))
        raise
    finally:
        report.save()

if __name__ == "__main__":
    main()
