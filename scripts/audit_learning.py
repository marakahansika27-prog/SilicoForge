import os
from pyexpat import model
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.network import SNRN
from src.ai_refinement.dataset import Phase1OutputSimDataset

def main():
    print("========================================")
    print("AI LEARNING AUDIT")
    print("========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load untrained model exactly as pipeline.py does
    model = SNRN().to(device)
    model.eval()
    print("="*60)

    for name, param in model.residual_head.named_parameters():
        print(name)
        print(param.data.mean())
        print(param.data.std())
        print()

    print("="*60)
    
    # 2. Generate 100 samples from the dataset to measure distributions
    ds = Phase1OutputSimDataset(num_samples=100, apply_aug=False)
    dl = DataLoader(ds, batch_size=100)
    batch = next(iter(dl))
    
    ref = batch['reference_patch'].to(device)
    cand = batch['candidate_patch'].to(device)
    target_delta = batch['target_delta'].numpy()
    
    with torch.no_grad():
        preds = model(ref, cand)
        print("="*60)
        print("Ground Truth Residual:")
        print(batch["target_delta"][:5])

        print("\nPredicted Residual:")
        print(preds["residual"][:5])

        print("="*60)
    pred_delta = preds['residual'].cpu().numpy()
    
    # 3. Compute Metrics
    target_mag = np.linalg.norm(target_delta, axis=1)
    pred_mag = np.linalg.norm(pred_delta, axis=1)
    
    mae_dx = np.mean(np.abs(target_delta[:, 0] - pred_delta[:, 0]))
    mae_dy = np.mean(np.abs(target_delta[:, 1] - pred_delta[:, 1]))
    
    bias_dx = np.mean(pred_delta[:, 0])
    bias_dy = np.mean(pred_delta[:, 1])
    
    print(f"Target Magnitude Mean     : {np.mean(target_mag):.4f} px")
    print(f"Target Magnitude Std      : {np.std(target_mag):.4f} px")
    print(f"Predicted Magnitude Mean  : {np.mean(pred_mag):.4f} px")
    print(f"Predicted Magnitude Std   : {np.std(pred_mag):.4f} px")
    print("")
    print(f"MAE (dx)                  : {mae_dx:.4f} px")
    print(f"MAE (dy)                  : {mae_dy:.4f} px")
    print("")
    print(f"Prediction Bias (dx)      : {bias_dx:.4f} px")
    print(f"Prediction Bias (dy)      : {bias_dy:.4f} px")
    
    # 4. Plot
    os.makedirs("outputs/debug", exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.scatter(target_delta[:, 0], target_delta[:, 1], alpha=0.5, label="Ground Truth (Uniform [-3,3])")
    plt.scatter(pred_delta[:, 0], pred_delta[:, 1], alpha=0.8, color='red', label="Predictions (Untrained)")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title("Predicted vs Ground Truth Residuals (Untrained Network)")
    plt.xlabel("Delta X (pixels)")
    plt.ylabel("Delta Y (pixels)")
    plt.legend()
    plt.savefig("outputs/debug/residual_distribution.png")
    plt.close()
    
    print("\nGenerated outputs/debug/residual_distribution.png")
    print("========================================")

if __name__ == "__main__":
    main()
