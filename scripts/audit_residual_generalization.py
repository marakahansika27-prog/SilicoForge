import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ai_refinement.network import SNRN
from src.ai_refinement.dataset import Phase1OutputSimDataset

def main():
    print("========================================")
    print("RESIDUAL GENERALIZATION AUDIT")
    print("========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_size = 1000
    batch_size = 16
    
    print(f"Loading Phase 2 Dataset (N={dataset_size})...")
    # Exact same settings as train.py
    dataset = Phase1OutputSimDataset(num_samples=dataset_size, apply_aug=True)
    
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    _, _, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = SNRN().to(device)
    model_path = "outputs/checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint {model_path} not found.")
        sys.exit(1)
        
    print(f"Loading checkpoint {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    target_dx_list = []
    target_dy_list = []
    pred_dx_list = []
    pred_dy_list = []
    
    print(f"Evaluating TEST split (N={len(test_ds)})...")
    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
                
            preds = model(batch['reference_patch'], batch['candidate_patch'])
            
            t_res = batch['target_delta'].cpu().numpy()
            p_res = preds['residual'].cpu().numpy()
            
            target_dx_list.extend(t_res[:, 0])
            target_dy_list.extend(t_res[:, 1])
            pred_dx_list.extend(p_res[:, 0])
            pred_dy_list.extend(p_res[:, 1])
            
    target_dx = np.array(target_dx_list)
    target_dy = np.array(target_dy_list)
    pred_dx = np.array(pred_dx_list)
    pred_dy = np.array(pred_dy_list)
    
    # Calculate metrics
    t_dx_mean, t_dx_std = np.mean(target_dx), np.std(target_dx)
    t_dy_mean, t_dy_std = np.mean(target_dy), np.std(target_dy)
    p_dx_mean, p_dx_std = np.mean(pred_dx), np.std(pred_dx)
    p_dy_mean, p_dy_std = np.mean(pred_dy), np.std(pred_dy)
    
    error_x = target_dx - pred_dx
    error_y = target_dy - pred_dy
    mae_x = np.mean(np.abs(error_x))
    mae_y = np.mean(np.abs(error_y))
    
    errors = np.sqrt(error_x**2 + error_y**2)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    # Handle Pearsonr and R2 carefully to avoid division by zero or warnings if variance is 0
    def safe_pearsonr(t, p):
        if np.std(p) < 1e-6 or np.std(t) < 1e-6:
            return 0.0
        return pearsonr(t, p)[0]

    r_dx = safe_pearsonr(target_dx, pred_dx)
    r_dy = safe_pearsonr(target_dy, pred_dy)
    
    def safe_r2(t, p):
        if np.std(t) < 1e-6:
            return 0.0
        return r2_score(t, p)
        
    r2_dx = safe_r2(target_dx, pred_dx)
    r2_dy = safe_r2(target_dy, pred_dy)
    
    t_mag = np.sqrt(target_dx**2 + target_dy**2)
    p_mag = np.sqrt(pred_dx**2 + pred_dy**2)
    
    t_mag_mean, t_mag_std = np.mean(t_mag), np.std(t_mag)
    p_mag_mean, p_mag_std = np.mean(p_mag), np.std(p_mag)
    
    print("\n========================================")
    print("METRICS")
    print("========================================")
    print(f"Target dx: Mean = {t_dx_mean:.4f}, Std = {t_dx_std:.4f}")
    print(f"Target dy: Mean = {t_dy_mean:.4f}, Std = {t_dy_std:.4f}")
    print(f"Pred dx  : Mean = {p_dx_mean:.4f}, Std = {p_dx_std:.4f}")
    print(f"Pred dy  : Mean = {p_dy_mean:.4f}, Std = {p_dy_std:.4f}")
    print(f"\nTarget Mag: Mean = {t_mag_mean:.4f}, Std = {t_mag_std:.4f}")
    print(f"Pred Mag  : Mean = {p_mag_mean:.4f}, Std = {p_mag_std:.4f}")
    print(f"\nMAE dx: {mae_x:.4f}")
    print(f"MAE dy: {mae_y:.4f}")
    print(f"Residual Vector MAE : {mae:.4f}")
    print(f"Residual Vector RMSE: {rmse:.4f}")
    print(f"\nPearson r (dx): {r_dx:.4f}")
    print(f"Pearson r (dy): {r_dy:.4f}")
    print(f"R² (dx): {r2_dx:.4f}")
    print(f"R² (dy): {r2_dy:.4f}")
    
    # Plotting
    os.makedirs("outputs/debug", exist_ok=True)
    
    def plot_scatter(t, p, axis_name, filename):
        plt.figure(figsize=(6, 6))
        plt.scatter(t, p, alpha=0.5, color='blue')
        
        min_val = min(np.min(t), np.min(p))
        max_val = max(np.max(t), np.max(p))
        # Draw y=x line
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x (Perfect Prediction)')
        
        plt.xlabel(f"Ground Truth {axis_name}")
        plt.ylabel(f"Predicted {axis_name}")
        plt.title(f"{axis_name} Correlation")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("outputs/debug", filename))
        plt.close()
        
    plot_scatter(target_dx, pred_dx, "dx", "residual_dx_correlation.png")
    plot_scatter(target_dy, pred_dy, "dy", "residual_dy_correlation.png")
    
    plt.figure(figsize=(10, 5))
    plt.hist(target_dx, bins=30, alpha=0.5, label='Target dx', color='blue')
    plt.hist(pred_dx, bins=30, alpha=0.5, label='Predicted dx', color='cyan')
    plt.hist(target_dy, bins=30, alpha=0.5, label='Target dy', color='red')
    plt.hist(pred_dy, bins=30, alpha=0.5, label='Predicted dy', color='orange')
    plt.title("Target and Predicted Distribution")
    plt.legend()
    plt.savefig(os.path.join("outputs/debug", "residual_prediction_distribution.png"))
    plt.close()

    # Classification
    print("\n========================================")
    print("CLASSIFICATION")
    print("========================================")
    
    if p_dx_std < 0.1 and p_dy_std < 0.1:
        print("CASE A: Prediction std is near zero.")
        print("=> Residual collapse.")
    elif abs(r_dx) < 0.2 and abs(r_dy) < 0.2:
        print("CASE B: Prediction std is substantial but correlation is near zero.")
        print("=> Model produces variation but has not learned the target relationship.")
    elif abs(r_dx) > 0.4 and abs(r_dy) > 0.4 and mae > 1.0:
        print("CASE C: Prediction correlation is substantial but MAE remains high.")
        print("=> Model is learning the correct direction but calibration/scale is poor.")
    elif abs(r_dx) > 0.7 and abs(r_dy) > 0.7 and mae < 1.0:
        print("CASE D: Correlation and R² are good and MAE is low.")
        print("=> Residual branch is actually learning; investigate why production/end-to-end error remains high.")
    else:
        print("Classification unclear or borderline. Review metrics manually.")
        
if __name__ == "__main__":
    main()
