import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ai_refinement.network import SNRN
from src.ai_refinement.loss import SNRNTotalLoss
from src.ai_refinement.dataset import Phase1OutputSimDataset

def main():
    print("========================================")
    print("NORMALIZATION ARCHITECTURE VERIFICATION")
    print("========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNRN().to(device)
    
    # 1 & 2. Verify layers
    bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    gn_count = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    
    print(f"BatchNorm layers remaining: {bn_count}")
    print(f"GroupNorm layers: {gn_count}")
    
    if bn_count != 0:
        print("[FAIL] BatchNorm2d was not completely removed.")
        sys.exit(1)
    if gn_count == 0:
        print("[FAIL] No GroupNorm layers found.")
        sys.exit(1)
        
    # 3. Verify Output Shapes
    print("\nVerifying output shapes...")
    dummy_ref = torch.randn(1, 1, 128, 128).to(device)
    dummy_cand = torch.randn(1, 1, 128, 128).to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(dummy_ref, dummy_cand)
        
    shapes_pass = True
    if out['residual'].shape != (1, 2):
        print(f"Residual shape mismatch: {out['residual'].shape}")
        shapes_pass = False
    if out['heatmap'].shape != (1, 1, 128, 128):
        print(f"Heatmap shape mismatch: {out['heatmap'].shape}")
        shapes_pass = False
    if out['confidence'].shape != (1, 1):
        print(f"Confidence shape mismatch: {out['confidence'].shape}")
        shapes_pass = False
        
    if not shapes_pass:
        print("Output shape verification: FAIL")
        sys.exit(1)
    print("Output shape verification: PASS")
    
    # 4. Run Memorization Experiment
    print("\nStarting 8-sample Memorization Experiment (300 Epochs)...")
    ds = Phase1OutputSimDataset(num_samples=8, apply_aug=False)
    batch = {
        'reference_patch': [],
        'candidate_patch': [],
        'target_delta': [],
        'target_heatmap': [],
        'confidence_label': []
    }
    for i in range(8):
        sample = ds[i]
        batch['reference_patch'].append(sample['reference_patch'])
        batch['candidate_patch'].append(sample['candidate_patch'])
        batch['target_delta'].append(sample['target_delta'])
        batch['target_heatmap'].append(sample['target_heatmap'])
        batch['confidence_label'].append(sample['confidence_label'])
        
    for k in batch.keys():
        batch[k] = torch.stack(batch[k]).squeeze(1) if len(batch[k][0].shape) == 4 else torch.stack(batch[k])
        batch[k] = batch[k].to(device)
        
    criterion = SNRNTotalLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 300
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        preds = model(batch['reference_patch'], batch['candidate_patch'])
        loss, loss_res, loss_hm, loss_conf = criterion(preds, batch)
        
        loss.backward()
        optimizer.step()

    # 5. Compare TEST A vs TEST B
    target_res = batch['target_delta'].cpu().numpy()
    
    # TEST A: Train Mode
    model.train()
    with torch.no_grad():
        preds_train = model(batch['reference_patch'], batch['candidate_patch'])
        pred_train = preds_train['residual'].cpu().numpy()
        train_mae = np.mean(np.linalg.norm(target_res - pred_train, axis=1))
        
    # TEST B: Eval Mode
    model.eval()
    with torch.no_grad():
        preds_eval = model(batch['reference_patch'], batch['candidate_patch'])
        pred_eval = preds_eval['residual'].cpu().numpy()
        eval_mae = np.mean(np.linalg.norm(target_res - pred_eval, axis=1))
        
    max_diff = np.abs(pred_train - pred_eval).max()
    
    print("\n========================================")
    print("FINAL REPORT")
    print("========================================")
    print("Normalization before: BatchNorm2d")
    print("Normalization after: GroupNorm")
    print(f"BatchNorm layers remaining: {bn_count}")
    print(f"GroupNorm layers: {gn_count}")
    print("Output shape verification: PASS")
    
    if max_diff < 1e-4:
        print("Train/eval consistency: PASS")
    else:
        print("Train/eval consistency: FAIL")
        
    print(f"Train MAE: {train_mae:.6f}")
    print(f"Eval MAE: {eval_mae:.6f}")
    print(f"Max train/eval prediction difference: {max_diff:.8f}")
    
    print("\n========================================")
    print("OLD CHECKPOINTS INVALID FOR NEW NORMALIZATION")
    print("========================================")

if __name__ == "__main__":
    main()
