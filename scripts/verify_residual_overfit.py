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
    print("RESIDUAL OVERFIT BATCHNORM DIAGNOSTIC")
    print("========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset (8 deterministic samples, no aug)
    ds = Phase1OutputSimDataset(num_samples=8, apply_aug=False)
    
    batch = {
        'reference_patch': [],
        'candidate_patch': [],
        'target_delta': [],
        'target_heatmap': [],
        'confidence_label': []
    }
    
    print("Loading 8 samples...")
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
    
    # 2. Fresh Model
    model = SNRN().to(device)
    criterion = SNRNTotalLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Register forward hook to capture dec_feat right before residual_head
    feature_tensors = {}
    def hook_fn(module, input, output):
        feature_tensors['dec_feat'] = input[0].detach()
        
    model.residual_head.register_forward_hook(hook_fn)
    
    epochs = 500
    print("Starting memorization test (500 epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        preds = model(batch['reference_patch'], batch['candidate_patch'])
        loss, loss_res, loss_hm, loss_conf = criterion(preds, batch)
        
        loss.backward()
        optimizer.step()
        
        if epoch == epochs:
            print(f"Final Training Epoch {epoch} Residual Loss: {loss_res:.4f}")
            
    print("========================================")
    print("CONTROLLED DIAGNOSTIC TESTS")
    print("========================================")
    
    target_res = batch['target_delta'].cpu().numpy()
    
    def print_metrics(test_name, preds, feat):
        pred_res = preds['residual'].cpu().numpy()
        error = np.linalg.norm(target_res - pred_res, axis=1)
        mae = np.mean(error)
        rmse = np.sqrt(np.mean(error**2))
        std_dx = np.std(pred_res[:, 0])
        std_dy = np.std(pred_res[:, 1])
        
        print(f"\n--- {test_name} ---")
        print("Prediction tensor:\n", pred_res)
        print(f"Residual MAE : {mae:.4f} px")
        print(f"Residual RMSE: {rmse:.4f} px")
        print(f"Prediction Std (dx, dy): ({std_dx:.4f}, {std_dy:.4f})")
        return pred_res, feat.clone(), mae

    # TEST A
    model.train()
    with torch.no_grad():
        preds_a = model(batch['reference_patch'], batch['candidate_patch'])
        pred_a, feat_a, mae_a = print_metrics("TEST A: model.train()", preds_a, feature_tensors['dec_feat'])
        
    # TEST B
    model.eval()
    with torch.no_grad():
        preds_b = model(batch['reference_patch'], batch['candidate_patch'])
        pred_b, feat_b, mae_b = print_metrics("TEST B: model.eval()", preds_b, feature_tensors['dec_feat'])
        
    # TEST C
    model.eval()
    def force_bn_train(m):
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
    model.apply(force_bn_train)
    
    with torch.no_grad():
        preds_c = model(batch['reference_patch'], batch['candidate_patch'])
        pred_c, feat_c, mae_c = print_metrics("TEST C: model.eval() + BatchNorm2d(training=True)", preds_c, feature_tensors['dec_feat'])
        
    print("\n========================================")
    print("BATCHNORM2D STATISTICS")
    print("========================================")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            print(f"Layer: {name}")
            print(f"  running_mean max: {module.running_mean.abs().max().item():.4f}")
            print(f"  running_var max: {module.running_var.max().item():.4f}")
            print(f"  weight gamma max: {module.weight.abs().max().item():.4f}")
            print(f"  bias beta max: {module.bias.abs().max().item():.4f}")
            print(f"  num_batches_tracked: {module.num_batches_tracked.item()}")
            
    print("\n========================================")
    print("DIFFERENCE ANALYSIS")
    print("========================================")
    
    diff_pred_a_b = np.abs(pred_a - pred_b).max()
    diff_feat_a_b = (feat_a - feat_b).abs().max().item()
    diff_pred_b_c = np.abs(pred_b - pred_c).max()
    diff_pred_a_c = np.abs(pred_a - pred_c).max()
    
    print(f"Max Diff (TRAIN vs EVAL predictions): {diff_pred_a_b:.6f}")
    print(f"Max Diff (TRAIN vs EVAL dec_feat): {diff_feat_a_b:.6f}")
    print(f"Max Diff (EVAL vs EVAL-batch-stats predictions): {diff_pred_b_c:.6f}")
    print(f"Max Diff (TRAIN vs EVAL-batch-stats predictions): {diff_pred_a_c:.6f}")
    
    print("\n========================================")
    if diff_pred_a_c < 0.1 and diff_pred_a_b > 1.0:
        print("[PASS] BatchNorm running-statistics mismatch confirmed.")
    else:
        print("[FAIL] BatchNorm is not sufficient to explain the train/eval discrepancy.")

if __name__ == "__main__":
    main()
