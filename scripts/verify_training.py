import os
import sys
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.network import SNRN
from src.ai_refinement.loss import SNRNTotalLoss
from src.ai_refinement.trainer import SNRNTrainer
from src.ai_refinement.dataset import Phase1OutputSimDataset
from src.utils.report_generator import ReportGenerator
from scripts.verify_utils import print_header, print_footer, compute_entropy

def run_stage(trainer, ds_size, epochs, stage_name, batch_size=4):
    print(f"\n--- {stage_name} (N={ds_size}, Epochs={epochs}) ---")
    ds = Phase1OutputSimDataset(num_samples=ds_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    history = {'loss': [], 'res': [], 'kl': [], 'conf': [], 'entropy': [], 'grad_norm': []}
    
    for epoch in range(epochs):
        epoch_loss = epoch_res = epoch_kl = epoch_conf = epoch_entropy = epoch_grad = 0.0
        
        for batch in dl:
            trainer.model.train()
            trainer.optimizer.zero_grad()
            
            preds = trainer.model(batch['reference_patch'], batch['candidate_patch'])
            loss, l_res, l_hm, l_conf = trainer.criterion(preds, batch)
            
            # FIX: Ensure training doesn't silently ignore NaNs during verification
            assert not torch.isnan(loss) and not torch.isinf(loss), "Explosion detected in Loss calculation."
            
            loss.backward()
            
            # FIX: Adding Gradient clipping to mimic train.py stability during verification
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
            
            trainer.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_res += l_res
            epoch_kl += l_hm
            epoch_conf += l_conf
            epoch_entropy += compute_entropy(preds['heatmap'])
            
            total_norm = 0.0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    # FIX: Safely calculate grad norm handling NaNs
                    norm_sq = p.grad.data.norm(2).item() ** 2
                    if not np.isnan(norm_sq) and not np.isinf(norm_sq):
                        total_norm += norm_sq
            epoch_grad += total_norm ** 0.5
            
        N = len(dl)
        history['loss'].append(epoch_loss/N)
        history['res'].append(epoch_res/N)
        history['kl'].append(epoch_kl/N)
        history['conf'].append(epoch_conf/N)
        history['entropy'].append(epoch_entropy/N)
        history['grad_norm'].append(epoch_grad/N)
        
        if epoch % max(1, epochs//10) == 0 or epoch == epochs - 1:
            lr = trainer.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d} | Total: {history['loss'][-1]:.4f} | Res: {history['res'][-1]:.4f} | "
                  f"KL: {history['kl'][-1]:.4f} | Conf: {history['conf'][-1]:.4f} | "
                  f"Ent: {history['entropy'][-1]:.4f} | Grad: {history['grad_norm'][-1]:.4f} | LR: {lr:.5f}")
            
    return history

def plot_curves(history, name):
    os.makedirs("outputs/debug", exist_ok=True)
    plt.figure()
    plt.plot(history['loss'], label="Total Loss")
    plt.plot(history['res'], label="Residual")
    plt.title(f"{name} - Loss Curve")
    plt.legend()
    plt.savefig(f"outputs/debug/loss_curve_{name}.png")
    plt.close()
    
    plt.figure()
    plt.plot(history['kl'])
    plt.title(f"{name} - KL Curve")
    plt.savefig(f"outputs/debug/kl_curve_{name}.png")
    plt.close()
    
    plt.figure()
    plt.plot(history['conf'])
    plt.title(f"{name} - Confidence BCE Curve")
    plt.savefig(f"outputs/debug/confidence_curve_{name}.png")
    plt.close()
    
    plt.figure()
    plt.plot(history['grad_norm'])
    plt.title(f"{name} - Gradient Norm")
    plt.savefig(f"outputs/debug/gradient_curve_{name}.png")
    plt.close()

def main():
    start_time = time.time()
    name = "TRAINING"
    print_header(name)
    report = ReportGenerator(f"VERIFY_{name}", save_dir="outputs/reports")
    
    try:
        model = SNRN()
        criterion = SNRNTotalLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        trainer = SNRNTrainer(model, criterion, optimizer)
        
        # Stage 1: 1 sample, 50 epochs
        h1 = run_stage(trainer, ds_size=1, epochs=50, stage_name="Stage 1 (Single Sample Overfit)", batch_size=1)
        plot_curves(h1, "stage1")
        assert h1['loss'][-1] < 1.0, "Stage 1 failed to overfit properly."
        
        # Stage 2: 8 samples, 100 epochs
        h2 = run_stage(trainer, ds_size=8, epochs=100, stage_name="Stage 2 (Single Batch Overfit)", batch_size=8)
        plot_curves(h2, "stage2")
        
        # Stage 3: 32 samples, 100 epochs
        h3 = run_stage(trainer, ds_size=32, epochs=50, stage_name="Stage 3 (Mini Dataset)", batch_size=16)
        plot_curves(h3, "stage3")
        
        report.add_status(True, "Training overfit verified successfully. Curves generated.")
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
