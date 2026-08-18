import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_refinement.network import SNRN
from src.ai_refinement.loss import SNRNTotalLoss
from src.ai_refinement.dataset import Phase1OutputSimDataset

def save_plots(history, output_dir="outputs/debug"):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure()
    plt.plot(history['train_loss'], label='Train Total Loss')
    plt.plot(history['val_loss'], label='Val Total Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

def main():
    print("========================================")
    print("PHASE 11: GEOMETRIC CONSISTENCY TRAINING")
    print("========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_size = 1000
    batch_size = 16
    epochs = 30
    lr = 1e-3
    
    print(f"Loading Phase 11 Dataset (N={dataset_size})...")
    dataset = Phase1OutputSimDataset(num_samples=dataset_size, apply_aug=True)
    
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    model = SNRN().to(device)
    criterion = SNRNTotalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    ckpt_dir = "outputs/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    last_model_path = os.path.join(ckpt_dir, "last_model.pth")
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    
    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_pos_conf': [], 'val_neg_conf': [], 'lr': []}
    
    if os.path.exists(last_model_path):
        print(f"Resuming from checkpoint: {last_model_path}")
        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and torch.cuda.is_available():
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint.get('history', history)
        
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
                
            optimizer.zero_grad()
            
            with autocast(enabled=torch.cuda.is_available()):
                preds = model(batch['reference_patch'], batch['candidate_patch'])
                loss, _, _, _ = criterion(preds, batch)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pos_conf = 0.0
        val_neg_conf = 0.0
        pos_count = 0
        neg_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    
                with autocast(enabled=torch.cuda.is_available()):
                    preds = model(batch['reference_patch'], batch['candidate_patch'])
                    loss, _, _, _ = criterion(preds, batch)
                
                val_loss += loss.item()
                
                # Confidence tracking
                conf_labels = batch['confidence_label'].squeeze(1)
                pred_confs = preds['confidence'].squeeze(1)
                for i in range(len(conf_labels)):
                    if conf_labels[i].item() > 0.5:
                        val_pos_conf += pred_confs[i].item()
                        pos_count += 1
                    else:
                        val_neg_conf += pred_confs[i].item()
                        neg_count += 1
                
        N = max(1, len(val_loader))
        val_loss /= N
        avg_pos_conf = val_pos_conf / max(1, pos_count)
        avg_neg_conf = val_neg_conf / max(1, neg_count)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pos_conf'].append(avg_pos_conf)
        history['val_neg_conf'].append(avg_neg_conf)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Pos Conf: {avg_pos_conf:.4f} | Neg Conf: {avg_neg_conf:.4f} | LR: {current_lr:.5f}")
              
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if torch.cuda.is_available() else None,
            'best_val_loss': best_val_loss,
            'history': history
        }
        
        torch.save(checkpoint, last_model_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_model_path)
            
    print("\nTraining Complete.")
    save_plots(history)

if __name__ == "__main__":
    main()
