import torch
import os
import cv2
import numpy as np

def save_tensor_image(tensor, path):
    """Converts a (1, H, W) tensor to an image and saves it."""
    img = tensor.squeeze().detach().cpu().numpy()
    if img.max() <= 1.01:
        img = img * 255
    cv2.imwrite(path, img.astype(np.uint8))
    
def save_attention_overlay(cand_patch, attn_map, path):
    """Overlays attention map onto the candidate patch."""
    cand = cand_patch.squeeze().detach().cpu().numpy()
    attn = attn_map.squeeze().detach().cpu().numpy()
    
    # attn_map is (1, 64, 64) due to cross-attention over 8x8 flattened features (64 tokens)
    # We just average across the queries to see general attention, or visualize a specific query
    attn = attn.mean(axis=0).reshape(8, 8)
    attn = cv2.resize(attn, (128, 128))
    attn = cv2.normalize(attn, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    cand = (cand * 255).astype(np.uint8) if cand.max() <= 1.01 else cand.astype(np.uint8)
    heatmap_img = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    cand_color = cv2.cvtColor(cand, cv2.COLOR_GRAY2BGR)
    
    overlay = cv2.addWeighted(cand_color, 0.5, heatmap_img, 0.5, 0)
    cv2.imwrite(path, overlay)

class SNRNTrainer:
    """Training loop for Sub-pixel Navigation Refinement Network."""
    def __init__(self, model, criterion, optimizer, debug_dir="outputs/debug/training"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.debug_dir = debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)
        
    def train_step(self, batch, epoch=0, batch_idx=0):
        self.model.train()
        self.optimizer.zero_grad()
        
        ref_patch = batch['reference_patch']
        cand_patch = batch['candidate_patch']
        
        preds = self.model(ref_patch, cand_patch)
        loss, l_res, l_hm, l_conf = self.criterion(preds, batch)
        
        loss.backward()
        self.optimizer.step()
        
        # Save visualizations for the first item in batch
        if batch_idx == 0:
            save_tensor_image(ref_patch[0], f"{self.debug_dir}/ref_patch.png")
            save_tensor_image(cand_patch[0], f"{self.debug_dir}/cand_patch.png")
            save_tensor_image(batch['target_heatmap'][0], f"{self.debug_dir}/gt_heatmap.png")
            
            # Predict heatmap ranges 0-1 (softmaxed)
            pred_hm_norm = preds['heatmap'][0] / (preds['heatmap'][0].max() + 1e-8)
            save_tensor_image(pred_hm_norm, f"{self.debug_dir}/pred_heatmap.png")
            save_attention_overlay(cand_patch[0], preds['attn_map'][0], f"{self.debug_dir}/attention_overlay.png")
            
            # Save intermediate backbone features
            save_tensor_image(preds['ref_feat'][0, 0:1], f"{self.debug_dir}/ref_backbone_feat0.png")
            
        return loss.item(), l_res, l_hm, l_conf
