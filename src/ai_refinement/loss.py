import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ai_refinement.config import SNRNConfig

class SNRNTotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.bce = nn.BCELoss()

    def forward(self, preds, targets):
        """
        preds: dict containing 'residual', 'heatmap_logits', 'heatmap', 'confidence'
        targets: dict containing 'target_delta', 'target_heatmap', 'confidence_label'
        """
        # Residual Loss (Masked for negatives)
        self.smooth_l1_none = nn.SmoothL1Loss(reduction='none')
        raw_loss_res = self.smooth_l1_none(preds['residual'], targets['target_delta']).mean(dim=1)
        loss_res = (raw_loss_res * targets['confidence_label'].squeeze()).mean()
        
        # Heatmap Loss (Masked for negatives)
        B = preds['heatmap_logits'].size(0)
        pred_log_prob = F.log_softmax(preds['heatmap_logits'].view(B, -1), dim=1)
        target_prob = targets['target_heatmap'].view(B, -1)
        
        self.kl_div_none = nn.KLDivLoss(reduction='none')
        raw_loss_hm = self.kl_div_none(pred_log_prob, target_prob).sum(dim=1) # Sum over spatial dims
        loss_hm = (raw_loss_hm * targets['confidence_label'].squeeze()).mean()
        
        # Confidence Loss (Always applied)
        safe_conf = torch.clamp(preds['confidence'], min=1e-6, max=1.0 - 1e-6)
        loss_conf = self.bce(safe_conf, targets['confidence_label'])
        
        # Total Weighted Loss
        total_loss = (SNRNConfig.WEIGHT_RESIDUAL * loss_res + 
                      SNRNConfig.WEIGHT_HEATMAP * loss_hm + 
                      SNRNConfig.WEIGHT_CONFIDENCE * loss_conf)
                      
        # FIX: Protect against NaN/Inf explosions gracefully
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=total_loss.device, requires_grad=True)
            
        return total_loss, loss_res.item(), loss_hm.item(), loss_conf.item()
