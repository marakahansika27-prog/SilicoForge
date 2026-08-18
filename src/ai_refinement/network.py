import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def _replace_bn_with_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            C = child.num_features
            # Find largest sensible divisor up to 32
            for G in [32, 16, 8, 4, 2, 1]:
                if C % G == 0:
                    setattr(module, name, nn.GroupNorm(G, C))
                    break
        else:
            _replace_bn_with_gn(child)

class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # Modify for 1-channel grayscale input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3 # Output stride 16 (8x8 for 128x128 input)
        # We drop layer4 to keep 8x8 spatial resolution and lightweight compute
        
        _replace_bn_with_gn(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x # (B, 256, 8, 8)

class CrossAttention(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.scale = d_model ** -0.5
        self.q_proj = nn.Conv2d(d_model, d_model, 1)
        self.k_proj = nn.Conv2d(d_model, d_model, 1)
        self.v_proj = nn.Conv2d(d_model, d_model, 1)
        self.out_proj = nn.Conv2d(d_model, d_model, 1)

    def forward(self, query_feat, key_val_feat):
        # query_feat: Candidate, key_val_feat: Reference
        B, C, H, W = query_feat.shape
        
        Q = self.q_proj(query_feat).view(B, C, -1) # (B, C, H*W)
        K = self.k_proj(key_val_feat).view(B, C, -1)
        V = self.v_proj(key_val_feat).view(B, C, -1)
        
        # Q^T K
        attn = torch.bmm(Q.transpose(1, 2), K) * self.scale # (B, H*W, H*W)
        attn = F.softmax(attn, dim=-1)
        
        # attn * V
        out = torch.bmm(V, attn.transpose(1, 2)) # (B, C, H*W)
        out = out.view(B, C, H, W)
        return self.out_proj(out), attn

class SNRN(nn.Module):
    """Sub-pixel Navigation Refinement Network"""
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.cross_attn = CrossAttention(d_model=256)
        
        # Shared Decoder (Feature Fusion)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU()
        )
        
        # Heads
        self.residual_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # Removed nn.init.zeros_ which mathematically paralyzed gradient flow.
        # Standard PyTorch initialization (Kaiming Uniform) will be used, breaking
        # symmetry and allowing gradients to propagate back to the backbone.
        
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32->64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)    # 64->128
        )
        
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, ref_patch, cand_patch):
        # 1. Feature Extraction
        ref_feat = self.backbone(ref_patch)
        cand_feat = self.backbone(cand_patch)
        
        # 2. Cross Attention (Query = Candidate, Key/Value = Reference)
        attn_out, attn_map = self.cross_attn(query_feat=cand_feat, key_val_feat=ref_feat)
        
        # 3. Fusion
        fused = torch.cat([cand_feat, attn_out], dim=1)
        dec_feat = self.decoder(fused)
        
        # 4. Heads
        residual = self.residual_head(dec_feat)
        
        # Output raw logits to allow numerically stable F.log_softmax in the Loss layer
        heatmap_logits = self.heatmap_head(dec_feat)
        B, _, H, W = heatmap_logits.shape
        heatmap = F.softmax(heatmap_logits.view(B, -1), dim=1).view(B, 1, H, W)
        
        confidence = torch.sigmoid(self.confidence_head(dec_feat))
        
        return {
            'residual': residual,
            'heatmap_logits': heatmap_logits, # Passed directly to KL Div loss
            'heatmap': heatmap,               # Passed to visualization/metrics
            'confidence': confidence,
            'attn_map': attn_map,
            'ref_feat': ref_feat,
            'cand_feat': cand_feat,
            'dec_feat': dec_feat
        }
