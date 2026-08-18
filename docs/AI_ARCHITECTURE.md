# Drift-Sense V2 AI Architecture

## 1. Sub-pixel Navigation Refinement Network (SNRN)
The SNRN is a Siamese network built on a lightweight `ResNet18` backbone (truncated at `layer3` to preserve $8 \times 8$ spatial resolution from $128 \times 128$ input patches).

## 2. Feature Fusion via Cross-Attention
- **Query**: Candidate Patch Features $(B, 256, 64)$
- **Key / Value**: Reference Patch Features $(B, 256, 64)$
The Cross-Attention mechanism spatially queries the reference template based on the visual semantics of the candidate, allowing explicit feature matching before the decoder stage.

## 3. Heads
The fused tensor $(B, 512, 8, 8)$ passes through a shared CNN decoder to generate $(B, 128, 8, 8)$ features.
- **Residual Head**: Adaptive Avg Pool $\to$ FC $\to$ Predicts $\Delta x, \Delta y$.
- **Heatmap Head**: $4\times$ ConvTranspose2d $\to$ Predicts $128 \times 128$ spatial distribution.
- **Confidence Head**: Adaptive Avg Pool $\to$ FC $\to$ Predicts $P_{success} \in [0,1]$.
