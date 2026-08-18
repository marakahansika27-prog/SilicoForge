import os

class SNRNConfig:
    """Configuration for Sub-pixel Navigation Refinement Network."""
    
    # Input
    PATCH_SIZE = 128
    
    # Loss Weights
    WEIGHT_RESIDUAL = 1.0    # SmoothL1 Loss
    WEIGHT_HEATMAP = 0.5     # KL Divergence Loss
    WEIGHT_CONFIDENCE = 0.2  # BCE Loss
    
    # Heatmap Generation
    HEATMAP_SIGMA = 3.0
    
    # Classical Validation
    MAX_CLASSICAL_ERROR = 5.0  # Pixels. Discard samples where classical error > this.
    
    # Confidence Label Generation
    CONFIDENCE_THRESHOLD = 2.0 # Pixels. If residual error < this, confidence label = 1.
