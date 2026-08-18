# Drift-Sense V2 Hybrid Localization Pipeline

The localization architecture sequentially fuses classical pattern-matching with deep learning refinement to achieve high-precision sub-pixel coordinate extraction. The pipeline minimizes catastrophic failure (drift) by falling back to robust classical priors whenever the AI refinement engine reports high uncertainty.

## 1. Image Conditioning Engine (ICE)
**Input:** Reference Image + Search Image
**Process:** The ICE is responsible for filtering out high-frequency sensor noise, photometric inconsistency, and structural degradation in scanning electron microscope (SEM) domains (like DRAM and FinFET captures). It performs edge-enhancement and local structural normalization.
**Output:** Normalized condition tensors.

## 2. Global Search Proposal Engine (GSPE)
**Input:** Condition tensors
**Process:** A highly optimized Normalized Cross-Correlation (NCC) sweeps over a multi-dimensional hypothesis space containing various scale and rotational invariances. It generates a holistic response surface and selects the top-$K$ candidate bounding boxes. 
**Output:** $K$ structural region proposals with associated confidence scores.

## 3. Subpixel Localization (Classical)
**Input:** Bounding Box + Response Surface
**Process:** Applies parabolic or affine sub-pixel interpolation on the localized cross-correlation peak to generate a classical pseudo-continuous coordinate $(x_c, y_c)$. 

## 4. Subpixel Navigation Refinement Network (SNRN)
**Input:** Cropped Candidate Search Tensor + Reference Tensor
**Process:** A residual neural network designed specifically for regression of micro-displacements. It acts upon the classical proposal and outputs a fine-grained subpixel offset $(\Delta x, \Delta y)$ alongside a neural confidence score.
**Output:** Predicted neural coordinate $(x_c + \Delta x, y_c + \Delta y)$ and $C_{ai}$.

## 5. Decision Fusion Engine
**Input:** Classical Coordinate + Neural Coordinate + Confidence Score
**Process:** Evaluates the AI refinement delta. If the displacement exceeds a physically plausible threshold (residual deadband) or if $C_{ai} < T_{conf}$, the fusion engine explicitly overrides the neural network and returns the classical coordinate to prevent structural hallucination.
**Output:** Final guaranteed $(x, y)$ coordinate output.
