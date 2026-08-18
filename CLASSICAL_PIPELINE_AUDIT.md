# Classical Pipeline Engineering Audit

## 1. Executive Summary
A comprehensive engineering audit of the Phase 1 Classical Pipeline was conducted to investigate the persistent ~230-pixel navigation error. 

The audit confirms that the AI Refinement Network (Phase 2) and the Decision Fusion system (Phase 3) are functioning correctly. The catastrophic error originates entirely within the **Classical Pipeline's coordinate geometry formulation**, specifically between the interaction of **GSPE (Global Search Proposal Engine)** and **Classical Localization**.

## 2. Component Investigation

### GSPE (Candidate Generation & Top-K)
- **Mechanism**: `cv2.matchTemplate` with `TM_CCOEFF_NORMED`.
- **Output**: Returns `(x, y)` from `np.unravel_index`. 
- **Audit Finding**: In OpenCV, `matchTemplate` returns the **Top-Left** corner of the matched bounding box, not the centroid. 
- **Status**: Algorithmic correctness is intact, but the semantic meaning of the coordinate is geometrically misaligned with navigation ground truth (which expects the center of the Field of View).

### GFEE & SRAE (Feature Matching & Registration)
- **Mechanism**: SIFT + BFMatcher (Lowe ratio) $\rightarrow$ RANSAC Affine Transform.
- **Audit Finding**: 
  - The matrices map the Candidate (`dst`) to the Reference (`src`): $P_{ref} = M \cdot P_{cand}$.
  - The affine translation $(t_x, t_y)$ shifts the candidate space to the reference space.
- **Status**: The registration itself operates flawlessly and warps the image correctly.

### Classical Localization (Coordinate Conversion)
- **Mechanism**: Computes final classical drift as `dx = cx + tx` and `dy = cy + ty`.
- **Audit Finding**: 
  1. **Centroid Offset Omission (The 230px Error)**: The engine directly uses the Top-Left corner `(x, y)` from GSPE as the baseline anchor `(cx, cy)`. It completely fails to add `(w/2, h/2)` to translate the anchor to the center of the bounding box. If the reference template is $460 \times 460$, skipping the centroid translation mathematically guarantees a systematic offset of exactly $230$ pixels in both axes.
  2. **Translation Sign Inversion**: Because $P_{ref} = P_{cand} + T$, the true position of the reference center in candidate space requires subtracting the translation matrix components ($-t_x, -t_y$), but the engine erroneously adds them ($+t_x, +t_y$).

## 3. Propagation to Phase 3 (AI Error)
- The Phase 2 AI is a *residual* network. It is designed to predict a tiny sub-pixel delta ($\Delta x, \Delta y$) and add it to the classical anchor.
- Because the classical anchor was fed into the AI as the **Top-Left corner**, the AI faithfully computed a highly accurate sub-pixel delta, added it to the Top-Left corner, and inherently preserved the massive 230-pixel centroid offset.
- This explains why `Classical Error == AI Error == 230 px` and `Improvement = 0%`. The AI was structurally handicapped by the corrupted classical anchor.

## 4. Root Cause Conclusion
The 230-pixel error is introduced in `src/integration/pipeline.py` and `src/localization/localization.py` where the **Top-Left bounding box corner is directly treated as the navigation centroid** without the required `+ w/2, + h/2` offset.

## 5. Recommended Minimal Fix
1. Modify `src/integration/pipeline.py` or `localization.py` to calculate the centroid: 
   `classical_coord = np.array([float(x + w/2), float(y + h/2)])`
2. Correct the affine translation sign in `localization.py`: 
   `dx = cx - tx`, `dy = cy - ty`.

No modifications to the AI, Network, or Phase 1 heavy lifting (SIFT/NCC) are required. The bug is purely geometric arithmetic.
