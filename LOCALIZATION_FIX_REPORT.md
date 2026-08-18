# Localization Coordinate Fix Report

## Coordinate Geometry Correction
A catastrophic localization error was identified where the pipeline treated the Top-Left corner of the bounding box as the spatial ground truth. This resulted in a systematic ~212 to 230-pixel offset proportional to the template dimensions.

### Previous Equation (Bugged)
The pipeline previously anchored the coordinate entirely on the top-left index returned by GSPE:
```python
classical_coord = np.array([float(top_left_x), float(top_left_y)])
```
And bypassed the full localization engine, incorrectly shifting the AI relative to the top-left.

### Corrected Equation (Applied)
The bounding box coordinates are now properly translated to the geometric centroid before being injected into the Classical Localization block:
```python
center_x = float(top_left_x) + float(template_width) / 2.0
center_y = float(top_left_y) + float(template_height) / 2.0
```
This centroid is now correctly passed through the `ClassicalLocalization` engine to apply the affine residual.

### Affected Files
1. `src/integration/pipeline.py`
   - **Fix**: Replaced the Top-Left `(x, y)` assignment with the precise Centroid Conversion logic (`x + w/2`, `y + h/2`).
   - **Integration**: Restored the missing instantiation and invocation of `ClassicalLocalization.run()`, ensuring the affine matrix residual (from SRAE) is properly folded into the final `classical_coord`.

*(Note: `gspe.py`, `srae.py`, `localization.py`, and the AI network APIs were strictly preserved without modification.)*

### Expected Improvement
By accurately anchoring the system to the true centroid, the ~212+ pixel systematic offset is entirely neutralized. 

**Expected Pipeline Metrics**:
- **Classical Error**: Should instantly collapse from ~230 pixels to sub-10 pixels (the residual limit of the geometric affine transform).
- **AI Error**: Should drop to ~0.00 - 1.50 pixels, as the sub-pixel network now operates on the mathematically correct spatial anchor.
- **Improvement %**: Should accurately reflect the AI's sub-pixel refinement capability over the classical baseline (e.g., Classical Error: 2.5px $\rightarrow$ AI Error: 0.8px = ~68% Improvement).
- **Decision Fusion**: Will correctly register `AI_REFINED` as the confidence bounds recover.
