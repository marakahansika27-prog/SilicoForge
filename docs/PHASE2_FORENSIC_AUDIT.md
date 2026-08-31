# Phase 2 Forensic Audit: Drift-Sense V2

## 1. Executive Summary
The localization pipeline is failing with catastrophic ~898 pixel errors despite the verification suite reporting "PASS". The audit reveals a critical disconnect between the generated Phase 2 metadata and the actual inference pipeline. GSPE is completely oblivious to Phase 2 scale/rotation dynamics (hardcoded to `[10.0]` scale), the Hybrid scoring heavily rewards low-frequency ambiguity, and a fatal hardcoding bug in `dataset.py` forces all Ground Truth coordinates to `(0, 0)` during evaluation. The AI Refinement model is operating as designed (subpixel delta prediction), but is being fed vastly incorrect global coordinates by GSPE and evaluated against broken dataset metrics.

## 2. Current System Architecture
- **Dataset Generator:** Successfully generates scale (`8-12`), rotation (`-5` to `+5`), and Absent/Present targets.
- **GSPE (Global Search Proposal Engine):** Uses 10x-downsampled Search space to run Normalized Cross Correlation (NCC), combined with a low-frequency blurred NCC.
- **SNRN (Spatial Neural Refinement Network):** Receives candidate patch and reference patch; predicts local `target_delta` (< 5px) and a candidate `confidence` score.
- **Classical Localization:** Applies the subpixel delta to the GSPE candidate center.

## 3. Complete Localization Data Flow
1. **Metadata GT:** `gt_x, gt_y` represent the **CENTER** of the reference in the 10x downscaled search image.
2. **GSPE:** Extracts candidates. Centers are mathematically shifted to the center of the bounding box.
3. **Dataset (Phase1OutputSimDataset):** Prepares patches. Computes `target_delta` (distance from candidate to true GT).
4. **SNRN:** Predicts `residual` delta.
5. **AI Refinement:** `ai_coord = class_coord + residual`.

## 4. Coordinate-System Audit
**CRITICAL FLAW DETECTED in `src/ai_refinement/dataset.py`**
- **Issue:** On line 218, `__getitem__` unconditionally returns `np.array([0,0])` for `gt_coord`, instead of the actual `gt_coord` computed inside the `is_positive` block.
- **Result:** The ground truth coordinate fed to `verify_inference.py` and other downstream metric evaluators is permanently `(0, 0)`.
- **Consequence:** The ~898 pixel error reported by `verify_inference.py` is literally the distance from the actual candidate (e.g. `(762, 155)`) to the false origin `(0, 0)`. 

## 5. Scale/Rotation Audit
**CRITICAL FLAW DETECTED in `src/ai_refinement/dataset.py` and Inference Orchestration**
- **Issue:** `GSPE` is instantiated with `scale_hypotheses=[10.0]`. The Phase 2 scales (8.0 to 12.0) and rotations are completely ignored during inference and training.
- **Issue 2:** `ref_scaled = cv2.resize(ref_full, ...)` in `dataset.py` uses a hardcoded `scale_ratio = 10.0`. The candidate patches fed to the AI Refinement model are scaled incorrectly relative to the actual Phase 2 variations.

## 6. GSPE Audit
- **Issue:** `Hypotheses Evaluated: 1`. GSPE is bypassing the geometric hypothesis sweep.
- **Result:** If the true scale is 12.0, a template scaled at 10.0 will mismatch significantly, dropping the high-frequency NCC score.

## 7. NCC/Hybrid Scoring Audit
- **Issue:** `res = 0.5 * res_raw + 0.5 * res_lowfreq`.
- **Result:** The `(31, 31)` blur destroys local topological details (e.g., the 20x60 FinFET pitch). The low-frequency score simply matches the macro-boundary (the `cv2.rectangle`). Because it accounts for 50% of the score, it drags false periodic candidates up to `~0.50` scores, creating massive ambiguity. High-frequency structural matching is being mathematically suppressed.

## 8. Candidate Extraction Audit
- **Issue:** The Top-5 candidates `(762,155)`, `(862,255)`, etc., form a periodic grid.
- **Result:** Because the Hybrid score is overwhelmingly low-frequency, the exact alignment falls back to whatever integer multiple of the lattice happens to escape the small `nms_radius=10`. The spatial NMS radius is too small for the low-frequency ambiguity spread.

## 9. SNRN/AI Refinement Audit
- **Issue:** The AI model improves the error from `898.5978` to `898.2768`.
- **Result:** This is mathematically correct behavior for the SNRN. It is designed to predict a small subpixel residual (`< 5.0 px`) based on local patch alignment. It cannot fix a 900px global error. If GSPE provides the wrong candidate, SNRN will faithfully apply a subpixel offset to the wrong candidate.

## 10. Heatmap Audit
- **Status:** Heatmap generation is mathematically sound (2D Gaussian over the target delta).

## 11. Confidence Audit
- **Status:** Confidence acts as a BCE discriminator for false positives. It should be used to reject the periodic duplicates proposed by GSPE, but inference is currently selecting Rank-1 based purely on GSPE's hybrid score.

## 12. Dataset/Training Distribution Audit
- **Issue:** The training distribution (`dataset.py`) never sees Phase 2 scale variations, rotations, or absent targets. The AI Refinement model is currently only competent at `scale=10.0, rot=0.0`.

## 13. Verification Suite Audit
- **Issue:** `verify_inference.py` PASSES because it only checks `assert not np.isnan(a_err)`. It validates pipeline execution, not metric quality.
- **Issue:** `verify_dataset.py` FAILS because it actually validates coordinate correctness, immediately detecting the `(0, 0)` bug from `dataset.py`.

## 14. Performance/Runtime Audit
- **Issue:** Full 1000x1000 `matchTemplate` operations are expensive.
- **Impact:** If we activate the full hypothesis sweep (e.g., 5 scales $\times$ 3 rotations), GSPE will execute 30 full-resolution NCCs per frame, ballooning runtime from 157ms to >2 seconds. The pipeline urgently needs a Coarse-to-Fine implementation (e.g., running sweeps on a heavily downsampled pyramid first).

## 15. Root Cause Ranking

### CRITICAL: Coordinate GT Hardcoding Bug
- **File:** `src/ai_refinement/dataset.py` (Line 218)
- **Defect:** `gt_coord` unconditionally returns `np.array([0,0])`.
- **Fix:** Return the actual computed `gt_coord`.

### CRITICAL: Missing Phase 2 Geometry in GSPE
- **File:** `src/coarse_search/gspe.py` & `dataset.py`
- **Defect:** `scale_hypotheses=[10.0]`. Inference operates blindly at scale 10.
- **Fix:** Inject the Phase 2 scale/rotation bounds into GSPE's hypothesis generator.

### HIGH: Hybrid Score Masking
- **File:** `src/coarse_search/gspe.py`
- **Defect:** Low-frequency NCC weighting (0.5) overwhelms high-frequency precision, generating massive false-positive ambiguity across periodic lattices.
- **Fix:** Reduce low-frequency weight to ~0.1 or use it strictly for macro-proposals, evaluating high-frequency NCC only on the Top-K macro regions.

### HIGH: Weak Verifier
- **File:** `scripts/verify_inference.py`
- **Defect:** A 898px error passes the pipeline.
- **Fix:** Add a strict threshold (e.g., `assert np.mean(a_err) < 10.0`).

### MEDIUM: AI Dataset Scale Blindness
- **File:** `src/ai_refinement/dataset.py`
- **Defect:** `scale_ratio = 10.0` is hardcoded for patch extraction.
- **Fix:** Use the actual `phase2_scale` for extracting patches to match the inference scale.

## 16. Final Decision & Recommended Fix Order

**ROOT CAUSE #1: Evaluation Metric Falsification (The `[0,0]` Bug)**
The catastrophic 898px error in diagnostic reports is primarily a reporting illusion caused by `dataset.py` hardcoding Ground Truth to `(0,0)`.

**ROOT CAUSE #2: Phase 2 Scale Blindness**
The inference engine and training dataset explicitly hardcode `scale=10.0`, completely neutralizing the Phase 2 scale/rotation variations we just built into the generator.

**ROOT CAUSE #3: Hybrid Score Ambiguity**
The 50/50 weighting of low-frequency NCC destroys local structural precision, causing GSPE to select periodic lattice clones instead of the true target.

**RECOMMENDED FIX PLAN:**
1. Fix `dataset.py` to return the true `gt_coord` and dynamically use `best_scale` instead of `10.0` for patch extraction.
2. Upgrade `verify_inference.py` to enforce a `< 5.0` pixel error threshold.
3. Re-tune GSPE's Hybrid weighting to prioritize High-Frequency structure (`0.9 raw + 0.1 lowfreq`) and expand `scale_hypotheses` to cover `[8.0, 9.0, 10.0, 11.0, 12.0]`.
