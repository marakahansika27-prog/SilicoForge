# Phase 2 Fix Report: Drift-Sense V2

## 1. Baseline
The localization pipeline was executing correctly but generating massive spatial ambiguity resulting in false positive candidates and a reported localization error of ~898 pixels. The error was artificially inflated by a dataset evaluator defect (`[0,0]` GT fallback), exacerbated by scale-blind hypothesis generation and a low-frequency Hybrid NCC configuration that rewarded macro-periodic lattice clones over true local distinctness.

## 2. Root Cause #1 (Dataset Ground-Truth Bug)
- **Defect:** `src/ai_refinement/dataset.py` unconditionally forced `gt_coord = np.array([0,0])` for all samples in `__getitem__`, destroying metrics.
- **Fix:** Correctly mapped the exact mathematical `gt_coord` for positive samples and assigned `[-1.0, -1.0]` for absent samples.

## 3. Root Cause #2 (Scale-Blind Inference)
- **Defect:** GSPE only evaluated `scale=10.0` and `rotation=0.0`.
- **Fix:** Implemented a coarse-to-fine multi-hypothesis search covering Phase 2 variance.

## 4. Root Cause #3 (Destructive Candidate Ranking)
- **Defect:** The 50/50 Hybrid score erased critical high-frequency topological pitch (e.g. 20x60 in FinFET) due to a 31px Gaussian blur.
- **Fix:** Shifted the scoring weight to `0.9 * Raw + 0.1 * LowFreq` and implemented a geometry-aware NMS threshold.

## 5. Root Cause #4 (Weak Verifiers)
- **Defect:** `verify_inference.py` passed any execution that did not return `NaN`.
- **Fix:** Split verification into `EXECUTION PASS` and `LOCALIZATION ACCURACY PASS`, explicitly filtering and logging valid metrics.

## 6. Exact Files Changed
- `src/ai_refinement/dataset.py`
- `src/coarse_search/gspe.py`
- `scripts/verify_inference.py`
- `scripts/verify_dataset.py`

## 7. Exact Functions Changed
- `Phase1OutputSimDataset._generate_sample` and `__getitem__`
- `GlobalSearchProposalEngine.run`
- `verify_inference.py:main`
- `verify_dataset.py:main`

## 8. Coordinate Convention (Contract)
The following mathematical convention governs the corrected pipeline:
1. **Generator Base Coords:** Center `(5000, 5000)` on a 10000x10000 image.
2. **Search Image Coords:** Scaled by 1/10. Origin is Top-Left.
3. **Ground-Truth Coordinate (`gt_coord`):** Floating-point coordinate of the true target **CENTER** in search space.
4. **GSPE Candidate (`x, y`):** Integer coordinate representing the **TOP-LEFT** of the candidate bounding box.
5. **Classical Coordinate:** Floating-point coordinate of the classical predicted **CENTER** in search space.
6. **Residual (`target_delta`):** `gt_coord - classical_coord`. Bounded strictly `< 5.0` px.

## 9. Scale/Rotation Search Design (Coarse-to-Fine)
GSPE now evaluates a 15-hypothesis spatial grid:
- **Scales:** 8.0, 9.0, 10.0, 11.0, 12.0
- **Rotations:** -5.0, 0.0, +5.0
1. All 15 hypotheses are swept across a 2x-downsampled image pyramid using raw NCC.
2. The Top-3 geometric hypotheses are retained based on coarse scoring.
3. Only the Top-3 hypotheses are promoted to full-resolution `matchTemplate` extraction.
4. This preserves the ~150ms runtime goal by avoiding 15 consecutive full-resolution NCC operations.

## 10. Candidate Ranking Design
- Changed weighting to `0.9` Raw and `0.1` LowFreq. The low-frequency component now functions purely to break ties in flat periodic macro-regions without masking the high-frequency topological matches.
- **Geometry-Aware NMS:** The fixed 10-pixel NMS radius (which previously failed to suppress 20x60 pitch periodic clones) is replaced by `max(10, w_cand / 4.0)`. NMS suppression is now proportional to the candidate's scale.

## 11. Verification Changes
- `verify_dataset.py` now explicitly writes correlation outcomes to `outputs/reports/DATASET_REPORT.md` and generates standard `0` or `1` exit codes.
- `verify_inference.py` filters metrics on `confidence_label == 1.0` and clearly bifurcates output between structural execution success and numerical localization accuracy.

## 12–15. Before/After Accuracy & Benchmarks
*(Note: As the terminal environment is actively blocking direct Python execution via `Access is denied`, the numerical verification benchmarks could not be executed locally in this session. The architectural pipelines have been repaired strictly based on the mathematical root causes requested. You may immediately measure the empirical outcomes of these structural fixes by running the standard Phase 2 verification suite.)*

- **Baseline Error:** ~898 px
- **Corrected GT Error:** [Pending Benchmark Execution]
- **Classical Error after GSPE fix:** [Pending Benchmark Execution]
- **AI Refined Error:** [Pending Benchmark Execution]
- **AI Improvement:** [Pending Benchmark Execution]
- **DRAM / FinFET Splits:** [Pending Benchmark Execution]
- **Mean Runtime:** [Pending Benchmark Execution]
- **GSPE Runtime:** [Pending Benchmark Execution]

## 16. Remaining Problems
- The SNRN (AI Refinement) model was trained exclusively on simulated data constrained strictly to `scale=10.0`. It may struggle to refine candidates at `scale=8.0` or `12.0` because it has never seen a patched geometry mapped outside 10.0x scale. 

## 17. Recommended Next Optimization
Before performing any further C++ level optimizations on GSPE, execute `verify_training.py` using the newly repaired `dataset.py` pipeline. If AI refinement errors are high at the extremities of the Phase 2 scales, we must augment the `Phase1OutputSimDataset` to extract patches at variable geometric scales to train SNRN for real-world Phase 2 environments.
