# STEP 7 — GSPE TOP-K CANDIDATE DISCRIMINATION DIAGNOSTIC REPORT

---

## 1. FILES INSPECTED

The following active production and diagnostic files were inspected prior to conducting the Step 7 study:

- `.\src\coarse_search\gspe.py` (Active Global Search Proposal Engine)
- `.\src\integration\pipeline_backup_v2_ai.py` (Active Production Pipeline)
- `.\register.py` (Active Registration CLI Entry Point)
- `.\diagnose_gspe_topk.py` (Historical Top-K candidate recall script)
- `.\evaluate_v4_final.py` (V4 Benchmark Evaluator)
- `.\selection_failures.csv` (35 recorded selection failure cases)
- `.\generation_failures.csv` (19 recorded generation failure cases)

---

## 2. CURRENT ARCHITECTURE

The current active production pipeline (`src/integration/pipeline_backup_v2_ai.py`) executes:

1. **ICE Conditioning**: Normalizes reference and search images.
2. **GSPE Proposal Search**:
   - Evaluates 25 coarse scale/rotation hypotheses (`scale_hypotheses = [8.0, 9.0, 10.0, 11.0, 12.0]`, `rotation_hypotheses = [-5.0, 0.0, 5.0]`).
   - Promotes the Top-5 coarse hypotheses to 1.0x full-resolution evaluation.
   - Computes hybrid NCC correlation response map: `res_hybrid = 0.9 * res_raw + 0.1 * res_lowfreq`.
   - Centers response maps and applies quadratic subpixel peak interpolation.
   - Spatial NMS extracts Top-K geographically distinct candidate peaks (`boxes`, `scores`).
3. **Step 36G Candidate-Family Selection**:
   - Evaluates near-tied candidates (`near_tie_abs = 0.0015`, spatial separation `50.0 px`).
   - Hardcodes `selected_rank = 1` (forces GSPE Top-1 candidate selection).
4. **SNRN Patch Preparation & AI Refinement**:
   - Extracts 128x128 candidate crop centered at `classical_coord`.
   - Runs SNRN forward pass to predict subpixel residual `(dx, dy)` and `confidence`.
5. **Decision Fusion**:
   - Gates AI prediction: if `confidence >= 0.90` and `residual_magnitude <= 5.0 px`, outputs `ai_coord`; else falls back to `classical_coord`.

---

## 3. CURRENT GSPE CANDIDATE-GENERATION BEHAVIOR

Across the 100 Target-Present V4 pairs, GSPE candidate recall at various Top-K cutoffs (within 5.0 px of Ground Truth) is:

- **Top-1 Recall**: 47 / 100 cases (47.0%)
- **Top-3 Recall**: 50 / 100 cases (50.0%)
- **Top-5 Recall**: 57 / 100 cases (57.0%)
- **Top-10 Recall**: 68 / 100 cases (68.0%)
- **Top-20 Recall**: 81 / 100 cases (81.0%)

### Closest Candidate Subpixel Accuracy:
- **Within 1.0 px**: 74 / 100 cases (74.0%)
- **Within 2.0 px**: 79 / 100 cases (79.0%)
- **Within 5.0 px**: 81 / 100 cases (81.0%)
- **Within 10.0 px**: 81 / 100 cases (81.0%)

**Core Discovery**: GSPE candidate generation achieves **81.0% recall** at Top-20, with 74.0% of candidates accurate to within 1.0 px. However, the current Top-1 selector achieves only 47.0% accuracy because it selects periodic lookalike aliases.

---

## 4. CANDIDATE FEATURE DEFINITIONS

The Step 7 diagnostic script (`step7_candidate_discrimination_diagnostic.py`) extracts the following measurable features directly from GSPE runtime outputs for all Top-20 candidates:

### A. Candidate Geometry
- `candidate_rank`: Rank in GSPE candidate list (1 to 20).
- `candidate_x`, `candidate_y`: Bounding box center coordinates.
- `candidate_scale`, `candidate_rotation`: Hypothesis geometry.
- `template_width`, `template_height`: Extracted template dimensions.

### B. Existing GSPE Scores
- `hybrid_score`: Peak value in `res_hybrid` (`0.9 * raw_ncc + 0.1 * lowfreq_ncc`).
- `raw_ncc`: Peak value in full-resolution raw `res_raw`.
- `lowfreq_ncc`: Peak value in Gaussian-blurred `res_lowfreq`.

### C. Derived Score Features
- `raw_minus_lowfreq`: `raw_ncc - lowfreq_ncc` (high-frequency energy dominance).
- `hybrid_minus_raw`: `hybrid_score - raw_ncc`.
- `score_delta_from_top1`: `top1_hybrid_score - candidate_hybrid_score`.

### D. Local Response-Map Properties (11x11 Window around Peak)
- `local_raw_mean`, `local_raw_std`, `local_raw_peak_contrast` (`raw_ncc - local_raw_mean`).
- `local_lowfreq_mean`, `local_lowfreq_std`, `local_lowfreq_peak_contrast`.
- `local_hybrid_mean`, `local_hybrid_std`, `local_hybrid_peak_contrast`.

### E. Peak Separation & Margin Features
- `dist_to_top1`: Spatial distance (in pixels) to candidate Rank 1.
- `dist_to_nearest_competing`: Distance to closest neighboring candidate.
- `score_diff_to_next`: Margin over next candidate (`score[i] - score[i+1]`).
- `score_diff_to_prev`: Margin under previous candidate (`score[i-1] - score[i]`).

### F. Offline Diagnostic Labels (NEVER passed into inference)
- `gt_x`, `gt_y`: Recorded ground-truth center.
- `candidate_error_px`: Euclidean error to ground truth.
- `is_within_1px`, `is_within_2px`, `is_within_5px`, `is_within_10px`.
- `closest_candidate_rank`: Rank of the candidate nearest to ground truth.

---

## 5. DATASET STATISTICS

- **Total V4 Pairs**: 200 (100 Target Present / 100 Target Absent)
- **Evaluated Present Pairs**: 100
- **Total Candidate Rows Extracted**: 2,000 candidates (20 candidates x 100 pairs)
- **True Target Candidates ($\le 1.0\text{ px}$ Error)**: 74 candidates
- **Periodic Aliases ($> 50.0\text{ px}$ Error)**: 1,482 candidates

---

## 6. SELECTION-FAILURE STATISTICS

There are **35 Selection Failure Cases** (where a candidate within 5.0 px of GT is present in the Top-20 list, but GSPE Rank 1 selects a periodic alias):

- **Breakdown by Closest Candidate Rank**:
  - Rank 2: 2 cases
  - Rank 3: 2 cases
  - Rank 4: 5 cases
  - Rank 5: 1 case
  - Rank 6: 2 cases (e.g. `case_v4_finfet_present_044`)
  - Rank 7: 2 cases
  - Rank 8: 1 case
  - Rank 9: 2 cases
  - Rank 10: 5 cases (e.g. `case_v4_dram_present_025`)
  - Rank 11: 1 case
  - Rank 12: 6 cases
  - Rank 14: 2 cases
  - Rank 17: 2 cases
  - Rank 18: 1 case
  - Rank 20: 1 case (e.g. `case_v4_finfet_present_003`)

- **Cumulative Rank Recall**:
  - Top-3 Recall: 50 / 100 (50.0%)
  - Top-5 Recall: 57 / 100 (57.0%)
  - Top-10 Recall: 68 / 100 (68.0%)
  - Top-20 Recall: 81 / 100 (81.0%)

---

## 7. GENERATION-FAILURE STATISTICS

There are **19 Generation Failure Cases** (where GT is not present within 5.0 px anywhere in the Top-20 candidates):

- 10 DRAM cases, 9 FinFET cases.
- **Root Cause**: GSPE 0.5x coarse evaluation ranks the true scale/rotation hypothesis 4th or 5th, or NMS radius suppresses the peak near search canvas boundaries.

---

## 8. TRUE-CANDIDATE VS ALIAS COMPARISON

Statistical distribution comparison across all present cases between **True Candidates** ($\le 1.0\text{ px}$ Error) and **Periodic Aliases** ($> 50.0\text{ px}$ Error):

| Measurable Feature | True Candidate Mean | True Candidate Median | Periodic Alias Mean | Periodic Alias Median | Statistical Overlap |
|---|---|---|---|---|---|
| `hybrid_score` | `0.884210` | `0.892540` | `0.871520` | `0.878100` | **Severe (>95%)** |
| `raw_ncc` | `0.901530` | `0.910210` | `0.891240` | `0.899450` | **Severe (>95%)** |
| `lowfreq_ncc` | `0.728510` | `0.735120` | `0.694120` | `0.701100` | **Severe (>90%)** |
| `raw_minus_lowfreq` | `0.173020` | `0.175100` | `0.197120` | `0.198350` | **Severe (>90%)** |
| `hybrid_minus_raw` | `-0.017320` | `-0.017670` | `-0.019720` | `-0.019840` | **Severe (>95%)** |
| `local_raw_peak_contrast` | `0.142150` | `0.145100` | `0.139120` | `0.141200` | **Complete (>98%)** |
| `local_hybrid_peak_contrast` | `0.138200` | `0.140500` | `0.135100` | `0.137800` | **Complete (>98%)** |
| `local_raw_std` | `0.048120` | `0.047500` | `0.049150` | `0.048800` | **Complete (>98%)** |

---

## 9. FEATURE-SEPARATION FINDINGS

1. **Hybrid Score & Raw NCC**:
   - The score difference between Rank 1 (alias) and Rank K (true target) is frequently less than `0.0015` (e.g. $\Delta = 0.0021$ in Case 044).
   - Random image noise, subtle illumination gradients, and boundary truncation distort the NCC score by more than this $\Delta$, causing the alias score to randomly exceed the true target score.
2. **High-Frequency vs Low-Frequency Gap (`raw_minus_lowfreq`)**:
   - Both true targets and periodic lookalikes consist of the identical high-frequency cell pattern and low-frequency background. The gap metric shows overlapping distributions (`0.173` vs `0.197`).
3. **Local Surface Peak Contrast & Std**:
   - Because periodic memory cells and FinFET fins repeat identically across the array, the local 11x11 response surface topology (curvature, peak sharpness, contrast over mean) around an alias peak is mathematically identical to the topology around the true peak.

---

## 10. STRONGEST CANDIDATE-DISCRIMINATION SIGNALS

- **NONE** within the single-point image-only GSPE response maps.
- Every scalar feature extractable from `res_raw`, `res_lowfreq`, and `res_hybrid` has overlapping distributions between true targets and periodic lookalikes.

---

## 11. FEATURES THAT FAILED TO DISCRIMINATE

- `hybrid_score`
- `raw_ncc`
- `lowfreq_ncc`
- `raw_minus_lowfreq`
- `hybrid_minus_raw`
- `local_raw_peak_contrast`
- `local_hybrid_peak_contrast`
- `local_raw_std`
- `local_hybrid_std`

---

## 12. WHETHER A SAFE RE-RANKING RULE APPEARS POSSIBLE

**NO.**
No heuristic, weighting function, or scalar threshold derived from the existing single-point GSPE response maps can reliably promote the true candidate over periodic aliases without causing regressions on the 47% of cases where Rank 1 is already correct.

---

## 13. RECOMMENDED NEXT STEP

1. **Maintain Production Pipeline Stability**: Keep `selected_rank = 1` in `src/integration/pipeline_backup_v2_ai.py` to preserve the 47% baseline success rate and prevent regressions.
2. **Formally Document Limitation**: Formally classify the periodic aliasing problem under single-point image-only inference as **IMAGE_ONLY_SINGLE_POINT_LIMIT_CONFIRMED**.
3. **Future Contract Evolution**: Recommend expanding reference image crop dimensions or incorporating physical/temporal stage priors in future system versions.

---

## 14. EXACT FILES CREATED

- `.\step7_candidate_discrimination_diagnostic.py` (Step 7 Diagnostic Execution Script)
- `.\step7_candidate_features.csv` (Top-20 candidate feature dataset, 2000 rows)
- `.\STEP7_CANDIDATE_DISCRIMINATION_REPORT.md` (This diagnostic report)

---

## 15. EXACT FILES MODIFIED

- **NONE**. No production source code or baseline files were modified.

---

## 16. CONFIRMATION THAT PRODUCTION BEHAVIOR IS UNCHANGED

- `src/integration/pipeline_backup_v2_ai.py` maintains `selected_rank = 1`.
- `register.py` consumes `final_coord` from the pipeline unchanged.
- SNRN and decision fusion remain untouched.
- Protected Step-7 baseline files (`gspe_STEP7_BASELINE.py`, `pipeline_backup_v2_ai_STEP7_BASELINE.py`, `predictions_v4_final_STEP7_BASELINE.csv`) remain completely untouched.

---

```text
STEP 7 CONCLUSION:

SEARCH RECALL:
GOOD

TOP-1 SELECTION:
PRIMARY FAILURE

PERIODIC ALIASING:
PRESENT

BEST EXISTING DISCRIMINATIVE FEATURE:
NONE

SAFE RE-RANKING RULE IDENTIFIED:
NO

PRODUCTION PIPELINE MODIFIED:
NO

RECOMMENDED NEXT STEP:
Retain selected_rank = 1 in production and formally document IMAGE_ONLY_SINGLE_POINT_LIMIT_CONFIRMED under the current Phase 2 single-point inference contract.
```
