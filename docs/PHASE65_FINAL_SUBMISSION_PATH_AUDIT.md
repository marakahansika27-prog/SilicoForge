# PHASE 65 — FINAL SUBMISSION PATH AUDIT

## 1. EXACT EXECUTION PATH
The execution path from the official CLI entry point strictly follows:

`python inference.py <reference> <search>`
↓
`inference.py:main()` (validates arguments, reads images)
↓
`src.integration.pipeline_backup_v2_ai.HybridNavigationPipeline` (initialized)
↓
`pipeline.run(ref, search)`
↓
`src.preprocessing.ice.ImageConditioningEngine.run()` (image normalization)
↓
`src.coarse_search.gspe.GlobalSearchProposalEngine.run()` (multi-scale, multi-rotation search)
↓
Extracts `boxes` and `scores`.
Executes `_analyze_candidate_family` but intentionally forces `selected_rank = 1` (Top-1 fallback).
↓
Computes `classical_coord` as the exact center of the integer `boxes` output.
↓
Extracts 128x128 candidate crop for SNRN.
↓
`src.ai_refinement.network.SNRN.forward()` (computes `dx, dy` and `confidence`).
↓
`src.integration.decision_fusion.DecisionFusionEngine.fuse()` (combines `classical_coord` + `dx, dy`).
↓
Returns `final_coord` dictionary.
↓
`inference.py` formats and prints `(x.xxxx, y.xxxx)` to `stdout`.

## 2. IMPORTED MODULES INVOLVED
- `inference.py`
- `src.integration.pipeline_backup_v2_ai`
- `src.preprocessing.ice`
- `src.coarse_search.gspe`
- `src.ai_refinement.network`
- `src.ai_refinement.config`
- `src.integration.decision_fusion`

## 3. PIPELINE IDENTIFICATION
- **Production Pipeline:** `src/integration/pipeline_backup_v2_ai.py` (This is the actual production file imported by `inference.py`).
- **Legacy Pipeline:** `src/integration/pipeline.py` (Frozen classical-only configuration from Phase 28, bypassed by actual inference).
- **Experimental Pipelines / Diagnostics:** `scripts/phase5*.py` series (Candidate evidence, spatial geometry, context disambiguation).
- **Backup Files:** `src/coarse_search/gspe_backup_phase62.py` (Reference GSPE prior to 0.5px float/int mathematical fix).
- **Evaluation Scripts:**
  - `scripts/evaluate_45case.py`: Incorrectly tests `pipeline.py` (Legacy).
  - `scripts/evaluate_phase63.py`: Evaluated `pipeline.py` (Legacy).
  - `scripts/evaluate_phase64.py`: Evaluates `pipeline_backup_v2_ai.py` (Actual Production).

## 4. VERIFICATION OF SUBMISSION PATH
The actual submission path is formally confirmed as:
`inference.py` -> `pipeline_backup_v2_ai.py` -> `gspe.py` -> `network.py (SNRN)` -> single `(x,y)` output.
This utilizes the GSPE `boxes` parameter and the SNRN AI correction, which is the mathematically correct intended V2 behavior.

## 5. README.MD INACCURACIES
A review of the `README.md` identified the following discrepancies:
1. **Candidate Ranking Claims (Lines 411-414):** The README claims candidates are ranked and "The strongest spatially meaningful candidate is selected". In reality, the production pipeline implements `selected_rank = 1` forcefully; it ignores Top-K ranking completely.
2. **Sub-Pixel Classical Localization (Lines 416-431):** The README claims a local correlation patch is used for peak interpolation before SNRN. In reality, the production pipeline passes the integer center of the GSPE bounding box directly to SNRN and bypasses `ClassicalLocalization` entirely.
3. **Architecture Diagram (Lines 281-345):** Displays "Sub-Pixel Peak Localization" and "Classical Coordinate Estimate" as independent blocks preceding SNRN. In production, these are replaced by a simple box-center integer coordinate calculation.

## 6. ANTI-GT-LEAKAGE AND INTERFACE REQUIREMENTS
- **`inference.py`**: Properly processes the required `<reference> <search>` CLI arguments. Outputs strictly `(x, y)` to stdout. Contains no GT leakage.
- **`verify_submission.py`**: Properly executes a fresh-machine test, verifies dependencies, and explicitly scans `inference.py` and `pipeline_backup_v2_ai.py` for `ground_truth.json` or `metadata.json` strings.
- **`verify_phase2.py`**: Ensures the underlying AI infrastructure is functional.
- **Conclusion:** The repository completely satisfies the Phase 2 inference interface and rigorously defends against offline metadata leakage.

## 7. LOCAL VERIFICATION COMMANDS
To verify the system locally, execute the following commands in order from the repository root:

- **Repository Verification (AI Components):**
  `python scripts/verify_phase2.py`

- **Submission Smoke Test (Fresh-Machine Sim):**
  `python scripts/verify_submission.py`

- **Official Phase 2 Evaluation (Actual Production Pipeline):**
  `python scripts/evaluate_phase64.py`

*(Note: `evaluate_45case.py` should NOT be used as the primary evaluator unless its import is updated to point to the actual production pipeline).*

## 8. FINAL CLASSIFICATION
**READY_WITH_DOCUMENTATION_FIXES**

No algorithmic modifications are necessary. The system mathematically executes the Phase 2 specification. The primary pending action before absolute repo freeze is aligning the `README.md` and `evaluate_45case.py` imports to match the true production architecture established in this audit.
