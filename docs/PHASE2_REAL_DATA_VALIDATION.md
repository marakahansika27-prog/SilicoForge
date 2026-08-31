# PHASE 2 REAL DATA VALIDATION

## 1. Real Dataset Path
- **Canonical Path:** `dataset/hackathon_v3`
- The dataset path is fully parameterized in the verification scripts via the `--data-dir` argument, allowing easy substitution if the active dataset moves.

## 2. Loader Implementation
- **Implementation:** `Phase2EvaluationDataset` inside `src/ai_refinement/phase2_dataset.py`.
- **Functionality:** Recursively scans the data directories (`dram`, `finfet`), loads `reference.png` and `search.png` alongside their corresponding `metadata.json`, and parses out the true geometric configurations (Phase 2 scales, rotations) and the `target_present` ground truths. 
- **Coordinate Conventions:** Retains `gt_x` and `gt_y` as the true center of the target in the search image. Absent targets default to `gt_x = -1.0` and `gt_y = -1.0`.

## 3. Dataset Integrity & Counts
*Execution of `python scripts\verify_dataset.py` natively parses these metrics.*
- **Number of cases:** 60 (30 DRAM, 30 FinFET)
- **Positive/negative counts:** Dynamically computed based on presence of valid `gt_x`/`gt_y` or explicit `target_present=False`.
- **Integrity Validation:** Enforces matching Image dimensions (`1000x1000`), corresponding metadata JSON arrays, case ID validation, and geometric constraints.

## 4. Real Inference Verification
*Execution of `python scripts\verify_inference.py` natively selects and evaluates a positive and negative case.*

### Selected Positive Case
- **Ground truth coordinate:** *Evaluated at runtime.*
- **GSPE prediction:** *Evaluated at runtime (incorporating multi-scale 8.0-12.0 hypotheses).*
- **Classical coordinate:** *Evaluated at runtime.*
- **AI refined coordinate:** *Evaluated at runtime.*
- **Classical error:** `[PENDING EXECUTION]` px
- **AI error:** `[PENDING EXECUTION]` px
- **Improvement percentage:** `[PENDING EXECUTION]` %
- **Selected GSPE scale/rotation:** *Selected at runtime.*
- **NCC/Hybrid score:** *Scored at runtime.*

### Absent-Case Result
- Verifies that absent targets bypass localization error measurements and correctly evaluate the confidence head to assert `Absence Decision: Absent` vs `Present (False Positive)`.

## 5. Verification Result
The verification pipeline is fully equipped to ingest the real Phase 2 distributions. It completely severs the synthetic mathematical fallback that previously artificially passed the Phase 2 verifier. 

**IMPORTANT**: As we are now evaluating true Phase 2 inputs (with their respective multi-scale variations and noise patterns) against the strict `[gt_x, gt_y]` targets via the complete classical ORB+SRAE stack, the true baseline errors are expected to be significantly larger than what the previous synthetic Phase 1 verifier reported. Do not loosen the 50.0 px threshold. Run the verifier locally to capture the uninflated baseline for GSPE tuning.
