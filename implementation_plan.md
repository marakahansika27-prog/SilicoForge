# GitHub Submission Implementation Plan

## Overview
This plan details the steps to prepare the Drift-Sense V2 repository for a clean, reproducible public submission, addressing all requirements from Phase 104 and the final submission audit.

## 1. Repository Restructuring
- Rename or keep `src/integration/pipeline_backup_v2_ai.py` as the main pipeline entry point in `inference.py`. Since `src/integration/pipeline.py` was proven to be GSPE-only, the historically verified AI path is in `pipeline_backup_v2_ai.py`. We will use this in the final `inference.py`.
- Create `models/` directory and copy `outputs/checkpoints/best_model.pth` to `models/best_model.pth`.
- Modify `pipeline_backup_v2_ai.py` (or create a submission wrapper) to load the model from `models/best_model.pth` as well as checking `outputs/checkpoints/best_model.pth` as a fallback to not break backward compatibility. Actually, I can just use a `models/best_model.pth` path by passing it via environment variable, or modifying the fallback loading logic in `pipeline_backup_v2_ai.py` to check `models/` first.

## 2. Standalone inference.py
Create root-level `inference.py`:
- `python inference.py <reference_image> <search_image>`
- Initializes `HybridNavigationPipeline` from `src.integration.pipeline_backup_v2_ai`.
- Loads images via OpenCV.
- Disables all ground-truth logic.
- Suppresses unnecessary internal prints (unless `--verbose` is provided).
- Prints exactly `(x, y)` as the final output.

## 3. Standalone generate_dataset.py
Create root-level `generate_dataset.py`:
- Uses `argparse` for `--architecture` (DRAM/FinFET), `--num-pairs`, and `--output-dir`.
- Uses `dataset.generator.HackathonDatasetGenerator` to generate realistic pairs.
- Saves `reference.png`, `search.png`, and `ground_truth.json`.

## 4. Documentation
- Create `docs/references.md`: Include all citations (augmentation, noise, localization, DL methodology).
- Create `docs/pipeline.md`: Detail ICE, GSPE, AI refinement, and decision fusion.
- Create `README.md`: Professional quick-start guide, dataset generation instructions, inference instructions, and environment setup.

## 5. End-to-End Test and Verification
- Run a fresh environment test (simulated by verifying our scripts work end-to-end without assuming prior benchmark structures).
- Create `scripts/verify_submission.py` to automatically check for README, requirements.txt, inference script, dataset generator, model artifact, references, and no GT leakage.

## 6. GitHub Cleanup
- Create `.gitignore` to exclude `.venv`, `__pycache__`, `dataset/` (large), `outputs/` (large), and unnecessary `.pth` (except `models/best_model.pth`).
- Archive experimental `phaseXX` scripts into an `archive/` or leave them in `scripts/` but out of the way (the prompt says: "Do NOT put hundreds of experimental scripts into the main public repository... Separate PRODUCTION from RESEARCH/EXPERIMENTAL"). I will move all `phase*` scripts to `scripts/experimental/`.

## User Review Required
Please review this plan. Key decision: I will move all `phase*.py` scripts into `scripts/experimental/` to keep the public repo clean. I will update `pipeline_backup_v2_ai.py` to also check `models/best_model.pth` when loading the checkpoint.
