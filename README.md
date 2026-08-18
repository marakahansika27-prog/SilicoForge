# Drift-Sense V2

Drift-Sense V2 is a hybrid navigation pipeline for sub-pixel localization of critical features (such as DRAM arrays and FinFET layouts) in semiconductor imagery. It combines classical Image Conditioning Engines (ICE) and Global Search Proposal Engines (GSPE) with an AI-driven Subpixel Navigation Refinement Network (SNRN) and a confidence-aware Decision Fusion module to achieve ultra-precise and robust localization.

## System Architecture

The pipeline consists of:
1. **ICE (Image Conditioning Engine):** Suppresses noise and structural variability, generating condition tensors.
2. **GSPE (Global Search Proposal Engine):** Uses template matching across scale/rotation to find high-probability regions.
3. **Sub-Pixel Localization:** Extracts classical affine-warped coordinates.
4. **SNRN (Subpixel Navigation Refinement Network):** A neural refinement module that predicts high-precision sub-pixel residuals.
5. **Decision Fusion:** Dynamically gates AI refinements based on network confidence and structural consistency to prevent catastrophic outliers.

## Repository Structure

```
Drift-Sense-V2/
├── README.md               # Quick-start documentation
├── requirements.txt        # Verified project dependencies
├── .gitignore              # Pre-configured git exclusions
├── inference.py            # Standalone evaluator inference script
├── generate_dataset.py     # Standalone dataset pair generator
├── models/                 # Distributable AI models (e.g., best_model.pth)
├── src/                    # Full Drift-Sense source code
├── scripts/                # Utility and experimental phase scripts
└── docs/                   # Methodology, Pipeline, and References documentation
```

## Setup & Installation

**Python Version:** 3.10+ recommended.

To run the project on a fresh machine without modifying the source:

```bash
# 1. Clone the repository
git clone <REPOSITORY_URL>
cd Drift-Sense-V2

# 2. Create an isolated virtual environment
python -m venv .venv

# 3. Activate the environment (Windows)
.venv\Scripts\activate
# (For Linux/macOS: source .venv/bin/activate)

# 4. Install requirements
python -m pip install -r requirements.txt
```

## Dataset Generation

Drift-Sense provides a powerful standalone data generator that produces synthetic pairs complete with subpixel drift, photometric noise, and bounding boxes.

### DRAM Generation Example
Generate 1 synthetic DRAM image pair:
```bash
python generate_dataset.py --architecture DRAM --num-pairs 1 --output-dir examples
```
### FinFET Generation Example
Generate 1 synthetic FinFET image pair:
```bash
python generate_dataset.py --architecture FinFET --num-pairs 1 --output-dir examples
```

*Note: The generator outputs `reference.png`, `search.png`, and a `ground_truth.json` file in each pair folder. The ground truth contains the TRUE center coordinate.*

## Standalone Inference

Use the standalone `inference.py` script to run the validated hybrid localization pipeline. The script automatically loads the required model weights (from `models/best_model.pth` or `outputs/checkpoints/best_model.pth`) and returns the final `(x, y)` subpixel coordinates.

**Command:**
```bash
python inference.py examples/dram/pair_0001/reference.png examples/dram/pair_0001/search.png
```

**Expected Output:**
```
(771.8257, 208.2563)
```

*(You can add `--verbose` for detailed step-by-step diagnostic prints from the pipeline).*

## Model-Weight Handling

The AI refinement model expects a PyTorch `best_model.pth` artifact. For the public release, the artifact should reside in `models/best_model.pth`. The pipeline handles loading transparently; if it fails to find the weights, it falls back to a random initialization (which will be caught by the decision fusion gating).

## Training Instructions

To reproduce the AI training process, execute the included training script:
```bash
python scripts/train.py
```
This script handles dataset preparation, model initialization, optimization, checkpointing, and exports the final trained weights.

## Benchmark / Evaluation

To run comprehensive evaluations on generated datasets, refer to the suite of benchmark tools located in `scripts/`. Note that `inference.py` strictly refuses to read from `ground_truth.json` to prevent label leakage during runtime execution.

## Troubleshooting
- **Missing Checkpoint:** Ensure the model weights are successfully downloaded to `models/best_model.pth`.
- **Environment Conflicts:** Always use a fresh virtual environment. If OpenCV has missing DLLs on Windows, install the `opencv-python-headless` variant or ensure Media Foundation is installed.

## Reproducibility Notes
This codebase has been strictly hardened to guarantee evaluator reproducibility. Modifying the GSPE thresholding or AI network weights manually may invalidate historical benchmark records.
