# SilicoForge

### Hybrid Sub-Pixel Localization for Semiconductor Pattern Images

SilicoForge is a hybrid computer-vision and deep-learning system for locating a semiconductor reference pattern inside a larger search image.

The system combines deterministic image conditioning, correlation-based global search, multi-hypothesis candidate generation, high-resolution verification, sub-pixel localization, and learned local refinement.

SilicoForge was developed through two stages:

- **Phase 1:** constrained-pose semiconductor pattern localization.
- **Phase 2:** registration under unknown pose, including scale, rotation, degraded images, absent targets, rejection, confidence, and CPU-only inference.

---

## Table of Contents

- [Overview](#overview)
- [Problem](#problem)
- [Solution](#solution)
- [Phase 1](#phase-1)
- [Phase 2](#phase-2)
- [System Architecture](#system-architecture)
- [Pipeline](#pipeline)
  - [Image Conditioning](#image-conditioning)
  - [Global Search](#global-search)
  - [Candidate Generation](#candidate-generation)
  - [Candidate Verification](#candidate-verification)
  - [Sub-Pixel Localization](#sub-pixel-localization)
  - [Learned Refinement](#learned-refinement)
  - [Decision and Rejection](#decision-and-rejection)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Pose Convention](#pose-convention)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Runtime](#runtime)
- [Model](#model)
- [RGB Handling](#rgb-handling)
- [Failure Analysis](#failure-analysis)
- [Known Limitations](#known-limitations)
- [Repository Structure](#repository-structure)
- [Important Files](#important-files)
- [Reproducibility](#reproducibility)
- [Submission Package](#submission-package)
- [Team](#team)
- [License](#license)

---

## Overview

The goal of SilicoForge is to find a small semiconductor reference pattern inside a larger search image and return its location and pose.

The challenge is that semiconductor layouts frequently contain repeated structures. Several regions can therefore look similar to the reference and produce strong matching responses.

SilicoForge addresses this by separating **global discovery** from **local precision refinement**.

```text
Reference + Search
        |
        v
Image Conditioning (ICE)
        |
        v
Global Search (GSPE)
        |
        v
Scale / Rotation Hypotheses
        |
        v
Spatially Diverse Candidates
        |
        v
High-Resolution Verification
        |
        v
Sub-Pixel Localization
        |
        v
SNRN / Learned Refinement
        |
        v
Decision / Rejection
        |
        v
Final Prediction
```

The production system is designed for CPU execution and does not require network access during inference.

---

## Problem

Given a reference semiconductor image and a larger search image, the system must determine whether the reference pattern is present.

If the target is present, the system estimates:

```text
x
y
theta
scale
```

If the target is absent or rejected:

```text
found = 0
```

with zero pose fields.

Phase 2 extends the localization problem to unknown scale, rotation, degraded inputs, and absent targets.

---

## Solution

SilicoForge uses a hybrid architecture rather than relying on a single neural network or a single correlation peak.

The main stages are:

1. **ICE** conditions the images.
2. **GSPE** performs broad correlation-based search.
3. Multiple scale and rotation hypotheses are evaluated.
4. Spatially distinct candidates are retained.
5. Candidates are verified at higher resolution.
6. The selected response is refined to sub-pixel precision.
7. **SNRN** provides a learned local residual correction.
8. Decision fusion produces the final prediction and rejection decision.

The central design principle is:

> Use classical search for global discovery and learned inference for local refinement.

---

# Phase 1

Phase 1 established the core localization pipeline under constrained pose.

```text
Reference
    |
    v
ICE
    |
    v
GSPE
    |
    v
Candidate Localization
    |
    v
Sub-Pixel Refinement
    |
    v
SNRN Residual
    |
    v
Decision Fusion
    |
    v
(x, y)
```

Phase 1 focused primarily on accurate spatial localization and forms the foundation of the Phase 2 registration pipeline.

---

# Phase 2

## Registration Under Unknown Pose

Phase 2 extends the system to unknown pose.

The production system handles:

- scale variation,
- rotation variation,
- degraded images,
- absent targets,
- localization,
- pose estimation,
- confidence,
- rejection,
- CPU-only inference.

Nominal disclosed search range:

```text
Scale:      8 to 12
Rotation:  -5 to +5 degrees
```

The evaluator-facing output contains:

```text
x
y
theta
scale
found
score
```

The official organizer dataset and scoring rules are authoritative for final evaluation.

---

# System Architecture

```text
Reference Image
      |
      v
     ICE
      |
      v
     GSPE
      |
      v
Scale / Rotation Search
      |
      v
Spatial Candidate Pool
      |
      v
High-Resolution NCC
      |
      v
Sub-Pixel Refinement
      |
      v
SNRN / Learned Residual
      |
      v
Decision Fusion
     / \
    /   \
 FOUND  REJECT
```

---

# Pipeline

## Image Conditioning

Reference and search images are passed through the Image Conditioning Engine (ICE).

ICE performs deterministic preprocessing before global matching.

Implementation:

```text
src/preprocessing/ice.py
```

## Global Search

The Global Search Proposal Engine (GSPE) performs broad correlation-based search.

Phase 2 searches across multiple scale and rotation hypotheses instead of assuming one fixed geometry.

Implementation:

```text
src/coarse_search/gspe.py
```

## Candidate Generation

Repeated semiconductor structures can produce multiple strong peaks.

GSPE therefore retains multiple spatially distinct candidates.

```text
Search Response
      |
      v
Strong Peaks
      |
      v
Spatial Diversity / NMS
      |
      v
Candidate Pool
```

## Candidate Verification

The candidate pool is evaluated at higher resolution.

The production path uses full-resolution normalized cross-correlation (NCC) during candidate verification.

## Sub-Pixel Localization

The selected correlation response is locally refined:

```text
Integer Peak
     |
     v
Local Response Neighborhood
     |
     v
Sub-Pixel Refinement
     |
     v
Fractional (x, y)
```

## Learned Refinement

SNRN predicts a small local residual:

```text
(dx, dy)
```

The learned component is used for local refinement rather than global target discovery.

## Decision and Rejection

The production decision uses:

```text
score >= 0.85  -> found = 1
score <  0.85  -> found = 0
```

For a rejected target:

```text
x     = 0
y     = 0
theta = 0
scale = 0
```

---

# Installation

## Requirements

Validated development environment:

```text
Python 3.11
CPU
8 GB RAM target environment
No GPU required
No network required during inference
```

## Windows PowerShell

```powershell
git clone -b phase2-development https://github.com/marakahansika27-prog/SilicoForge.git SilicoForge-Phase2
cd SilicoForge-Phase2
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
python -m pip check
```

Expected:

```text
No broken requirements found.
```

## Linux / macOS

```bash
git clone -b phase2-development https://github.com/marakahansika27-prog/SilicoForge.git SilicoForge-Phase2
cd SilicoForge-Phase2
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

---

# Quick Start

Check the production interface:

```bash
python register.py --help
```

Run registration:

```bash
python register.py --input pairs.csv --output predictions.csv
```

The trained model is expected at:

```text
models/best_model.pth
```

The output file is:

```text
predictions.csv
```

with header:

```text
pair_id,x,y,theta,scale,found,score
```

---

# Input Format

The production program accepts:

```text
--input pairs.csv
```

The evaluator supplies pair identifiers and image paths.

A simplified development example is:

```csv
pair_id,reference,search
example_001,path/to/reference.png,path/to/search.png
example_002,path/to/reference2.png,path/to/search2.png
```

The exact organizer-provided input schema takes precedence over this simplified example.

The supplied `pair_id` must be preserved exactly.

Every input pair must produce exactly one output row.

---

# Output Format

Required output header:

```csv
pair_id,x,y,theta,scale,found,score
```

Accepted target:

```csv
example_001,652.07,161.02,0.0,10.0,1,0.94
```

Rejected target:

```csv
example_002,0,0,0,0,0,0.42
```

## Output Rules

For every input pair:

- exactly one output row,
- preserve `pair_id`,
- `x` and `y` are search-image coordinates,
- `theta` is reported in degrees,
- `scale` is the recovered scale factor,
- `found=1` means an accepted match,
- `found=0` means rejected or absent.

When `found=0`, all pose fields must be zero.

---

# Pose Convention

## Coordinates

`x` and `y` represent the center of the located reference pattern in the wide-search image.

## Rotation

`theta` represents the reference-pattern rotation in the wide-search image.

Positive rotation is counter-clockwise.

## Scale

`scale` represents the recovered down-scaling factor.

Nominal Phase 2 range:

```text
8 to 12
```

---

# Dataset

The organizer-defined Phase 2 composition is:

| Set | Description | Count |
|---|---|---:|
| Set A | Nominal present | 70 |
| Set B | Degraded present | 70 |
| Set C | Absent | 40 |
| Set D | RGB optical bonus | 20 |
| **Total** | | **200** |

Sets A, B, and C form the core evaluation.

Set D is the optical/RGB bonus group.

## Development Dataset

The local V4 engineering benchmark contained:

```text
200 total
100 present
100 absent

50 DRAM present
50 DRAM absent

50 FinFET present
50 FinFET absent
```

This local benchmark is not the official Phase 2 A/B/C/D dataset and should not be presented as official leaderboard data.

---

# Evaluation

The Phase 2 base score is:

| Dimension | Points |
|---|---:|
| Localization | 40 |
| Pose | 20 |
| Rejection | 15 |
| Confidence | 10 |
| Efficiency | 5 |
| Documentation | 10 |
| **Base Total** | **100** |

Additional bonus points may be available under organizer-defined conditions.

The official evaluator, dataset, and scoring rules remain authoritative.

## Local Engineering Results

On the local V4 benchmark, the production rejection decision produced:

```text
TP = 98
FN = 2
FP = 0
TN = 100
```

Rejection metrics:

```text
Precision = 1.0000
Recall    = 0.9800
F1        = 0.9899
```

For accepted present cases:

```text
Mean error   = 114.5358 px
Median error = 58.4594 px
Maximum      = 399.935 px
```

Localization thresholds:

```text
Within 1 px  = 35.71%
Within 2 px  = 44.90%
Within 3 px  = 45.92%
Within 5 px  = 46.94%
Within 10 px = 46.94%
Within 50 px = 48.98%
```

These are local engineering measurements only, not official Phase 2 leaderboard results.

---

# Runtime

Target evaluation environment:

```text
4-core x86 CPU
8 GB RAM
Python 3.11
No GPU
No network
```

Organizer runtime requirement:

```text
Median <= 5 seconds per pair
Hard timeout = 20 seconds per pair
```

## Local Measurement

A local Python 3.11 CPU run over 200 engineering cases completed in:

```text
278.0669 seconds total
```

Average:

```text
~1.39 seconds per pair
```

This is a local measurement and is not the official evaluator median.

---

# Model

The production model is:

```text
models/best_model.pth
```

The model is loaded locally during inference.

Inference does not require downloading model weights from the network.

The learned model is used for local residual refinement rather than global target discovery.

---

# RGB Handling

The current production image-loading path uses grayscale images.

RGB input therefore follows:

```text
RGB Image
    |
    v
Grayscale Conversion
    |
    v
Normal Production Pipeline
```

The system can ingest RGB image files, but color information is not currently exploited.

Therefore:

- RGB files are ingestible.
- Color channels are discarded.
- No color-specific feature is claimed.
- No official Set D bonus result is claimed.

Set D performance should only be claimed after validation on the official optical dataset.

---

# Failure Analysis

## Periodic Aliasing

Repeated semiconductor structures can generate several strong correlation peaks.

Therefore:

```text
Highest Score != Always Correct Location
```

## Top-1 Selection

The correct candidate can exist in the retained candidate pool while an incorrect repeated structure ranks first.

This is a primary observed localization failure mode.

## Candidate Generation

If the correct target is not retained in the candidate pool, downstream verification and refinement cannot recover it.

## Rejection False Negatives

A present target can occasionally fall below the fixed rejection threshold.

## Learned Refinement Limitation

SNRN performs local correction and cannot reliably repair a globally incorrect candidate.

### Engineering Conclusion

The major remaining localization weakness is candidate selection in repetitive structures.

The production path retains multiple spatial candidates and performs higher-resolution verification before final output.

No unvalidated experimental reranking rule is enabled in the frozen production path.

---

# Known Limitations

### Periodic Structures

Repeated semiconductor patterns can produce strong false correlation peaks.

### Candidate Selection

The strongest matching candidate is not guaranteed to be the required target.

### Candidate Pool

A target missing from the retained candidate set cannot be recovered by later refinement.

### Grayscale Processing

The production path does not exploit RGB color information.

### Set D

No official Set D optical bonus result is claimed.

### Rejection Threshold

The `0.85` threshold is an engineering choice based on available local validation and is not claimed to be globally optimal for the blind evaluation dataset.

### Development Benchmark

The local V4 benchmark is separate from the official Phase 2 dataset.

---

# Repository Structure

```text
SilicoForge/
|
+-- register.py
+-- requirements.txt
+-- generate_dataset.py
+-- failure_analysis.pdf
+-- README.md
|
+-- models/
|   +-- best_model.pth
|
+-- src/
    |
    +-- preprocessing/
    |   +-- ice.py
    |
    +-- coarse_search/
    |   +-- gspe.py
    |
    +-- ai_refinement/
    |   +-- network.py
    |   +-- inference.py
    |   +-- dataset.py
    |   +-- augmentations.py
    |   +-- loss.py
    |   +-- trainer.py
    |
    +-- integration/
    |   +-- pipeline_backup_v2_ai.py
    |
    +-- utils/
```

---

# Important Files

| File | Purpose |
|---|---|
| `register.py` | Evaluator-facing production entry point |
| `requirements.txt` | Pinned Python dependencies |
| `generate_dataset.py` | Dataset generation |
| `models/best_model.pth` | Production trained model |
| `src/preprocessing/ice.py` | Image conditioning |
| `src/coarse_search/gspe.py` | Global candidate generation |
| `src/ai_refinement/` | Learned local refinement |
| `src/integration/pipeline_backup_v2_ai.py` | Integrated registration pipeline |
| `failure_analysis.pdf` | Failure analysis and engineering findings |
| `README.md` | Project documentation |

---

# Reproducibility

## Clone

```bash
git clone -b phase2-development https://github.com/marakahansika27-prog/SilicoForge.git SilicoForge-Phase2
cd SilicoForge-Phase2
```

## Create Environment

Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Verify

```bash
python -m pip check
```

## Run

```bash
python register.py --input pairs.csv --output predictions.csv
```

The model is stored in the repository, so inference does not require downloading model weights from the network.

---

# Submission Package

The Phase 2 submission package contains the evaluator-facing artifacts:

```text
register.py
requirements.txt
generate_dataset.py
failure_analysis.pdf
best_model.pth
```

Before submission, verify:

- all required files are present,
- model weights are included,
- dependencies install successfully,
- `python register.py --help` works,
- the output header is correct,
- every input pair produces exactly one output row,
- rejected rows contain zero pose fields,
- development-only datasets and diagnostics are excluded from the submission package.

---

# Team

**SilicoForge**

B V Raju Institute of Technology, Narsapur

---

# License

No separate open-source license has been declared for this project.

Unless a license file is added, repository contents should be treated as project/submission source code.

---

## Production Command

```bash
python register.py --input pairs.csv --output predictions.csv
```

Expected output header:

```text
pair_id,x,y,theta,scale,found,score
```

The evaluator-supplied dataset and official scoring rules remain authoritative.

'@


Write-Host "README.md written successfully:"
Write-Host $path
Write-Host ""
Write-Host "First lines:"
Get-Content $path -TotalCount 10
Write-Host ""
Write-Host "Markdown headings found:"
Select-String -Path $path -Pattern '^# |^## ' | Select-Object -First 40
