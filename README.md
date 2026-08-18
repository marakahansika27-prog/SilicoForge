# SilicoForge

## Hybrid Sub-Pixel Localization for Semiconductor Pattern Images

SilicoForge is a hybrid computer-vision and deep-learning framework for precise localization of semiconductor layout patterns in high-resolution search images.

The framework combines classical image processing, multi-hypothesis global search, correlation-based candidate localization, sub-pixel estimation, learned residual refinement, and confidence-aware decision fusion.

It is designed for repetitive semiconductor structures such as DRAM and FinFET patterns, where conventional template matching can be affected by scale variation, rotation, noise, blur, repeated structures, local ambiguity, and sub-pixel displacement.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Motivation](#motivation)
- [Objectives](#objectives)
- [Scope](#scope)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Complete Pipeline Workflow](#complete-pipeline-workflow)
- [Core Components](#core-components)
  - [Image Conditioning Engine](#1-image-conditioning-engine-ice)
  - [Global Search Proposal Engine](#2-global-search-proposal-engine-gspe)
  - [Candidate Generation and Ranking](#3-candidate-generation-and-ranking)
  - [Sub-Pixel Localization](#4-sub-pixel-localization)
  - [Subpixel Navigation Refinement Network](#5-subpixel-navigation-refinement-network-snrn)
  - [Confidence Estimation](#6-confidence-estimation)
  - [Decision Fusion](#7-confidence-aware-decision-fusion)
- [Data Flow](#data-flow)
- [Mathematical Formulation](#mathematical-formulation)
- [Dataset Generation](#dataset-generation)
- [Training Pipeline](#training-pipeline)
- [Inference Pipeline](#inference-pipeline)
- [Input Specification](#input-specification)
- [Output Specification](#output-specification)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Inference](#running-inference)
- [Training](#training)
- [Evaluation and Verification](#evaluation-and-verification)
- [Ground-Truth Leakage Prevention](#ground-truth-leakage-prevention)
- [Reproducibility](#reproducibility)
- [Technical Design Decisions](#technical-design-decisions)
- [Why a Hybrid Architecture?](#why-a-hybrid-architecture)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Technology Stack](#technology-stack)
- [Project Status](#project-status)
- [Minimal Evaluator Workflow](#minimal-evaluator-workflow)
- [References](#references)
- [License](#license)

---

## Overview

Modern semiconductor layouts contain highly repetitive and geometrically complex structures.

A localization system must determine where a known reference pattern occurs inside a larger search image.

The challenge is not simply detecting whether the pattern exists. The system must estimate its position accurately enough to support sub-pixel localization.

SilicoForge therefore separates the problem into multiple stages:

```text
Reference Image
      |
      v
Image Conditioning
      |
      v
Global Search
      |
      v
Candidate Generation
      |
      v
Candidate Ranking
      |
      v
Sub-Pixel Localization
      |
      +----------------------+
      |                      |
      v                      v
Classical Estimate      SNRN Refinement
      |                      |
      |                    (dx,dy)
      |                      |
      +----------+-----------+
                 |
                 v
       Confidence Estimation
                 |
                 v
        Decision Fusion
                 |
                 v
          Final (x,y)
```

The final output is a two-dimensional coordinate `(x, y)` representing the predicted center of the reference pattern in the search image.

---

## Problem Statement

Given:

1. A reference image containing a semiconductor pattern.
2. A larger search image containing the same or corresponding pattern at an unknown location.

The objective is:

> To accurately estimate the center coordinate of the reference pattern inside the search image, including sub-pixel displacement.

The localization problem becomes difficult because semiconductor imagery may contain:

- Repetitive structures
- Similar neighboring patterns
- Scale variation
- Rotation
- Noise
- Blur
- Illumination changes
- Geometric variation
- Local correlation ambiguity
- Sub-pixel displacement

A naive single-template matching approach may therefore select a visually similar but incorrect periodic location.

SilicoForge addresses the problem using a layered localization strategy instead of depending on a single matching operation.

---

## Motivation

Semiconductor layouts are highly structured and frequently contain repeated patterns. This creates an important localization challenge:

```text
Correct Pattern
      |
      |        Similar Pattern
      |              |
      v              v
+-----------+  +-----------+
| TARGET    |  | LOOKALIKE |
+-----------+  +-----------+
```

A conventional maximum-correlation approach may identify a strong local match without necessarily identifying the correct structural instance.

Therefore, the system must consider:

- Global search
- Multiple hypotheses
- Spatially distinct candidates
- Correlation quality
- Local peak structure
- Sub-pixel displacement
- Confidence
- Reliability of learned refinement

---

## Objectives

### Objective 1 — Robust Global Localization

Identify the most probable region containing the reference pattern inside the search image. The system searches across multiple spatial and geometric hypotheses rather than assuming a fixed location.

### Objective 2 — Sub-Pixel Localization

Estimate the target position beyond integer-pixel precision. The local correlation surface around the best candidate is used to estimate fractional displacement.

```text
Integer Peak
     |
     v
Local Neighborhood
     |
     v
Sub-Pixel Interpolation
     |
     v
(x + dx, y + dy)
```

### Objective 3 — Learned Residual Refinement

Use a neural network to learn small residual corrections to the classical localization estimate. The network predicts `(dx, dy)` and produces:

```
x_refined = x_classical + dx
y_refined = y_classical + dy
```

### Objective 4 — Confidence-Aware Decision Making

The neural network should not automatically override a strong classical estimate. The system evaluates confidence before accepting the learned refinement.

```text
Classical Estimate
       +
   AI Residual
       +
  Confidence
       |
       v
Decision Fusion
       |
       v
Final Coordinate
```

### Objective 5 — Reproducible Execution

The project provides:

- Dataset generation
- Model artifact
- Training implementation
- Inference interface
- Dependency specification
- Verification scripts
- Technical documentation

---

## Scope

The current implementation focuses on:

- Semiconductor pattern localization
- DRAM patterns
- FinFET patterns
- Synthetic evaluation data
- Global candidate search
- Correlation-based localization
- Sub-pixel estimation
- Learned residual refinement
- Confidence-aware fusion

The system is intended as a localization framework rather than a general-purpose semiconductor inspection platform.

---

## Key Features

- Hybrid classical + deep-learning architecture
- Synthetic semiconductor image generation
- DRAM pattern generation
- FinFET pattern generation
- Ground-truth center recording
- Deterministic image conditioning
- Multi-scale search
- Multi-rotation search
- Correlation-based matching
- Candidate peak extraction
- Spatial candidate separation
- Candidate ranking
- Sub-pixel peak interpolation
- Neural residual prediction
- Confidence estimation
- Confidence-aware decision fusion
- Standalone inference
- Automatic model loading
- Ground-truth leakage prevention
- Repository verification
- Reproducible execution

---

## System Architecture

```text
                         +----------------------+
                         |    Reference Image   |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Image Conditioning   |
                         |      Engine (ICE)    |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |     Search Image     |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Global Search        |
                         | Proposal Engine      |
                         |       (GSPE)         |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Candidate Generation |
                         | and Ranking          |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Sub-Pixel Peak       |
                         | Localization         |
                         +----------+-----------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
          +------------------+             +------------------+
          | Classical        |             | SNRN             |
          | Coordinate       |             | Neural           |
          | Estimate         |             | Refinement       |
          +--------+---------+             +--------+---------+
                   |                                |
                   |                              (dx,dy)
                   |                                |
                   +---------------+----------------+
                                   |
                                   v
                         +----------------------+
                         | Confidence           |
                         | Estimation            |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Confidence-Aware     |
                         | Decision Fusion      |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Final Sub-Pixel      |
                         | Coordinate (x,y)     |
                         +----------------------+
```

---

## Complete Pipeline Workflow

### Stage 1 — Input

The pipeline receives:

```
Reference Image
Search Image
```

### Stage 2 — Image Conditioning

The reference and search images are transformed into representations suitable for robust matching.

```text
Raw Images
    |
    v
Normalization
    |
    v
Filtering / Conditioning
    |
    v
Matching Representation
```

The purpose is to reduce irrelevant variation while preserving structural information.

### Stage 3 — Global Search

The Global Search Proposal Engine searches the larger image for possible locations of the reference pattern. The search considers:

```
Spatial Location
Scale
Rotation
```

rather than relying on a single fixed template.

### Stage 4 — Candidate Generation

Instead of immediately accepting the global maximum, multiple strong candidate peaks are retained. This is important for repetitive semiconductor structures.

```text
Correlation Surface

      Peak A
         *

              Peak B
                 *

     Peak C
        *
```

Candidate coordinates are preserved for further analysis.

### Stage 5 — Candidate Ranking

Candidates are ranked using matching quality and structural evidence. The strongest spatially meaningful candidate is selected for refinement.

### Stage 6 — Sub-Pixel Localization

The selected candidate is refined using its local correlation neighborhood.

```text
correlation
            ^
            |
            |          *
            |        *   *
            |      *       *
            |    *           *
            +----------------------> position
```

The local surface is used to estimate the fractional displacement from the integer coordinate.

### Stage 7 — Classical Coordinate

The classical stage produces `(x_classical, y_classical)`. This serves as the primary localization estimate.

### Stage 8 — Neural Refinement

The local candidate information is passed to SNRN. SNRN predicts:

```
dx
dy
confidence
```

### Stage 9 — Confidence Evaluation

The predicted residual is evaluated together with its confidence. A strong classical result may be retained when the neural prediction is uncertain.

### Stage 10 — Decision Fusion

The final decision combines:

```
Classical Estimate
+
Neural Residual
+
Confidence
```

### Stage 11 — Final Output

The final coordinate is returned as `(x, y)`. Both coordinates may contain fractional pixels.

---

## Core Components

### 1. Image Conditioning Engine (ICE)

**Purpose**

Prepare input images for robust localization.

**Responsibilities**

- Normalize image representation
- Reduce irrelevant variations
- Preserve semiconductor structural information
- Prepare consistent matching inputs

**Implementation**

`src/preprocessing/ice.py`

### 2. Global Search Proposal Engine (GSPE)

**Purpose**

Perform robust global localization.

**Responsibilities**

- Search across spatial positions
- Evaluate scale hypotheses
- Evaluate rotation hypotheses
- Generate matching responses
- Produce candidate peaks

**Implementation**

`src/coarse_search/gspe.py`

### 3. Candidate Generation and Ranking

The candidate stage prevents the pipeline from relying blindly on a single global maximum. Multiple strong and spatially distinct candidates can be retained.

```text
Search Image
     |
     v
Correlation Surface
     |
     v
Peak Detection
     |
     v
Spatial NMS
     |
     v
Top-K Candidates
     |
     v
Candidate Ranking
```

This is particularly important for periodic semiconductor structures.

### 4. Sub-Pixel Localization

The candidate location is initially estimated at integer-pixel resolution. A local neighborhood around the peak is then analyzed to estimate fractional displacement.

```text
(x_integer, y_integer)
          |
          v
Local Correlation Patch
          |
          v
Peak Interpolation
          |
          v
(x_subpixel, y_subpixel)
```

### 5. Subpixel Navigation Refinement Network (SNRN)

SNRN performs learned residual refinement.

**Input:** Local Candidate Representation

**Output:**

```
dx
dy
confidence
```

The final learned correction is:

```
x_refined = x_classical + dx
y_refined = y_classical + dy
```

**Implementation**

```
src/ai_refinement/network.py
src/ai_refinement/inference.py
```

### 6. Confidence Estimation

Confidence is used to determine whether the learned correction should be trusted.

```text
Strong Classical Match
        +
High AI Confidence
        |
        v
Accept Refinement
```

```text
Strong Classical Match
        +
Low AI Confidence
        |
        v
Retain Classical Estimate
```

### 7. Confidence-Aware Decision Fusion

The final decision is not based exclusively on either classical matching or neural prediction.

```text
Classical Estimate
                        |
                        v
                 +-------------+
                 |   Fusion    |
                 |    Engine   |
                 +------+------+
                        ^
                        |
                AI Residual
                        +
                   Confidence
                        |
                        v
                  Final (x,y)
```

**Implementation**

`src/integration/decision_fusion.py`

---

## Data Flow

```text
REFERENCE IMAGE
      |
      v
+----------------------+
| Image Conditioning   |
+----------+-----------+
           |
           v
+----------------------+
| Template / Reference |
| Representation       |
+----------+-----------+

SEARCH IMAGE
      |
      v
+----------------------+
| Image Conditioning   |
+----------+-----------+
           |
           v
+----------------------+
| Global Search        |
| Multi-Hypothesis     |
+----------+-----------+
           |
           v
+----------------------+
| Correlation Surface  |
+----------+-----------+
           |
           v
+----------------------+
| Candidate Generation |
+----------+-----------+
           |
           v
+----------------------+
| Candidate Ranking    |
+----------+-----------+
           |
           v
+----------------------+
| Sub-Pixel            |
| Localization         |
+----------+-----------+
           |
           +--------------------+
           |                    |
           v                    v
+------------------+   +------------------+
| Classical        |   | SNRN             |
| Estimate         |   | Residual         |
+--------+---------+   +--------+---------+
         |                      |
         |                    dx,dy
         |                      |
         +----------+-----------+
                    |
                    v
          +------------------+
          | Confidence       |
          | Evaluation       |
          +--------+---------+
                   |
                   v
          +------------------+
          | Decision Fusion  |
          +--------+---------+
                   |
                   v
             FINAL (x,y)
```

---

## Mathematical Formulation

Let:

- `R` = reference image
- `S` = search image
- `h` = geometric hypothesis
- `C(x,y,h)` = matching/correlation score
- `(x_c, y_c)` = selected classical coordinate
- `(dx,dy)` = neural residual

The global search can be expressed as:

```
(x*, y*, h*) = argmax C(x, y, h)
```

where the hypothesis space includes:

```
h = {scale, rotation, spatial position}
```

After candidate selection, sub-pixel localization estimates `(x_sp, y_sp)`.

The neural refinement predicts:

```
(dx, dy) = SNRN(local_candidate)
```

The refined coordinate is:

```
x_refined = x_sp + dx
y_refined = y_sp + dy
```

The decision-fusion stage determines whether the refined coordinate should replace or modify the classical estimate based on confidence.

---

## Dataset Generation

Synthetic data generation supports controlled evaluation of the pipeline without requiring proprietary fab imagery.

- DRAM-style repeating cell layouts and FinFET-style fin/gate layouts are generated procedurally.
- Each generated sample records the ground-truth reference-pattern center coordinate.
- Search images are produced by embedding a reference pattern into a larger canvas at a known, randomized offset.
- Configurable variation includes scale, rotation, additive noise, blur, and illumination change, so the dataset exercises the same failure modes described in [Motivation](#motivation).
- Generation scripts live under `src/data_generation/`, with generated assets and ground-truth records written to a dataset output directory (not committed to version control).

---

## Training Pipeline

- Synthetic samples are split into training and validation sets, with generation parameters logged alongside each split for reproducibility.
- The classical pipeline (ICE → GSPE → candidate ranking → sub-pixel estimate) is run first to produce the local candidate representation SNRN consumes as input.
- SNRN is trained to predict `(dx, dy)` residuals against the recorded ground-truth center, together with a confidence value calibrated against localization error.
- Training configuration, loss weighting, and checkpointing are managed under `src/ai_refinement/`.
- Trained weights are saved as a model artifact used by the inference pipeline.

---

## Inference Pipeline

- A reference image and a search image are provided as input.
- The classical stages (conditioning, global search, candidate generation/ranking, sub-pixel localization) execute deterministically.
- SNRN loads its trained artifact automatically and produces a residual and confidence for the top candidate.
- Decision fusion combines the classical estimate with the neural residual according to the confidence policy.
- The pipeline returns the final `(x, y)` coordinate, optionally alongside the intermediate classical estimate and confidence score for diagnostics.

---

## Input Specification

- **Reference image** — a cropped image of the target semiconductor pattern.
- **Search image** — a larger image expected to contain the reference pattern.
- Supported image formats and expected channel/bit-depth conventions are defined by the Image Conditioning Engine and should match the conventions used during dataset generation and training.

---

## Output Specification

- **Final coordinate** — `(x, y)`, representing the predicted sub-pixel center of the reference pattern within the search image.
- **Diagnostic outputs (optional)** — classical estimate, neural residual `(dx, dy)`, and confidence score, useful for evaluation and debugging.

---

## Repository Structure

```text
SilicoForge/
├── src/
│   ├── preprocessing/
│   │   └── ice.py
│   ├── coarse_search/
│   │   └── gspe.py
│   ├── ai_refinement/
│   │   ├── network.py
│   │   └── inference.py
│   ├── integration/
│   │   └── decision_fusion.py
│   └── data_generation/
├── models/
├── scripts/
├── tests/
└── README.md
```

---

## Requirements

- Python 3.x
- Standard scientific Python stack (array/numerical computing, image processing)
- Deep learning framework used to define and run SNRN
- See `requirements.txt` for the pinned dependency list

---

## Installation

```bash
<<<<<<< HEAD
git clone https://github.com/marakahansika27-prog/SilicoForge.git
=======
 git clone https://github.com/marakahansika27-prog/SilicoForge.git
>>>>>>> 24ab1d73042bf1bd0c69b37c5fe913531334fe63
cd SilicoForge
pip install -r requirements.txt
```

---

## Quick Start

```bash
python scripts/run_inference.py \
    --reference path/to/reference.png \
    --search path/to/search.png
```

---

## Running Inference

```bash
python scripts/run_inference.py \
    --reference <reference_image_path> \
    --search <search_image_path> \
    --output <output_path>
```

The script loads the trained SNRN artifact automatically and prints the final `(x, y)` coordinate along with diagnostic values.

---

## Training

```bash
python scripts/train_snrn.py \
    --dataset <dataset_path> \
    --epochs <num_epochs> \
    --output <checkpoint_path>
```

---

## Evaluation and Verification

- Evaluation scripts compare predicted coordinates against recorded ground-truth centers on held-out synthetic data.
- Repository verification scripts check that required artifacts, configuration, and directory structure are present before training or inference is run.

---

## Ground-Truth Leakage Prevention

- Ground-truth center coordinates are used only for dataset labeling and evaluation, never as an input feature to the classical pipeline or SNRN.
- Training and validation splits are generated from disjoint random seeds to prevent overlap between synthetic scenes.
- Evaluation scripts are kept separate from training code paths to avoid inadvertent use of ground-truth during inference.

---

## Reproducibility

- Dataset generation, training, and evaluation are seeded for deterministic reruns.
- Model artifacts, training configuration, and generation parameters are versioned alongside results.

---

## Technical Design Decisions

- Classical global search is retained rather than replaced by an end-to-end learned detector, to preserve deterministic, interpretable global reasoning for repetitive structures.
- SNRN is scoped to residual refinement rather than full coordinate regression, since the classical pipeline already narrows the search to a strong local neighborhood.
- Confidence-aware fusion is used instead of always trusting the neural output, to guard against learned refinement errors on inputs that fall outside the training distribution.

---

## Why a Hybrid Architecture?

A purely classical system provides strong deterministic global search but may have difficulty learning systematic residual errors.

A purely neural system may learn powerful representations but can become sensitive to training distribution, domain shift, and repetitive structures.

SilicoForge combines both:

```text
Classical CV
    |
    | Global spatial reasoning
    v
Strong Candidate
    |
    | Local geometric reasoning
    v
Sub-Pixel Estimate
    |
    | Learned residual
    v
SNRN
    |
    | Confidence
    v
Decision Fusion
    |
    v
Final Coordinate
```

This division of responsibility allows:

- Classical methods to perform global discovery.
- Correlation analysis to identify candidate locations.
- Sub-pixel estimation to provide a precise geometric baseline.
- SNRN to correct small, learnable systematic errors that classical methods alone cannot capture.
- Confidence-aware fusion to keep the final result reliable even when the learned component is uncertain.

---

## Limitations

- Performance is currently validated primarily on synthetic DRAM/FinFET-style imagery rather than production fab data.
- Extreme scale, rotation, or noise outside the ranges used during dataset generation and training may degrade localization accuracy.
- Highly periodic structures with near-identical neighboring patterns remain a fundamentally hard case for any correlation-based candidate search.
- SNRN's residual correction is only as reliable as its training distribution; confidence estimation mitigates but does not eliminate this risk.

---

## Future Improvements

- Extend evaluation to real semiconductor imagery beyond synthetic data.
- Expand the global search hypothesis space (e.g., additional geometric transformations).
- Explore joint or end-to-end training of the classical and learned stages.
- Improve confidence calibration with additional uncertainty-estimation techniques.
- Add batch/parallel inference support for large-scale wafer or layout scanning.

---

## Technology Stack

- Python
- Classical computer vision / image processing libraries
- Deep learning framework for SNRN
- Synthetic data generation utilities

---

## Project Status

Active development. Core pipeline (conditioning, global search, candidate ranking, sub-pixel localization, SNRN refinement, confidence-aware fusion) is implemented; ongoing work focuses on evaluation against broader data and further refinement of the fusion policy.

---

## Minimal Evaluator Workflow

```bash
# 1. Generate or obtain a reference/search image pair with known ground truth
python scripts/generate_dataset.py --output data/eval

# 2. Run inference
python scripts/run_inference.py --reference data/eval/ref.png --search data/eval/search.png

# 3. Compare predicted (x, y) against the recorded ground-truth center
python scripts/evaluate.py --predictions <predictions_path> --ground-truth data/eval/ground_truth.json
```

---

## References

- Classical template matching and normalized cross-correlation methods.
- Sub-pixel peak interpolation techniques for correlation surfaces.
- Residual learning approaches for refining classical estimators.

---

## License

See `LICENSE` for license terms.
