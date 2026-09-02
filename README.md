SilicoForge

Hybrid Sub-Pixel Localization for Semiconductor Pattern Images

SilicoForge is a hybrid computer-vision and deep-learning system for locating a semiconductor reference pattern inside a larger search image.

The system combines deterministic image conditioning, correlation-based global search, candidate verification, sub-pixel localization, and learned local refinement.

Table of Contents

Overview

Phase 2 Task

Architecture

How It Works

Installation

Quick Start

Input Format

Output Format

Pose Convention

Dataset

Evaluation

Runtime

Rejection and Confidence

RGB Handling

Failure Analysis

Known Limitations

Repository Structure

Important Files

Reproducibility

Team

License

Overview

The problem is to locate a reference semiconductor pattern inside a larger search image.

Semiconductor layouts often contain repeated structures. As a result, several locations can produce strong correlation responses, so simply selecting the global maximum is not always reliable.

SilicoForge uses a staged approach:

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

The architecture separates global search from local refinement so that the learned model does not have to discover the target across the entire search image.

Phase 2 Task

Phase 2 extends the localization problem to Registration under Unknown Pose.

The production system handles:

unknown scale,

unknown rotation,

degraded images,

absent targets,

localization,

pose estimation,

confidence,

rejection,

CPU-only inference.

The nominal disclosed pose range is:

Scale:     8 to 12
Rotation: -5 to +5 degrees

The evaluator-facing command is:

python register.py --input pairs.csv --output predictions.csv

Architecture

The Phase 2 production path is:

Reference + Search
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
Spatial Candidate Set
        |
        v
Candidate Verification
        |
        v
Full-Resolution NCC
        |
        v
Sub-Pixel Refinement
        |
        v
SNRN / AI Refinement
        |
        v
Decision Fusion
       / \
      /   \
 FOUND   REJECT
   |       |
   +---+---+
       |
       v
(x, y, theta, scale, found, score)

Main Components

ICE — Image Conditioning Engine

Performs deterministic preprocessing before global matching.

src/preprocessing/ice.py

GSPE — Global Search Proposal Engine

Performs broad correlation-based search across pose hypotheses and retains spatially distinct candidates.

src/coarse_search/gspe.py

Sub-Pixel Refinement

Refines the selected correlation peak to obtain fractional-pixel coordinates.

SNRN — Subpixel Navigation Refinement Network

Predicts a small local coordinate correction:

(dx, dy)

The learned model is used for local refinement rather than global discovery.

Decision Fusion

Produces the final evaluator-facing localization and rejection decision.

How It Works

1. Image Conditioning

Reference and search images are passed through ICE.

The conditioning stage is separated from localization so that preprocessing and search can be evaluated independently.

2. Global Search

GSPE searches across multiple scale and rotation hypotheses.

The purpose is to obtain a broad set of plausible target locations.

3. Candidate Diversity

Repeated semiconductor structures can produce multiple strong peaks.

Therefore, the system retains multiple spatially distinct candidates rather than relying only on one global peak.

4. Candidate Verification

The retained candidates are evaluated at higher resolution.

This concentrates expensive processing on plausible locations instead of processing the entire search image at full resolution.

5. Sub-Pixel Localization

The selected correlation response is locally refined:

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

6. Learned Refinement

SNRN estimates a small residual correction:

Classical Coordinate
        |
        v
       SNRN
        |
        v
     (dx, dy)
        |
        v
Final Coordinate

7. Decision and Rejection

The production decision uses a matching threshold:

score >= 0.85  -> found = 1
score <  0.85  -> found = 0

Rejected targets receive zero pose fields.

Installation

SilicoForge Phase 2 is validated for:

Python 3.11

CPU execution

no GPU requirement

no network requirement during inference

Windows PowerShell

Clone the Phase 2 branch:

git clone -b phase2-development https://github.com/marakahansika27-prog/SilicoForge.git SilicoForge-Phase2
cd SilicoForge-Phase2

Create a Python 3.11 environment:

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

Verify Python:

python --version

Install dependencies:

python -m pip install -r requirements.txt

Verify dependencies:

python -m pip check

Expected:

No broken requirements found.

Linux / macOS

git clone -b phase2-development https://github.com/marakahansika27-prog/SilicoForge.git SilicoForge-Phase2
cd SilicoForge-Phase2
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check

Quick Start

Check the production interface:

python register.py --help

Run registration:

python register.py --input pairs.csv --output predictions.csv

The trained model is expected at:

models/best_model.pth

Expected output:

predictions.csv

with the header:

pair_id,x,y,theta,scale,found,score

Input Format

The production entry point accepts:

--input pairs.csv

The input file contains the evaluator-supplied pair identifiers and image paths.

A development example is:

pair_id,reference,search
example_001,path/to/reference.png,path/to/search.png

The exact organizer-provided input schema takes precedence over development examples.

The supplied pair_id must be preserved exactly.

Output Format

The production output header is:

pair_id,x,y,theta,scale,found,score

Example detected target:

example_001,652.07,161.02,0.0,10.0,1,0.94

Example rejected target:

example_002,0,0,0,0,0,0.42

Output Rules

For every input pair:

exactly one output row,

preserve pair_id,

output x and y in wide-search coordinates,

output theta in degrees,

output the recovered scale,

use found=1 for accepted matches,

use found=0 for rejected or absent targets.

When:

found = 0

the pose fields are:

x     = 0
y     = 0
theta = 0
scale = 0

The score may remain nonzero because it represents the matching evidence used by the decision stage.

Pose Convention

Coordinates

x and y represent the center of the located reference pattern in the wide-search image.

Rotation

theta represents the reference-pattern rotation in the wide-search image.

Positive rotation is:

Counter-clockwise

Scale

scale represents the recovered down-scaling factor.

Nominal Phase 2 range:

8 to 12

Dataset

The organizer-defined Phase 2 composition is:

Set

Description

Count

Set A

Nominal present

70

Set B

Degraded present

70

Set C

Absent

40

Set D

RGB optical bonus

20

Total



200

Sets A, B, and C form the core grayscale evaluation.

Set D is the optical/RGB bonus group.

Development Dataset

The local V4 engineering dataset used during development is a separate 200-case benchmark:

100 present
100 absent

50 DRAM present
50 DRAM absent

50 FinFET present
50 FinFET absent

It should not be treated as the official Phase 2 A/B/C/D dataset.

Evaluation

Phase 2 evaluates multiple dimensions:

Dimension

Points

Localization

40

Pose

20

Rejection

15

Confidence

10

Efficiency

5

Documentation

10

Base Total

100

Additional bonus points may be available under the organizer-defined conditions.

The official evaluator and official dataset remain authoritative.

Local Engineering Results

On the local V4 benchmark, the production rejection decision produced:

TP = 98
FN = 2
FP = 0
TN = 100

Resulting rejection metrics:

Precision = 1.0000
Recall    = 0.9800
F1        = 0.9899

For accepted present cases:

Mean error   = 114.5358 px
Median error = 58.4594 px
Maximum      = 399.935 px

Localization within thresholds:

Within 1 px  = 35.71%
Within 2 px  = 44.90%
Within 3 px  = 45.92%
Within 5 px  = 46.94%
Within 10 px = 46.94%
Within 50 px = 48.98%

These measurements are local engineering results only, not official Phase 2 leaderboard results.

Runtime

The target evaluator environment is:

CPU-only
Python 3.11
8 GB RAM
No GPU
No network during inference

The organizer runtime requirement is:

Median <= 5 seconds per pair
Hard timeout = 20 seconds per pair

Local Runtime

A local Python 3.11 CPU run over 200 engineering cases completed in:

278.0669 seconds total

Average:

~1.39 seconds per pair

This is a local measurement and should not be presented as the official evaluator median.

Rejection and Confidence

The production rejection threshold is:

GSPE_REJECTION_THRESHOLD = 0.85

Decision rule:

score >= 0.85  -> found = 1
score <  0.85  -> found = 0

The output score is intended to be monotonic with matching confidence.

It should not be interpreted as a calibrated probability.

For rejected targets:

found = 0
x = 0
y = 0
theta = 0
scale = 0

RGB Handling

The current production input path reads images in grayscale.

Therefore:

RGB Input
    |
    v
Grayscale Conversion
    |
    v
Normal Production Pipeline

RGB files can be ingested, but color information is not currently used.

Therefore:

RGB input is supported at the file-ingestion level,

color channels are discarded,

no color-specific feature is claimed,

no official Set D bonus result is claimed.

Set D performance requires validation on the official optical dataset.

Failure Analysis

Periodic Aliasing

Repeated semiconductor structures can generate multiple strong correlation peaks.

Therefore:

Highest Score != Always Correct Location

Top-1 Selection

The correct candidate can exist among the search candidates while an incorrect repeated structure ranks first.

Candidate Generation

If the correct location is not retained in the candidate pool, later verification cannot recover it.

Rejection False Negatives

A present target can occasionally fall below the fixed rejection threshold.

Learned Refinement

A local learned refinement model cannot correct a globally incorrect candidate.

Engineering Conclusion

The primary localization weakness is candidate selection in repetitive structures, rather than the absence of a global search mechanism.

The frozen production path does not enable an unvalidated reranking rule.

Known Limitations

Periodic Structures

Repeated layouts can produce strong false correlation peaks.

Candidate Selection

The strongest candidate is not guaranteed to be the required target.

Candidate Pool

A target missing from the retained candidate set cannot be recovered by later stages.

RGB Information

The current pipeline uses grayscale information and does not exploit color.

Set D

No official Set D optical bonus result is claimed.

Rejection Threshold

The 0.85 threshold is an engineering choice based on available local validation and is not claimed to be globally optimal for the blind evaluator dataset.

Development Benchmark

The local V4 benchmark is separate from the official Phase 2 dataset.

Repository Structure

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

Important Files

File

Purpose

register.py

Evaluator-facing production entry point

requirements.txt

Pinned Python dependencies

generate_dataset.py

Dataset generation

models/best_model.pth

Production trained model

src/preprocessing/ice.py

Image conditioning

src/coarse_search/gspe.py

Global candidate generation

src/ai_refinement/

Learned refinement

src/integration/pipeline_backup_v2_ai.py

Integrated production pipeline

failure_analysis.pdf

Failure analysis

README.md

Project documentation

Reproducibility

Clone the Phase 2 branch:

git clone -b phase2-development https://github.com/marakahansika27-prog/SilicoForge.git SilicoForge-Phase2
cd SilicoForge-Phase2

Create and activate the environment:

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

Install dependencies:

python -m pip install -r requirements.txt

Verify:

python -m pip check

Run the production interface:

python register.py --help

Run registration:

python register.py --input pairs.csv --output predictions.csv

The model is stored in the repository, so inference does not require downloading model weights from the network.

Team

SilicoForge

B V Raju Institute of Technology, Narsapur

License

No separate open-source license has been declared for this project.

Unless a license file is added, repository contents should be treated as project/submission source code.
