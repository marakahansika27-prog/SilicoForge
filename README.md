SilicoForge

Hybrid Sub-Pixel Localization for Semiconductor Pattern Images

SilicoForge is a hybrid computer-vision and deep-learning framework for locating semiconductor layout patterns inside larger search images.

The project combines:

deterministic image conditioning,

correlation-based global search,

multi-scale and multi-rotation hypothesis generation,

spatially diverse candidate retention,

full-resolution verification,

sub-pixel localization,

learned residual refinement,

confidence-aware decision making,

and explicit target rejection.

The project evolved through two major phases:

Phase 1: constrained-pose semiconductor pattern localization.

Phase 2: registration under unknown pose, including scale, rotation, degraded images, absent targets, pose recovery, rejection, and evaluator-facing CPU execution.

The public project name is SilicoForge. The working development name used during implementation was Drift-Sense V2.

Target semiconductor pattern families include:

DRAM

FinFET

Table of Contents

Project Identity

Team

Project Evolution

Phase 1

Phase 1 Problem Definition

Phase 1 Objectives

Phase 1 Architecture

Phase 1 Pipeline

Phase 1 Image Conditioning

Phase 1 Global Search

Phase 1 Candidate Localization

Phase 1 Sub-Pixel Localization

Phase 1 SNRN

Phase 1 Decision Fusion

Phase 1 Dataset Generation

Phase 1 Training

Phase 1 Evaluation

Phase 1 Final Benchmark

Phase 1 Lessons Learned

Phase 2

Phase 2 Problem Definition

Phase 2 Official Dataset Structure

Phase 2 Scoring Structure

Phase 2 Engineering Changes

Phase 2 Architecture

Phase 2 Image Conditioning

Phase 2 GSPE

Phase 2 Candidate Diversity

Phase 2 Candidate Verification

Phase 2 Pose Recovery

Phase 2 Sub-Pixel Refinement

Phase 2 AI Refinement

Phase 2 Rejection

Phase 2 Confidence

Phase 2 Output Contract

Phase 2 Runtime

Phase 2 Local Validation

Phase 2 Failure Analysis

Phase 2 RGB Handling

Phase 2 Submission Package

Production Entry Point

Repository Structure

Core Modules

Dataset Generator

Model Artifact

Requirements

Installation

Phase 1 Quick Start

Phase 2 Quick Start

Evaluator Workflow

Input Specification

Output Specification

Coordinate Convention

Training Workflow

Evaluation Workflow

Ground-Truth Leakage Prevention

Reproducibility

Performance

Known Limitations

Design Decisions

Why Hybrid CV + AI

Periodic Ambiguity

Candidate Recall

Confidence and Rejection

Submission Safety

Troubleshooting

Documentation

References

Project Status

License

Project Identity

Item

Value

Public project name

SilicoForge

Working development name

Drift-Sense V2

Domain

Semiconductor image localization

Pattern families

DRAM and FinFET

Primary task

Locate a reference pattern in a larger search image

Phase 1 output

(x, y)

Phase 2 output

(x, y, theta, scale, found, score)

Global search

GSPE

Image conditioning

ICE

Learned refinement

SNRN

Production entry point

register.py

Model artifact

models/best_model.pth

Execution target

CPU-only

Network during inference

Not required

Current milestone

Phase 2 submission engineering

Team

Team SilicoForge

Member

Year

Maraka Hansika

III Year

C.H. Lakshmi Varshitha

III Year

K. Rohith Reddy

III Year

R. Jahnavi

III Year

Institution: B V Raju Institute of Technology, Narsapur

Project Evolution

SilicoForge was developed as a layered localization system rather than as a single end-to-end black-box predictor.

The engineering progression was:

Constrained localization
|
v
Image conditioning
|
v
Global correlation search
|
v
Sub-pixel localization
|
v
Learned residual refinement
|
v
Decision fusion
|
v
Phase 1 localization
|
v
Phase 2 unknown-pose registration
|
v
Scale + rotation search
|
v
Candidate diversity
|
v
Pose recovery + rejection
|
v
Evaluator-facing production pipeline

The central design principle is to let classical computer vision perform broad spatial reasoning and let the learned model perform a small, local residual correction.

This keeps the neural model's task narrow and interpretable.

Phase 1

Phase 1 established the core localization framework.

The original task assumed a constrained pose and concentrated on accurate spatial localization of a semiconductor reference pattern inside a larger search image.

The final Phase 1 output was a sub-pixel coordinate:

(x, y)

Phase 1 also established the separation between:

global localization,

local precision refinement,

learned residual correction,

and final decision fusion.

The neural network was not designed to replace global search.

Instead:

Global CV search
|
v
Candidate
|
v
Local precision
|
v
Small learned residual

Phase 1 Problem Definition

Given:

Reference image
Search image

the system estimates:

Center x
Center y

The coordinate is expressed in search-image coordinates.

The reference image contains the semiconductor pattern to locate.

The search image contains the larger scene in which that pattern occurs.

The system must estimate the center rather than only a template corner.

Fractional-pixel coordinates are supported by the refinement stage.

Ground-truth coordinates are not required by inference.

Ground truth is used only for:

dataset generation,

training labels,

and evaluation.

Phase 1 Objectives

Objective 1: Global Localization

Find a plausible region containing the reference pattern.

Objective 2: Fine Localization

Improve the coarse coordinate to sub-pixel precision.

Objective 3: Residual Correction

Use SNRN to predict a small coordinate correction.

Objective 4: Confidence-Aware Refinement

Avoid blindly replacing a strong classical estimate with an unreliable learned estimate.

Objective 5: Reproducibility

Provide:

source code,

model weights,

dataset generation,

dependency specification,

evaluator interface,

and documentation.

Phase 1 Architecture

Reference Image
|
v
Image Conditioning Engine (ICE)
|
v
Search Image
|
v
Global Search Proposal Engine (GSPE)
|
v
Candidate Localization
|
v
Sub-Pixel Peak Estimation
|
+-----------------------+
|                       |
v                       v
Classical Coordinate      SNRN Residual
|
v
(dx, dy)
|                       |
+-----------+-----------+
|
v
Confidence-Aware
Decision Fusion
|
v
Final (x,y)

Phase 1 Pipeline

The Phase 1 production concept is:

Load reference

Load search image

Condition image pair

Generate global search hypotheses

Compute correlation responses

Extract candidate peaks

Select candidate

Refine peak

Prepare AI input

Predict residual

Estimate confidence

Fuse classical and AI results

Return final coordinate

The separation between broad search and local refinement is intentional.

It avoids applying expensive fine processing to the entire search image.

Phase 1 Image Conditioning

The Image Conditioning Engine is abbreviated as ICE.

Implementation:

src/preprocessing/ice.py

ICE is placed before global matching.

Its purpose is to reduce nuisance variation while preserving the structural information required for localization.

The conditioning stage is kept separate from GSPE so that:

preprocessing can be tested independently,

global search can be diagnosed independently,

and the overall pipeline remains modular.

Phase 1 Global Search

The Global Search Proposal Engine is abbreviated as GSPE.

Implementation:

src/coarse_search/gspe.py

GSPE performs the broad localization stage.

The search is based on correlation responses derived from image similarity.

Normalized cross-correlation concepts provide a spatial response surface.

High-response regions become candidate locations.

The important property is that the response surface remains spatially interpretable.

Phase 1 Candidate Localization

After global search, the response surface contains candidate peaks.

The highest peak is not guaranteed to be the correct target.

This became an important engineering lesson because semiconductor layout patterns can be highly repetitive.

Repeated structures can create several visually similar peaks.

The selected candidate is therefore passed to local refinement.

The candidate coordinate is represented in search-image coordinates.

Phase 1 Sub-Pixel Localization

The system refines an integer candidate using the local correlation surface.

Conceptually:

Integer peak
|
v
Local response neighborhood
|
v
Sub-pixel interpolation
|
v
Fractional coordinate

The resulting coordinate may contain fractional values such as:

(817.7891, 676.1043)

The exact value depends on the image pair.

Sub-pixel estimation is performed only after candidate localization.

Phase 1 SNRN

SNRN stands for Subpixel Navigation Refinement Network.

It is the learned refinement component.

The network predicts a small residual:

(dx, dy)

The refined coordinate can be expressed as:

x_final = x_classical + dx
y_final = y_classical + dy

Relevant implementation area:

src/ai_refinement/

Relevant components include:

network.py
inference.py
dataset.py
augmentations.py
loss.py
trainer.py

The trained model artifact is:

models/best_model.pth

The model is used as a local refinement component rather than as the global search engine.

Phase 1 Decision Fusion

Decision fusion prevents the AI estimate from automatically overriding the classical estimate.

The classical localization provides the primary spatial signal.

The AI model provides a residual correction.

The system evaluates whether the learned correction should be trusted.

Conceptually:

Classical coordinate
|
+------------------+
|
AI residual -------------->|
| Decision Fusion
AI confidence ------------>|
|
v
Final coordinate

This design was chosen for interpretability and safety.

Phase 1 Dataset Generation

The project contains a standalone dataset generator:

generate_dataset.py

The generator supports the principal semiconductor pattern families:

DRAM
FinFET

Generated pairs contain:

reference images,

search images,

and ground-truth metadata used for evaluation.

Dataset generation is separated from inference.

This separation helps prevent accidental use of ground-truth coordinates during registration.

Phase 1 Training

Training uses synthetic data and is separate from evaluator inference.

The AI component is trained for residual coordinate refinement.

The training process produces a PyTorch checkpoint.

The public repository includes the trained production model artifact.

Training should not be required for ordinary evaluator inference.

Phase 1 Evaluation

Phase 1 evaluation compares predicted coordinates with recorded ground truth.

Typical metrics include:

Mean error
Median error
RMSE
Threshold success
Duplicate checks
Missing-case checks

Evaluation also checks data integrity.

Ground-truth coordinates are not supplied to the inference pipeline.

Phase 1 Final Benchmark

The historical Phase 1 benchmark contained:

60 total cases
30 DRAM
30 FinFET

Recorded result:

40 / 60
= 66.7%

Recorded mean error:

65.78 px

Recorded median error:

1.10 px

Recorded RMSE:

134.53 px

The benchmark included integrity checks for:

duplicate cases,

missing cases,

and ground-truth leakage.

These numbers are historical Phase 1 results.

They are not official Phase 2 leaderboard results.

Phase 1 Lessons Learned

Phase 1 demonstrated that strong local accuracy can coexist with large catastrophic errors.

The much smaller median error compared with the mean indicated a distribution containing both accurate predictions and severe failures.

The most important lesson was periodic ambiguity.

Repeated semiconductor structures can produce:

Correct high peak
+
Incorrect high peak
+
Other visually similar peaks

A single global argmax can therefore select the wrong repeated structure.

This led directly to the Phase 2 focus on candidate recall and spatial diversity.

Phase 2

Phase 2 extends the system to registration under unknown pose.

The Phase 2 problem introduces:

unknown scale,

unknown rotation,

degraded images,

possible target absence,

pose recovery,

explicit rejection,

confidence,

and strict CPU runtime requirements.

The Phase 2 architecture retains the proven Phase 1 hybrid philosophy while expanding the search and decision layers.

Phase 2 Problem Definition

Given:

Reference semiconductor pattern
Larger search image

the system must determine:

Target position
Target rotation
Target scale
Target presence / absence

and return:

x
y
theta
scale
found
score

The disclosed search range is approximately:

Scale:    8 to 12
Rotation: -5 to +5 degrees

The system must also handle degraded image conditions and repetitive structures.

Phase 2 Official Dataset Structure

The organizer-defined Phase 2 composition is:

Set A: 70 nominal present pairs
Set B: 70 degraded present pairs
Set C: 40 absent pairs
Set D: 20 RGB optical bonus pairs

Total:

70 + 70 + 40 + 20 = 200 pairs

Sets A, B, and C form the core grayscale evaluation groups.

Set D is an optical/RGB bonus group.

The local V4 benchmark used during engineering is not identical to this official composition.

Phase 2 Scoring Structure

The Phase 2 base scoring dimensions are:

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

Base total

100

A possible bonus adds up to:

10 points

The bonus includes:

optical credit,

and a rejection F1 component.

The evaluation therefore rewards more than raw coordinate accuracy.

Phase 2 Engineering Changes

Phase 2 introduced or strengthened the following capabilities:

Explicit scale search.

Explicit rotation search.

Multiple spatial candidates.

Candidate verification.

Pose recovery.

Absent-target rejection.

Confidence-aware decision making.

Phase-2-style dataset generation.

CPU-oriented execution.

Evaluator-facing register.py.

The implementation keeps the global-search and local-refinement stages separate.

Phase 2 Architecture

The production path is:

Reference + Search
|
v
ICE
|
v
GSPE
|
+--> Multi-scale hypotheses
|
+--> Multi-rotation hypotheses
|
v
Top-K spatial candidates
|
v
Candidate verification
|
v
Full-resolution NCC
|
v
Sub-pixel refinement
|
v
SNRN / AI refinement
|
v
Decision Fusion
|
+----------------+
|                |
v                v
FOUND           REJECT
|                |
+-------+--------+
|
v
(x, y, theta, scale, found, score)

Phase 2 Image Conditioning

ICE remains part of the Phase 2 production path.

The reason is unchanged:

Condition input
|
v
Reduce nuisance variation
|
v
Run pose-aware global search

Keeping conditioning separate from search makes the system easier to diagnose and maintain.

Phase 2 GSPE

GSPE becomes the major broad-search engine for Phase 2.

Instead of assuming one fixed geometry, the system evaluates multiple scale and rotation hypotheses.

The scale range is approximately:

8
9
10
11
12

The rotation range is approximately:

-5 degrees
...
0 degrees
...
+5 degrees

The exact evaluated hypothesis set is an implementation detail.

The key design change is that global search becomes pose-aware.

Phase 2 Candidate Diversity

Candidate diversity is critical because semiconductor structures are repetitive.

A single global maximum can be misleading.

The system therefore retains multiple spatially distinct candidates.

The purpose is not to output multiple final predictions.

The purpose is to preserve alternatives for downstream verification.

The production GSPE configuration evaluates multiple coarse hypotheses and retains a Top-K candidate set.

Spatial separation is used to avoid filling the candidate set with near-duplicate peaks.

Phase 2 Candidate Verification

Candidate verification happens after broad search.

Expensive precision processing is not applied to every possible location.

Instead:

Global search
|
v
Shortlist
|
v
High-resolution verification

Candidate geometry includes:

x
y
scale
rotation

Verification uses correlation-based evidence at higher resolution.

Phase 2 Pose Recovery

Phase 2 output includes:

theta
scale

The rotation convention is:

theta = rotation of the reference as it appears
in the wide-search image

Positive rotation is counter-clockwise.

Scale represents the recovered down-scaling factor.

The nominal search interval is approximately:

8 to 12

Phase 2 Sub-Pixel Refinement

After candidate selection, the system performs local high-resolution localization.

The process is:

Selected candidate
|
v
Full-resolution NCC
|
v
Local peak
|
v
Sub-pixel refinement
|
v
Fractional x,y

The final coordinate remains in wide-search coordinates.

The output represents the center of the located reference pattern.

Phase 2 AI Refinement

The learned refinement stage remains local.

The design is:

Classical candidate
|
v
Sub-pixel coordinate
|
+------> SNRN
|
v
(dx, dy)
|
v
Decision Fusion

The neural model is not responsible for discovering the target over the entire search image.

This limits the learned component to a smaller residual-correction problem.

Phase 2 Rejection

Phase 2 requires explicit handling of absent targets.

The production entry point uses a fixed GSPE rejection threshold:

GSPE_REJECTION_THRESHOLD = 0.85

The decision logic is:

score >= 0.85
|
v
found = 1

score < 0.85
|
v
found = 0

When a target is rejected, pose fields are zeroed in the evaluator output.

This produces a contract-safe representation:

found = 0
x = 0
y = 0
theta = 0
scale = 0

The score is retained.

Phase 2 Confidence

The score field is the system's evaluator-facing confidence value.

It is intended to be monotonic with match confidence rather than to represent a calibrated probability.

An internal AI/refinement confidence may also be used during decision fusion.

These values should not be confused:

Internal refinement confidence
!=
Evaluator-facing score

For example, a local engineering demo produced an internal confidence near 0.9987 while the final GSPE/evaluator-facing score was approximately 0.9422.

That example is a demonstration of the distinction, not an official benchmark result.

Phase 2 Output Contract

The production output header is exactly:

pair_id,x,y,theta,scale,found,score

One row is required for every input pair.

For a detected target:

found = 1
x, y, theta, scale = recovered values

For a rejected or absent target:

found = 0
x = 0
y = 0
theta = 0
scale = 0

The pair_id must remain exactly as supplied by the evaluator.

Phase 2 Runtime

The target evaluator environment is CPU-only.

The project is validated with:

Python 3.11
CPU execution
No GPU required
No network required during inference

The organizer runtime requirement is:

Median <= 5 seconds per pair
Hard timeout = 20 seconds per pair

A 200-pair local Python 3.11 production run completed in:

278.0669 seconds total

Average:

approximately 1.39 seconds per pair

This is a local engineering measurement.

It should not be presented as the organizer's official runtime measurement.

Phase 2 Local Validation

The local V4 benchmark used during engineering contained:

200 total cases
100 present
100 absent
50 DRAM present
50 DRAM absent
50 FinFET present
50 FinFET absent

It was a balanced synthetic engineering benchmark.

It is not the official A/B/C/D composition.

The final production threshold test produced:

TP = 98
FN = 2
FP = 0
TN = 100

Derived rejection metrics:

Precision = 1.0000
Recall    = 0.9800
F1        = 0.9899

For present cases that were found, local coordinate results were:

Mean error   = 114.5358 px
Median error = 58.4594 px
Maximum      = 399.935 px

Threshold percentages:

Within 1 px  = 35.71%
Within 2 px  = 44.90%
Within 3 px  = 45.92%
Within 5 px  = 46.94%
Within 10 px = 46.94%
Within 50 px = 48.98%

These measurements are local engineering evidence only.

They are not official Phase 2 leaderboard results.

Phase 2 Failure Analysis

The principal observed failure modes are:

Periodic aliasing

Repeated layout structures can produce strong incorrect correlation peaks.

Top-1 selection failure

The correct candidate can exist in the search response while an incorrect repeated structure ranks first.

Candidate generation failure

If the correct target is not retained in the candidate set, later refinement cannot recover it.

Rejection false negatives

A present target can occasionally fall below the fixed rejection threshold.

AI refinement limitation

A local learned residual cannot repair a globally incorrect candidate.

The resulting engineering conclusion was:

Candidate recall is relatively strong.
Candidate selection remains the primary localization weakness.

No unvalidated reranking rule is enabled in the frozen production path.

Phase 2 RGB Handling

The Phase 2 input loader explicitly reads images in grayscale.

The production path therefore accepts RGB image files but converts them to grayscale during ingestion.

This means:

RGB file input
|
v
Grayscale processing

The system does not claim to exploit color information.

Therefore:

RGB files are ingestible,

color channels are discarded,

no color-specific feature is claimed,

and no official Set D bonus result is claimed.

Set D performance must be measured using the organizer's official optical dataset before making any bonus claim.

Phase 2 Submission Package

The Phase 2 submission package contains the required evaluator-facing artifacts.

Required files:

register.py
requirements.txt
generate_dataset.py
failure_analysis.pdf
best_model.pth

The production model is included inside the submission package.

The evaluator-facing repository also contains documentation and source modules required by the implementation.

The official submission ZIP should not contain:

.venv/
local datasets/
temporary outputs/
debug artifacts/
temporary evaluation scripts/

Production Entry Point

The authoritative production entry point is:

register.py

Run:

python register.py --input pairs.csv --output predictions.csv

The command-line interface is intentionally simple:

usage: register.py [-h] --input INPUT --output OUTPUT

The entry point:

reads the supplied pair CSV,

loads the reference and search images,

runs the Phase 2 registration pipeline,

applies the production rejection logic,

writes one prediction row per pair.

Repository Structure

A simplified repository structure is:

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

Additional source files and directories are retained in the repository for training, evaluation, utilities, and development support.

Core Modules

ICE

src/preprocessing/ice.py

Handles image conditioning before global matching.

GSPE

src/coarse_search/gspe.py

Performs broad correlation-based candidate generation.

SNRN

src/ai_refinement/

Provides learned local residual refinement.

Integration Pipeline

src/integration/pipeline_backup_v2_ai.py

Contains the integrated production path used by the evaluator-facing entry point.

Register

register.py

Provides the final evaluator-facing interface and output contract.

Dataset Generator

The public generator is:

generate_dataset.py

Its role is dataset creation, not evaluator inference.

The generator supports the project's semiconductor pattern families, including:

DRAM
FinFET

The generated data can contain:

reference images,

search images,

transformed targets,

and ground-truth metadata.

The inference path must not use the generated ground-truth coordinates.

Model Artifact

The production model is:

models/best_model.pth

The model is a PyTorch checkpoint.

It is loaded by the production pipeline when AI refinement is enabled.

Inference is CPU-compatible.

The repository contains the model artifact so that evaluator execution does not require network access or external model downloads.

Requirements

Dependencies are pinned in:

requirements.txt

The validated environment uses Python 3.11.

Important validated packages include:

numpy==2.4.6
scipy==1.17.1
opencv-python==5.0.0.93
pandas==3.0.5
scikit-image==0.26.0
torch==2.13.0
torchvision==0.28.0
tifffile==2026.3.3

The dependency file is the authoritative source for the complete package list.

After installation, verify with:

python -m pip check

Expected:

No broken requirements found.

Installation

SilicoForge Phase 2 is validated for Python 3.11 and CPU-only execution.

Clone the Phase 2 Repository

Clone the Phase 2 branch directly:

git clone -b phase2-development https://github.com/marakahansika27-prog/SilicoForge.git SilicoForge-Phase2
cd SilicoForge-Phase2

This explicitly checks out the phase2-development branch.

If a local repository already exists, do not clone into the existing non-empty directory.

Create a Python 3.11 Environment

Windows PowerShell:

py -3.11 -m venv .venv
..venv\Scripts\Activate.ps1

Verify:

python --version

Expected:

Python 3.11.x

Linux/macOS:

python3.11 -m venv .venv
source .venv/bin/activate
python --version

Install Dependencies

python -m pip install -r requirements.txt

Verify Dependencies

python -m pip check

Expected:

No broken requirements found.

Verify the Entry Point

python register.py --help

Verify the Model

The production model should exist at:

models/best_model.pth

Run Registration

python register.py --input pairs.csv --output predictions.csv

Phase 1 Quick Start

Phase 1 is the completed constrained-pose localization foundation.

The processing concept is:

Reference
|
v
ICE
|
v
GSPE
|
v
Candidate
|
v
Sub-pixel refinement
|
v
SNRN residual
|
v
Decision fusion
|
v
(x, y)

Phase 1 benchmark numbers in this README are historical project measurements.

They should not be interpreted as Phase 2 results.

Phase 2 Quick Start

Clone

git clone -b phase2-development https://github.com/marakahansika27-prog/SilicoForge.git SilicoForge-Phase2
cd SilicoForge-Phase2

Create Environment

Windows:

py -3.11 -m venv .venv
..venv\Scripts\Activate.ps1

Install

python -m pip install -r requirements.txt

Verify

python -m pip check

Run

python register.py --input pairs.csv --output predictions.csv

Output

pair_id,x,y,theta,scale,found,score

Exactly one row must be produced per input pair.

Evaluator Workflow

The evaluator workflow is:

Evaluator pair CSV
|
v
register.py
|
v
Load reference/search
|
v
ICE
|
v
GSPE
|
v
Candidate verification
|
v
Sub-pixel + AI refinement
|
v
Decision / rejection
|
v
predictions.csv

The evaluator should execute the command from the repository root.

No network access should be required during inference.

Input Specification

The production command accepts:

--input pairs.csv

The input CSV contains the evaluator-supplied pair identifiers and image paths.

The supplied pair_id must be preserved exactly.

A simple development example is:

pair_id,reference,search
example_001,path/to/reference.png,path/to/search.png

The exact organizer input schema should always take precedence over local development examples.

Output Specification

The output CSV header is exactly:

pair_id,x,y,theta,scale,found,score

For a found target:

example_001,652.07,161.02,0.0,10.0,1,0.94

For a rejected target:

example_002,0,0,0,0,0,0.42

Rules:

preserve every pair_id,

output exactly one row per input pair,

use wide-search coordinates,

output floating-point x and y,

output rotation in degrees,

output the recovered scale,

use found=1 for accepted matches,

use found=0 for rejected/absent targets,

zero pose fields when found=0.

Coordinate Convention

Coordinates are expressed in the wide search image.

The output position represents the center of the located reference pattern.

The rotation convention is:

theta = rotation of the reference as it appears
in the wide-search image

Positive rotation is counter-clockwise.

Scale is the recovered down-scaling factor.

The nominal Phase 2 range is:

8 to 12

Training Workflow

Training is separate from evaluator inference.

A conceptual training workflow is:

Generate synthetic data
|
v
Create reference/search pairs
|
v
Compute residual labels
|
v
Train SNRN
|
v
Validate
|
v
Save checkpoint

The resulting checkpoint is used during inference.

Training is not required for ordinary evaluator execution when the supplied model artifact is available.

Evaluation Workflow

A local evaluation workflow is:

Generate / prepare benchmark
|
v
Create input CSV
|
v
Run register.py
|
v
Read predictions
|
v
Compare against ground truth
|
v
Calculate metrics
|
v
Inspect failure modes

Evaluation should distinguish:

Found present target
Rejected present target
Correctly rejected absent target
False positive absent target

Localization error should only be computed for present cases that were accepted as found.

Ground-Truth Leakage Prevention

Ground truth is reserved for:

dataset generation,

training labels,

and evaluation.

The production inference path should not receive:

ground-truth x
ground-truth y
ground-truth theta
ground-truth scale

as input.

The inference path must determine the result from the supplied reference and search images.

This separation is an important reproducibility and benchmark-integrity requirement.

Reproducibility

For a clean reproducibility check:

git clone -b phase2-development https://github.com/marakahansika27-prog/SilicoForge.git SilicoForge-Phase2
cd SilicoForge-Phase2

Create the environment:

py -3.11 -m venv .venv
..venv\Scripts\Activate.ps1

Install:

python -m pip install -r requirements.txt

Verify:

python -m pip check

Verify the production interface:

python register.py --help

Then run the evaluator command:

python register.py --input pairs.csv --output predictions.csv

A clean-clone engineering test has been performed successfully with the Phase 2 branch.

The test verified:

repository clone,

Python 3.11 environment,

dependency installation,

importability,

model availability,

production CLI,

actual image-pair processing,

and valid output CSV generation.

Performance

The system is designed around a staged architecture:

Cheap broad search
|
v
Small candidate set
|
v
Expensive local verification
|
v
Small AI refinement

This avoids applying full-resolution and AI processing across the entire search image.

A local Python 3.11 CPU run over 200 engineering cases completed in:

278.0669 seconds

Average:

~1.39 seconds/pair

This local measurement is provided as engineering evidence.

Official evaluator runtime remains authoritative.

Known Limitations

Periodic Structures

Repeated semiconductor structures can produce strong incorrect correlation peaks.

Top-1 Selection

The strongest correlation peak is not guaranteed to correspond to the required target.

Candidate Pool

If the correct location is not retained in the candidate pool, local refinement cannot recover it.

RGB

The current production path converts RGB input to grayscale and does not exploit color information.

Official Set D

No official Set D optical bonus result is claimed because the official blind Set D data was not available for local validation.

Threshold Calibration

The 0.85 rejection threshold is a production engineering choice based on the available local validation.

It is not claimed to be globally optimal for the organizer's blind dataset.

Local Benchmark

The V4 engineering dataset is not the organizer's official Phase 2 A/B/C/D dataset.

Design Decisions

Layered Architecture

Global search and local refinement are intentionally separated.

Candidate Diversity

Multiple spatial candidates are retained because repeated structures create aliases.

Full-Resolution Verification

Coarse correlation is followed by higher-resolution verification before final output.

Sub-Pixel Refinement

Local response interpolation improves coordinate precision.

Learned Residual

SNRN predicts a small correction instead of solving the entire global localization problem.

Confidence-Aware Fusion

AI refinement is treated as a correction signal rather than an unconditional replacement.

Fixed Production Threshold

The production path uses a fixed rejection threshold to keep evaluator behavior deterministic.

Why Hybrid CV + AI

A pure classical approach has difficulty with:

degraded imagery,

local appearance variation,

and small residual localization errors.

A pure neural global locator has difficulty with:

precise spatial search,

repeated semiconductor structures,

deterministic behavior,

and interpretability.

The hybrid architecture combines their strengths:

Classical CV
|
+--> broad spatial reasoning
|
+--> correlation surface
|
+--> candidate generation
|
v
Learned refinement
|
+--> local residual correction
|
v
Decision fusion

This division of responsibility keeps the neural problem smaller.

Periodic Ambiguity

Periodic ambiguity is one of the most important failure modes.

Consider a repeated pattern:

|--A--|--A--|--A--|--A--|

A template may correlate strongly with several occurrences.

Therefore:

Highest score
!=
Always correct location

The system addresses this by retaining multiple spatially distinct candidates.

However, candidate diversity does not completely solve the final selection problem.

This is why periodic aliasing remains a known limitation.

Candidate Recall

Candidate recall is a prerequisite for correct localization.

If:

Correct candidate is generated

then downstream verification and refinement have an opportunity to select it.

If:

Correct candidate is never generated

then:

NCC cannot recover it
SNRN cannot recover it
Decision fusion cannot recover it

Therefore the Phase 2 architecture prioritizes candidate diversity before expensive refinement.

Confidence and Rejection

The rejection problem is different from coordinate refinement.

For an absent target, the system must avoid returning a random high-scoring repeated structure as a valid match.

The production decision is:

GSPE score >= 0.85
|
v
FOUND

GSPE score < 0.85
|
v
REJECT

On rejection:

found = 0
x = 0
y = 0
theta = 0
scale = 0

This prevents stale or invalid pose values from being emitted for rejected pairs.

Local engineering validation produced:

Precision = 1.0000
Recall    = 0.9800
F1        = 0.9899

These values are local validation results, not official leaderboard results.

Submission Safety

Before creating the final submission ZIP, verify:

register.py
requirements.txt
generate_dataset.py
failure_analysis.pdf
best_model.pth

Verify the model is actually present.

Verify the requirements file installs in a clean Python 3.11 environment.

Verify:

python -m pip check

Verify:

python register.py --help

Verify the output header:

pair_id,x,y,theta,scale,found,score

Do not include:

.venv/
dataset/
outputs/
temporary logs/
temporary benchmark CSVs/
debug scripts/

unless the submission instructions explicitly require them.

Troubleshooting

destination path already exists

If:

fatal: destination path 'SilicoForge' already exists

appears, clone into a new directory:

git clone -b phase2-development https://github.com/marakahansika27-prog/SilicoForge.git SilicoForge-Phase2

Wrong Python Version

Check:

python --version

Use Python 3.11.

Dependency Problems

Run:

python -m pip install -r requirements.txt
python -m pip check

Model Not Found

Verify:

Test-Path models\best_model.pth

Expected:

True

register.py Usage Error

Run:

python register.py --help

Correct usage:

python register.py --input pairs.csv --output predictions.csv

Input Image Not Found

Check that the paths in the input CSV are valid from the repository root.

Rejected Target Has Coordinates

For found=0, production output should contain:

x = 0
y = 0
theta = 0
scale = 0

The evaluator-facing score may remain nonzero because it records the internal matching score.

Documentation

The repository includes:

README.md
failure_analysis.pdf

The README documents:

Phase 1 architecture,

Phase 2 architecture,

installation,

evaluator usage,

output contract,

dataset structure,

limitations,

validation,

and submission safety.

The failure analysis documents:

observed failure modes,

periodic aliasing,

selection failures,

rejection behavior,

local validation,

and production design decisions.

References

Organizer Phase 2 Specification

The organizer-provided Phase 2 specification defines:

registration under unknown pose,

scale and rotation search,

present and absent cases,

the official A/B/C/D composition,

output fields,

scoring,

rejection,

runtime,

and submission requirements.

The organizer specification is authoritative for evaluation.

Core Technical Concepts

The implementation uses established computer-vision concepts including:

normalized cross-correlation,

image conditioning,

multi-scale matching,

rotation hypotheses,

local peak refinement,

sub-pixel interpolation,

and learned residual regression.

The project-specific implementation is contained in this repository.

Project Status

Phase 1

Status:

COMPLETED

Historical benchmark:

40 / 60
66.7%

Phase 2

Status:

SUBMISSION ENGINEERING / FROZEN PRODUCTION PATH

Implemented:

Unknown scale                  [x]
Unknown rotation               [x]
Multi-candidate search         [x]
Candidate verification         [x]
Sub-pixel localization         [x]
AI residual refinement         [x]
Pose output                    [x]
Absent-target rejection        [x]
Confidence score               [x]
CPU-only execution             [x]
Evaluator-facing register.py   [x]
Submission model artifact      [x]
Failure analysis               [x]
Installation documentation     [x]

Local clean-clone verification:

PASS

The clean-clone test verified that the Phase 2 branch can be cloned into a fresh directory, dependencies can be installed, the model can be loaded, the production entry point can execute, and a real image pair can produce a valid prediction CSV.

Final Production Command

For Phase 2 evaluator execution:

python register.py --input pairs.csv --output predictions.csv

Expected output header:

pair_id,x,y,theta,scale,found,score

The evaluator-supplied dataset and scoring rules remain authoritative.

License

No separate open-source license has been declared in this project documentation.

Unless a license file is added to the repository, users should treat the repository contents as source code provided for the intended project/submission context rather than assuming broad redistribution rights.