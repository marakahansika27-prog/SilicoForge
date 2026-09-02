# SilicoForge

## Hybrid Sub-Pixel Localization for Semiconductor Pattern Images

SilicoForge is a hybrid computer-vision and deep-learning framework for
locating semiconductor layout patterns inside larger search images. The
project is organized around two engineering phases. **Phase 1**
established the core localization pipeline under a constrained/known
pose assumption. **Phase 2** extends the same foundation to unknown
scale, unknown rotation, degraded images, absent targets, pose recovery,
confidence-aware rejection, and CPU-only evaluator execution. The public
project name is **SilicoForge**. The working project name used during
development was **Drift-Sense V2**. The target semiconductor pattern
families include **DRAM** and **FinFET**. The core philosophy is:

``` text
Classical Computer Vision
        +
Learned Residual Refinement
        +
Confidence-Aware Decision Making
        =
Robust Semiconductor Pattern Localization
```

  ---------------------------------------------------------------------
  \# Table of Contents - [Project Identity](#project-identity) -
  [Project Evolution](#project-evolution) - [Phase 1](#phase-1) -
  [Phase 1 Problem Definition](#phase-1-problem-definition) - [Phase 1
  Objectives](#phase-1-objectives) - [Phase 1
  Architecture](#phase-1-architecture) - [Phase 1
  Pipeline](#phase-1-pipeline) - [Phase 1 Image
  Conditioning](#phase-1-image-conditioning) - [Phase 1 Global
  Search](#phase-1-global-search) - [Phase 1 Candidate
  Localization](#phase-1-candidate-localization) - [Phase 1 Sub-Pixel
  Localization](#phase-1-sub-pixel-localization) - [Phase 1
  SNRN](#phase-1-snrn) - [Phase 1 Decision
  Fusion](#phase-1-decision-fusion) - [Phase 1 Dataset
  Generation](#phase-1-dataset-generation) - [Phase 1
  Training](#phase-1-training) - [Phase 1
  Evaluation](#phase-1-evaluation) - [Phase 1 Final
  Benchmark](#phase-1-final-benchmark) - [Phase 1 Lessons
  Learned](#phase-1-lessons-learned) - [Phase 2](#phase-2) - [Phase 2
  Problem Definition](#phase-2-problem-definition) - [Phase 2 Official
  Dataset Structure](#phase-2-official-dataset-structure) - [Phase 2
  Scoring Structure](#phase-2-scoring-structure) - [Phase 2 Engineering
  Changes](#phase-2-engineering-changes) - [Phase 2
  Architecture](#phase-2-architecture) - [Phase 2 Image
  Conditioning](#phase-2-image-conditioning) - [Phase 2
  GSPE](#phase-2-gspe) - [Phase 2 Candidate
  Diversity](#phase-2-candidate-diversity) - [Phase 2 Candidate
  Verification](#phase-2-candidate-verification) - [Phase 2 Pose
  Recovery](#phase-2-pose-recovery) - [Phase 2 Sub-Pixel
  Refinement](#phase-2-sub-pixel-refinement) - [Phase 2 AI
  Refinement](#phase-2-ai-refinement) - [Phase 2
  Rejection](#phase-2-rejection) - [Phase 2
  Confidence](#phase-2-confidence) - [Phase 2 Output
  Contract](#phase-2-output-contract) - [Phase 2
  Runtime](#phase-2-runtime) - [Phase 2 Local
  Validation](#phase-2-local-validation) - [Phase 2 Failure
  Analysis](#phase-2-failure-analysis) - [Phase 2 RGB
  Handling](#phase-2-rgb-handling) - [Phase 2 Submission
  Package](#phase-2-submission-package) - [Production Entry
  Point](#production-entry-point) - [Repository
  Structure](#repository-structure) - [Core Modules](#core-modules) -
  [Dataset Generator](#dataset-generator) - [Model
  Artifact](#model-artifact) - [Requirements](#requirements) -
  [Installation](#installation) - [Phase 1 Quick
  Start](#phase-1-quick-start) - [Phase 2 Quick
  Start](#phase-2-quick-start) - [Evaluator
  Workflow](#evaluator-workflow) - [Input
  Specification](#input-specification) - [Output
  Specification](#output-specification) - [Coordinate
  Convention](#coordinate-convention) - [Training
  Workflow](#training-workflow) - [Evaluation
  Workflow](#evaluation-workflow) - [Ground-Truth Leakage
  Prevention](#ground-truth-leakage-prevention) -
  [Reproducibility](#reproducibility) - [Performance](#performance) -
  [Known Limitations](#known-limitations) - [Design
  Decisions](#design-decisions) - [Why Hybrid CV +
  AI](#why-hybrid-cv--ai) - [Periodic Ambiguity](#periodic-ambiguity) -
  [Candidate Recall](#candidate-recall) - [Confidence and
  Rejection](#confidence-and-rejection) - [Submission
  Safety](#submission-safety) - [Troubleshooting](#troubleshooting) -
  [Documentation](#documentation) - [References](#references) -
  [Project Status](#project-status) - [License](#license)
  ---------------------------------------------------------------------
  \# Project Identity Item Value

  ---------------------------------------------------------------------

Public project name SilicoForge Working project name Drift-Sense V2
Domain Semiconductor image localization Pattern families DRAM and FinFET
Primary task Locate a reference pattern in a search image Phase 1 output
`(x, y)` Phase 2 output `(x, y, theta, scale, found, score)` Core
approach Hybrid classical CV + learned residual refinement Global search
GSPE Conditioning ICE AI refinement SNRN Final decision Confidence-aware
fusion Production entry point `register.py` Model artifact
`best_model.pth` Execution target CPU-only Network requirement None
during inference Current milestone Phase 2 submission engineering
------------------------------------------------------------------------
\# Project Evolution SilicoForge was not created as a single monolithic
model. The project evolved through a sequence of controlled engineering
stages. The first stage concentrated on deterministic localization. The
second stage introduced learned residual refinement. The next stage
concentrated on reproducibility. The project then moved to
candidate-search diagnostics. The diagnostics identified periodic
ambiguity as a major failure mode. The project was subsequently selected
for Phase 2. Phase 2 changes the task from constrained localization to
registration under unknown pose. The system therefore retains the proven
Phase 1 foundation while expanding the search and decision layers. The
evolution can be summarized as:

``` text
Phase 1
Known / constrained pose
        |
        v
ICE
        |
        v
GSPE
        |
        v
Sub-pixel localization
        |
        v
SNRN residual correction
        |
        v
Decision fusion
        |
        v
(x, y)
```

Phase 2 extends this to:

``` text
Phase 2
Unknown pose + possible absence
        |
        v
ICE
        |
        v
Multi-scale / multi-rotation GSPE
        |
        v
Multiple spatial candidates
        |
        v
Candidate verification
        |
        v
(x, y, theta, scale)
        |
        v
Local / AI refinement
        |
        v
Presence / rejection
        |
        v
Confidence-aware decision
        |
        v
(x, y, theta, scale, found, score)
```

  ---------------------------------------------------------------------
  \# Phase 1 Phase 1 established the original localization framework.
  The target was a known semiconductor pattern inside a larger search
  image. The original setting used a constrained pose. The original
  production path assumed a fixed scale near 10x. The original pipeline
  concentrated on accurate spatial localization. The final Phase 1
  result was a sub-pixel `(x, y)` coordinate. Phase 1 also established
  the use of learned residual refinement. The AI model was not intended
  to replace global search. Instead, the neural component learned a
  local residual. This design reduced the burden placed on the neural
  model. The classical pipeline first narrowed the search. The neural
  stage then operated near a candidate. The final fusion stage decided
  whether the refinement should be trusted.
  ---------------------------------------------------------------------
  \# Phase 1 Problem Definition The Phase 1 problem can be expressed
  as:

  ---------------------------------------------------------------------

# Phase 1 Objectives

Phase 1 had five major objectives. \## Objective 1: Global localization
Find a plausible region containing the reference. \## Objective 2: Fine
localization Improve the coordinate beyond the coarse search resolution.
\## Objective 3: Residual correction Use SNRN to predict a small
coordinate correction. \## Objective 4: Confidence-aware refinement
Avoid blindly replacing a strong classical estimate with a weak learned
estimate. \## Objective 5: Reproducibility Provide code, model weights,
dataset generation, dependency specification, and verification support.
------------------------------------------------------------------------
\# Phase 1 Architecture The Phase 1 architecture is:

``` text
Reference Image
       |
       v
Image Conditioning Engine
       |
       v
Search Image
       |
       v
Global Search Proposal Engine
       |
       v
Candidate Localization
       |
       v
Sub-Pixel Peak Estimation
       |
       +-------------------+
       |                   |
       v                   v
Classical Coordinate   SNRN Residual
       |                   |
       |                 (dx, dy)
       |                   |
       +---------+---------+
                 |
                 v
        Confidence-Aware
          Decision Fusion
                 |
                 v
             Final (x,y)
```

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  \# Phase 1 Pipeline The Phase 1 workflow is:
  `text 1. Load reference 2. Load search image 3. Condition image pair 4. Generate global search hypotheses 5. Compute correlation responses 6. Extract candidate peaks 7. Select candidate 8. Refine the peak 9. Prepare AI input 10. Predict residual 11. Estimate confidence 12. Fuse classical and AI results 13. Return final coordinate`
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  \# Phase 1 Image Conditioning The Image Conditioning Engine is abbreviated as ICE. Its purpose is to reduce irrelevant imaging variation. The module operates before global matching. The conditioning stage attempts to preserve structural information. The stage can normalize intensity behavior. It can reduce sensitivity to image
  variation. It is implemented in:

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Phase 1 Global Search

The Global Search Proposal Engine is abbreviated as GSPE. GSPE performs
the broad localization stage. The Phase 1 setting was based on a
constrained pose. The global search therefore concentrated on finding
the target spatially. The response surface is derived from image
similarity. The implementation is:

``` text
src/coarse_search/gspe.py
```

The project uses normalized cross-correlation concepts. The correlation
surface provides a spatially interpretable response. High response
regions are treated as candidate locations.
------------------------------------------------------------------------
\# Phase 1 Candidate Localization After global search, the response
surface contains candidate peaks. The highest candidate is not
automatically guaranteed to be correct. This became an important lesson
during later diagnostics. Repeated semiconductor patterns can produce
multiple high responses. Candidate selection therefore became a major
engineering concern. The selected candidate is passed to local
refinement. The candidate coordinate is represented in search-image
coordinates.
------------------------------------------------------------------------
\# Phase 1 Sub-Pixel Localization The system refines an integer
candidate using the local correlation surface. The conceptual sequence
is:

``` text
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
```

The result can contain fractional values. For example:

``` text
(817.7891, 676.1043)
```

The exact values depend on the image pair. Sub-pixel estimation is
performed after candidate localization. This ordering avoids applying
expensive fine refinement to the entire image.
------------------------------------------------------------------------
\# Phase 1 SNRN SNRN means Subpixel Navigation Refinement Network. SNRN
is the learned refinement component. The network predicts a small
residual. The residual is represented as:

``` text
(dx, dy)
```

The refined coordinate can be expressed as:

``` text
x_final = x_classical + dx
y_final = y_classical + dy
```

The model is located under:

``` text
src/ai_refinement/
```

Relevant modules include:

``` text
network.py
inference.py
dataset.py
augmentations.py
loss.py
trainer.py
```

The model artifact is stored under:

``` text
models/best_model.pth
```

  ---------------------------------------------------------------------
  \# Phase 1 Decision Fusion Decision fusion prevents the AI output
  from automatically overriding the classical coordinate. The classical
  result provides the primary localization signal. The AI result
  provides residual correction. Confidence is used to determine whether
  the refinement should be trusted. The conceptual design is:
  ---------------------------------------------------------------------
  \# Phase 1 Dataset Generation The project includes a standalone
  generator. The main public generator is:

  ---------------------------------------------------------------------

# Phase 1 Training

Training is provided under the project scripts. The training workflow
uses synthetic data. The AI component is trained for residual
refinement. Training is distinct from evaluator inference. The model
checkpoint is saved as a PyTorch artifact. The public repository
includes the trained model artifact.
------------------------------------------------------------------------
\# Phase 1 Evaluation Phase 1 evaluation compares predicted coordinates
with recorded ground truth. Evaluation metrics include:

``` text
mean error
median error
RMSE
threshold success
duplicate checks
missing-case checks
```

Evaluation also verifies data integrity. Ground-truth coordinates are
not supplied to the inference pipeline.
------------------------------------------------------------------------
\# Phase 1 Final Benchmark The final recorded Phase 1 benchmark
contained:

``` text
60 total cases
30 DRAM
30 FinFET
```

The final recorded Phase 1 success was:

``` text
40 / 60
= 66.7%
```

The recorded mean error was:

``` text
65.78 px
```

The recorded median error was:

``` text
1.10 px
```

The recorded RMSE was:

``` text
134.53 px
```

The benchmark integrity checks passed. There were no duplicate cases.
There were no missing cases. Ground-truth leakage was not observed.
These figures belong to the historical Phase 1 benchmark. They are not
Phase 2 leaderboard results.
------------------------------------------------------------------------
\# Phase 1 Lessons Learned Phase 1 demonstrated that strong local
accuracy can coexist with large catastrophic errors. The median error
was much smaller than the mean error. This indicates a distribution
containing both successful and severe failures. Periodic structures were
a major source of ambiguity. A global argmax can select a visually
similar but incorrect repeated structure. Candidate recall is therefore
fundamental. If the correct candidate is absent from the candidate pool,
later refinement cannot recover it. This motivated the Phase 2
candidate-diversity strategy.
------------------------------------------------------------------------
\# Phase 2 Phase 2 is the registration-under-unknown-pose extension. The
Phase 2 system continues the Phase 1 hybrid philosophy. The major
difference is that pose is no longer assumed to be fixed. The target may
also be absent. The evaluator therefore expects both localization and
rejection behavior. Phase 2 additionally requires scale and rotation
recovery. Confidence becomes an explicit scoring dimension. Runtime
becomes a strict engineering constraint.
------------------------------------------------------------------------
\# Phase 2 Problem Definition The Phase 2 task can be summarized as:

``` text
Given:
    reference semiconductor pattern
    larger search image
Unknown:
    target position
    scale
    rotation
    possible presence / absence
Return:
    x
    y
    theta
    scale
    found
    score
```

The search must handle approximately:

``` text
scale:    8x to 12x
rotation: -5 degrees to +5 degrees
```

The system must also handle degraded image conditions. These include
noise, blur, brightness variation, and repetitive structures.
------------------------------------------------------------------------
\# Phase 2 Official Dataset Structure The official Phase 2 composition
discussed for evaluation is:

``` text
Set A: 70 nominal present pairs
Set B: 70 degraded present pairs
Set C: 40 absent pairs
Set D: 20 RGB optical bonus pairs
```

The total is:

``` text
70 + 70 + 40 + 20 = 200 pairs
```

Sets A, B, and C are the core grayscale evaluation groups. Set D is an
optical/RGB bonus group. The local V4 benchmark used during engineering
is not identical to this official composition.
------------------------------------------------------------------------
\# Phase 2 Scoring Structure The Phase 2 base scoring dimensions are:

``` text
Localization       40 points
Pose recovery      20 points
Rejection          15 points
Confidence          10 points
Efficiency           5 points
Documentation      10 points
```

The base total is:

``` text
100 points
```

A possible bonus can add up to:

``` text
10 points
```

The bonus includes an optical credit component and a rejection F1
component. The official evaluation therefore rewards more than
coordinate accuracy.
------------------------------------------------------------------------
\# Phase 2 Engineering Changes Phase 2 required several changes. The
first change was explicit scale search. The second change was explicit
rotation search. The third change was candidate diversity. The fourth
change was periodic ambiguity handling. The fifth change was
absent-target rejection. The sixth change was pose output. The seventh
change was confidence-aware decision making. The eighth change was
Phase-2-style data generation. The ninth change was CPU optimization.
The tenth change was the evaluator-facing `register.py` interface.
------------------------------------------------------------------------
\# Phase 2 Architecture The production Phase 2 path is:

``` text
Reference + Search
        |
        v
ICE
        |
        v
GSPE
multi-scale / multi-rotation
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
SNRN / local refinement
        |
        v
Decision Fusion
        |
        +------------+
        |            |
        v            v
    FOUND          REJECT
        |            |
        +------+- ----+
               |
               v
(x, y, theta, scale, found, score)
```

  ---------------------------------------------------------------------
  \# Phase 2 Image Conditioning ICE remains part of the Phase 2
  production path. The reason is unchanged. Global search should
  operate on conditioned inputs. The conditioning stage reduces
  nuisance variation. The Phase 2 search then operates over pose
  hypotheses. This preserves the architectural separation between
  conditioning and search.
  ---------------------------------------------------------------------
  \# Phase 2 GSPE GSPE becomes the major Phase 2 search engine. Instead
  of relying on one fixed geometry, it evaluates multiple scale and
  rotation hypotheses. The target scale range is approximately:

  ---------------------------------------------------------------------

# Phase 2 Candidate Diversity

Candidate diversity is critical because semiconductor structures are
repetitive. A single global maximum can be misleading. The system
therefore retains multiple spatially distinct candidates. The purpose is
not to return multiple final predictions. The purpose is to preserve
alternatives for downstream verification. The production GSPE
configuration can evaluate multiple coarse hypotheses. The pipeline then
retains a Top-K candidate set. Spatial separation is used to avoid
returning near-duplicate peaks.
------------------------------------------------------------------------
\# Phase 2 Candidate Verification Candidate verification happens after
global search. The expensive precision stages should not run over every
possible location. Instead, the shortlisted candidates are evaluated.
Verification uses higher-resolution information. The candidate geometry
includes:

``` text
x
y
scale
rotation
```

The candidate score is derived from correlation-based evidence. The
system can also compare supporting local evidence.
------------------------------------------------------------------------
\# Phase 2 Pose Recovery Phase 2 requires the recovered pose. The output
includes:

``` text
theta
scale
```

The convention is:

``` text
theta = rotation of the reference as it appears in the wide search image
```

Positive rotation is counter-clockwise. Scale represents the recovered
down-scaling factor. The nominal scale interval is approximately 8 to
12.
------------------------------------------------------------------------
\# Phase 2 Sub-Pixel Refinement After a candidate is selected, the
system performs high-resolution localization. Full-resolution NCC is
used for local verification. The response peak is then refined. The
result can contain fractional coordinates. The output remains in
wide-search coordinates. The final center is therefore not simply a
top-left template coordinate.
------------------------------------------------------------------------

# License

See the repository license file, if present.

# Final Phase 1 + Phase 2 Status

Phase 1 is the completed constrained-pose foundation of SilicoForge.

Phase 2 is the unknown-pose extension with scale, rotation, candidate
verification, rejection, confidence, and CPU-only evaluator execution.

The public repository is:

``` text
https://github.com/marakahansika27-prog/SilicoForge
```

The Phase 2 evaluator entry point is:

``` text
python register.py --input pairs.csv --output predictions.csv
```

The project remains committed to reproducible, interpretable, offline
semiconductor pattern localization.

# Engineering Verification Matrix

  Area                  Phase 1                 Phase 2
  --------------------- ----------------------- ---------------------------------
  Input                 Reference + search      Reference + search manifest
  Localization          Yes                     Yes
  Sub-pixel             Yes                     Yes
  Scale search          Constrained             Approximately 8--12x
  Rotation search       Constrained             Approximately ±5°
  Candidate diversity   Limited                 Explicit Top-K retention
  Pose output           No                      Yes
  Absence handling      Not central             Explicit rejection
  Confidence            Refinement confidence   Acceptance/rejection confidence
  Runtime target        Development-focused     Strict CPU constraint
  Evaluator interface   `inference.py`          `register.py`
  Output                `(x,y)`                 CSV contract

------------------------------------------------------------------------

# Phase 1 Reproducibility Record

The Phase 1 repository was audited before the Phase 2 transition.

The audit covered the README.

The audit covered dataset generation.

The audit covered DRAM generation.

The audit covered FinFET generation.

The audit covered ground-truth recording.

The audit covered inference execution.

The audit covered the model artifact.

The audit covered the training implementation.

The audit covered dependency specification.

The audit covered references.

The audit covered pipeline documentation.

The audit covered ground-truth leakage protection.

The audit covered visual outputs.

The audit covered repository structure.

The final Phase 1 fresh-machine verification was recorded as ready.

------------------------------------------------------------------------

# Phase 2 Repository Hardening

Phase 2 required a new evaluator-facing entry point.

The new entry point is `register.py`.

The script is responsible for producing the complete CSV contract.

The production path loads the bundled model.

The production path operates without network access.

The production path is intended for CPU execution.

The output is one row per pair.

Rejected rows zero their pose fields.

This behavior is deliberately explicit.

------------------------------------------------------------------------

# Phase 2 Validation Interpretation

The local benchmark is an engineering benchmark.

It is useful for identifying regressions.

It is useful for testing output integrity.

It is useful for measuring runtime.

It is useful for evaluating rejection behavior.

It is useful for observing localization failure modes.

It is not a replacement for the official blind evaluation.

The official evaluator remains authoritative.

------------------------------------------------------------------------

# Important Numeric Distinction

The Phase 1 recorded result was:

``` text
40 / 60 = 66.7%
```

The Phase 2 local rejection F1 was:

``` text
0.9899
```

These numbers describe different experiments.

They should not be combined.

The Phase 1 result belongs to the historical 60-case benchmark.

The Phase 2 rejection result belongs to the local balanced V4 benchmark.

Neither number should be presented as the official Phase 2 leaderboard
score.

------------------------------------------------------------------------

# Practical Debugging Commands

Check the repository:

``` bash
git status
```

Check the branch:

``` bash
git branch -vv
```

Check the remote:

``` bash
git remote -v
```

Check recent commits:

``` bash
git log --oneline --decorate -10
```

Check tracked files:

``` bash
git ls-files
```

Check the model:

``` bash
git ls-files best_model.pth
```

Check the Phase 2 branch:

``` bash
git fetch origin
git rev-parse HEAD
git rev-parse origin/phase2-development
```

------------------------------------------------------------------------

# Practical Runtime Commands

Check Python:

``` bash
python --version
```

Check dependencies:

``` bash
python -m pip check
```

Check PyTorch:

``` bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Check OpenCV:

``` bash
python -c "import cv2; print(cv2.__version__)"
```

Run the production interface:

``` bash
python register.py --input pairs.csv --output predictions.csv
```

------------------------------------------------------------------------

# Output Validation Checklist

After running the evaluator command, check the row count.

Check that the header is exactly:

``` text
pair_id,x,y,theta,scale,found,score
```

Check that all pair identifiers are unique.

Check that every input identifier appears exactly once.

Check that `found` contains only zero or one.

Check that rejected pose fields are zero.

Check that accepted coordinates are numeric.

Check that theta is numeric.

Check that scale is numeric.

Check that score is numeric.

Check that no row is malformed.

------------------------------------------------------------------------

# Architecture Stability

The project treats the production path as a protected artifact.

Experimental changes should be evaluated separately.

A new selector should not be promoted solely because it looks plausible.

A new reranking rule should be measured.

A new threshold should be measured.

A new AI model should be compared against a fixed baseline.

A new preprocessing path should be checked for exact input changes.

This protects reproducibility.

------------------------------------------------------------------------

# Final Repository Intent

The repository serves two purposes.

The first purpose is to preserve the completed Phase 1 system.

The second purpose is to provide the Phase 2 submission implementation.

Phase 1 history explains the origin of the architecture.

Phase 2 demonstrates how the architecture was extended.

The README intentionally documents both.

This makes the repository understandable without relying on conversation
history.

------------------------------------------------------------------------

# End-to-End Summary

``` text
PHASE 1
Constrained pose
     |
     v
ICE
     |
     v
GSPE
     |
     v
Sub-pixel
     |
     v
SNRN
     |
     v
Fusion
     |
     v
(x,y)
```

``` text
PHASE 2
Unknown pose
     |
     v
ICE
     |
     v
Multi-scale / multi-rotation GSPE
     |
     v
Spatial Top-K candidates
     |
     v
Verification
     |
     v
x,y,theta,scale
     |
     v
Local refinement
     |
     v
Rejection + confidence
     |
     v
(x,y,theta,scale,found,score)
```

The architecture therefore evolves without discarding the original
foundation.

------------------------------------------------------------------------

# Final Project Statement

SilicoForge is a hybrid semiconductor localization framework.

Phase 1 established the classical-plus-learned localization
architecture.

Phase 2 extends that architecture to unknown pose and absent targets.
The system uses interpretable global search. The system preserves
multiple hypotheses. The system performs precise local verification. The
system supports sub-pixel localization. The system uses learned residual
refinement. The system performs confidence-aware decisions. The system
produces an evaluator-safe CSV. The system is designed for CPU-only
offline execution. The repository preserves the engineering history from
Phase 1 through Phase 2.
