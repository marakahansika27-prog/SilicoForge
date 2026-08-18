# Drift-Sense V2 — Final Validated Architecture (Frozen Champion)

This document defines the strictly validated, mathematically proven production architecture for Drift-Sense V2. Any module not explicitly listed here has been mathematically audited and formally rejected under controlled ablation.

## Core Processing Pipeline

```
INPUT VALIDATION
(Validates NaN, Inf, empty bounds)
        ↓
10× INTER_AREA SCALE LOCK
(Reference downsampled physically to match Search scale)
        ↓
RAW ZNCC
(Spatial cv2.matchTemplate CCOEFF_NORMED)
        +
LOW-FREQUENCY ZNCC
(Gaussian Blur sigma=15, 31x31 kernel → ZNCC)
        ↓
HYBRID SCORE
(0.5 × Raw + 0.5 × LowFreq)
        ↓
GSPE TOP-1
(Global integer peak via minMaxLoc)
        ↓
3×3 SUBPIXEL PARABOLIC REFINEMENT
(Deterministic fractional interpolation on Hybrid ZNCC surface)
        ↓
GSPE-ONLY FINAL COORDINATE
(Classical subpixel localization)
```

## Conditional Robustness Modules

The following module is considered a strict safety fallback and is **NOT** part of the always-on Champion pipeline:

```
SHEAR ESTIMATION
    ↓
if shear ≤ 0.5°
    → no correction (Run Frozen Champion)
if shear > 0.5°
    → conditional affine correction (De-shear → Champion → Re-warp)
```

## Bypassed / Rejected Modules

The following architectural components have been permanently deactivated or rejected because they fundamentally harmed localization precision or added zero mathematical value:

- **SRAE (Spatial Registration Alignment Engine):** Bypassed (Failed due to periodic cell snapping).
- **SNRN (Subpixel Neural Refinement Network):** Bypassed (Failed due to noise-chasing and lack of macro context).
- **Decision Fusion:** Bypassed (Unnecessary under GSPE-Only mode).
- **FFT-ZNCC:** Rejected (No numerical benefit over OpenCV spatial, 2x slower).
- **Local ECC Refinement:** Rejected (Worsened subpixel error by tracking Poisson jitter).
- **Blind Poisson-Gaussian NLF & Anscombe VST:** Rejected (Squashed signal contrast, flattening correlation peaks).
- **Lattice-Aware NMS & Deterministic Tie-Break:** Rejected (Mathematical non-issue under Hybrid scoring).
- **Explicit LF/HF Decomposition:** Rejected (Lost the subpixel noise-regularization blend).
