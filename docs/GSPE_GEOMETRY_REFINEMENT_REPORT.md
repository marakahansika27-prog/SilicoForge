# GSPE Geometry Refinement Report

## 1. Configuration Table
- **Baseline**: Scales [8,9,10,11,12], Rots [-5,0,5]
- **Exp A (Scale Refine)**: Scale step 0.25
- **Exp B (Rot Refine)**: Rot step 1.25
- **Exp C (Combined Dense)**: Scale step 0.25, Rot step 1.25
- **Exp D (Local Refine)**: Coarse top-3 -> local grid (scale +-0.5 step 0.25, rot +-2.5 step 1.25) -> top-3 full res

## 2. Accuracy Table
| Variant | Mean | Median | P90 | Max | <=1 | <=5 | <=10 | <=25 | <=50 | >50 |
|---------|------|--------|-----|-----|-----|-----|------|------|------|-----|
| Baseline | 117.62 | 60.59 | 332.37 | 416.92 | 23 | 42 | 42 | 42 | 45 | 55 |
| Exp A (Scale Refine) | 116.54 | 60.58 | 329.62 | 400.32 | 26 | 43 | 43 | 43 | 45 | 55 |
| Exp B (Rot Refine) | 119.10 | 89.92 | 302.37 | 400.41 | 28 | 42 | 42 | 43 | 44 | 56 |
| Exp C (Combined Dense) | 103.36 | 60.19 | 268.52 | 381.86 | 33 | 43 | 43 | 44 | 46 | 54 |
| Exp D (Local Refine) | 103.40 | 60.19 | 268.52 | 381.86 | 32 | 43 | 43 | 44 | 46 | 54 |

## 3. Geometry Recovery & Classification
| Variant | Geom Improved & Loc Improved | Geom Improved, Loc Not | Geom Unchanged | Geom Regressed | Spatial Recovery Without Geom Change |
|---------|------------------------------|------------------------|----------------|----------------|--------------------------------------|
| Exp A (Scale Refine) | 12 | 13 | 32 | 37 | 6 |
| Exp B (Rot Refine) | 32 | 14 | 25 | 17 | 12 |
| Exp C (Combined Dense) | 35 | 22 | 11 | 28 | 4 |
| Exp D (Local Refine) | 34 | 22 | 11 | 29 | 4 |

## 4. Per-Case Regression/Recovery Counts
| Variant | Recovered (>50 to <=50) | Recovered to <=5 | Regressed (<=50 to >50) | Loc Regressed (>1px) | Loc Improved (>1px) | GT Geom Recovered |
|---------|-------------------------|------------------|-------------------------|----------------------|---------------------|-------------------|
| Exp A (Scale Refine) | 3 | 2 | 3 | 14 | 16 | 0 |
| Exp B (Rot Refine) | 2 | 1 | 3 | 13 | 13 | 10 |
| Exp C (Combined Dense) | 5 | 2 | 4 | 19 | 31 | 7 |
| Exp D (Local Refine) | 5 | 2 | 4 | 19 | 30 | 7 |

## 5. Runtime Table
| Variant | Mean Runtime (ms/case) |
|---------|------------------------|
| Baseline | 545.67 |
| Exp A (Scale Refine) | 992.59 |
| Exp B (Rot Refine) | 917.76 |
| Exp C (Combined Dense) | 2224.37 |
| Exp D (Local Refine) | 1328.62 |

## 6. Analysis of periodic-alias interaction
If 'Geom Improved, Loc Not' is high, it means refining geometry does NOT resolve periodic aliasing. Instead, the refined search simply latches onto a mathematically identical periodic peak with the new optimal geometry. If 'Recovered (>50 to <=50)' is very low compared to baseline failures, then geometry quantization is NOT the root cause of large >50px errors.

## 7. Recommended smallest mathematically justified refinement strategy
Based on the decision rule:
If Exp D (Local Refine) runtime is <1000ms (within 5s limit) and recovers the localization improvements of Exp C (Combined Dense), it represents the optimal balance of efficiency and continuous geometry search.
However, if the total successful cases (<=50px) across ALL configurations remains stubbornly around 45-50%, then geometry refinement fundamentally CANNOT resolve the remaining 50% periodic alias failures. In that scenario, geometry refinement provides marginal local benefit (<=5px improvements) but fails to break macroscopic ambiguity, requiring us to return to spatial context expansion.

NO PRODUCTION CODE WAS MODIFIED.
