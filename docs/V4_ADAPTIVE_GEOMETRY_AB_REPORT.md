# V4 Adaptive Geometry A/B Report

## 1. Accuracy by Configuration
| Configuration | Mean | Median | P90 | Max | <=1 | <=5 | <=10 | <=25 | <=50 | >50 | Runtime(ms) |
|---------------|------|--------|-----|-----|-----|-----|------|------|------|-----|-------------|
| Baseline | 94.72 | 29.92 | 275.79 | 613.01 | 5 | 37 | 46 | 49 | 51 | 49 | 644.3 |
| Adaptive_K5_Small | 80.69 | 3.30 | 235.22 | 319.06 | 17 | 51 | 52 | 55 | 55 | 45 | 5639.6 |
| Adaptive_K10_Small | 80.69 | 3.30 | 235.22 | 319.06 | 17 | 51 | 52 | 55 | 55 | 45 | 349891.4 |
| Adaptive_K10_Medium | 77.63 | 5.92 | 235.22 | 319.06 | 27 | 50 | 51 | 54 | 55 | 45 | 33879.6 |

## 2. Failure Classifications (Relative to Baseline)
| Configuration | Recoveries | Regressions | GEOMETRY_STILL_WRONG | PERIODIC_ALIAS | OTHER |
|---------------|------------|-------------|----------------------|----------------|-------|
| Baseline | 0 | 0 | - | - | - |
| Adaptive_K5_Small | 6 | 2 | 0 | 45 | 0 |
| Adaptive_K10_Small | 6 | 2 | 0 | 45 | 0 |
| Adaptive_K10_Medium | 7 | 3 | 0 | 45 | 0 |
