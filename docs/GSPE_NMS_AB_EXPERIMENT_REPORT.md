# GSPE NMS A/B Experiment Report (3000 Context)

## 1. Accuracy by NMS Multiplier
| NMS Multiplier | Mean | Median | P90 | Max | <=1 | <=5 | <=10 | <=25 | <=50 | >50 |
|----------------|------|--------|-----|-----|-----|-----|------|------|------|-----|
| 0.25x | 94.74 | 29.63 | 275.33 | 612.82 | 7 | 37 | 46 | 49 | 51 | 49 |
| 0.5x | 94.74 | 29.63 | 275.33 | 612.82 | 7 | 37 | 46 | 49 | 51 | 49 |
| 0.75x | 94.74 | 29.63 | 275.33 | 612.82 | 7 | 37 | 46 | 49 | 51 | 49 |
| 1.0x | 94.74 | 29.63 | 275.33 | 612.82 | 7 | 37 | 46 | 49 | 51 | 49 |
| 1.25x | 94.74 | 29.63 | 275.33 | 612.82 | 7 | 37 | 46 | 49 | 51 | 49 |
| 1.5x | 94.74 | 29.63 | 275.33 | 612.82 | 7 | 37 | 46 | 49 | 51 | 49 |

## 2. Failure Recoveries and Regressions (Relative to 1.0x Baseline)
| NMS Multiplier | 1.0x Failures Recovered | 1.0x Successes Regressed |
|----------------|-------------------------|--------------------------|
| 0.25x | 0 | 0 |
| 0.5x | 0 | 0 |
| 0.75x | 0 | 0 |
| 1.0x | 0 | 0 |
| 1.25x | 0 | 0 |
| 1.5x | 0 | 0 |

## 3. Failure Classifications (>50px)
| NMS Multiplier | NMS_FAILURE (GT Suppressed) | PERIODIC_ALIAS | TRUE_AMBIGUITY | GEOMETRY_COARSE_RANK_FAILURE | GEOMETRY_NOT_COVERED | SCORING_FAILURE | OTHER |
|----------------|-----------------------------|----------------|----------------|------------------------------|----------------------|-----------------|-------|
| 0.25x | 9 | 14 | 0 | 8 | 10 | 8 | 0 |
| 0.5x | 19 | 11 | 0 | 8 | 10 | 1 | 0 |
| 0.75x | 20 | 10 | 0 | 8 | 10 | 1 | 0 |
| 1.0x | 27 | 4 | 0 | 8 | 10 | 0 | 0 |
| 1.25x | 27 | 4 | 0 | 8 | 10 | 0 | 0 |
| 1.5x | 27 | 4 | 0 | 8 | 10 | 0 | 0 |

## 4. Runtime per case (ms)
| NMS Multiplier | Mean Runtime |
|----------------|--------------|
| 0.25x | 313.51 |
| 0.5x | 298.50 |
| 0.75x | 299.11 |
| 1.0x | 296.61 |
| 1.25x | 299.24 |
| 1.5x | 304.53 |

## Conclusion
Evaluate whether adjusting NMS suppression size meaningfully recovers remaining periodic aliases after context expansion, without triggering spatial regressions.
