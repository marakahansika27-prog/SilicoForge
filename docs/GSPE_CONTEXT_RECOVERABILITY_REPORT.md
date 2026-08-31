# GSPE Context Recoverability Report

## 1. Accuracy by Context Size
| Context Size | Mean | Median | P90 | Max | <=1 | <=5 | <=10 | <=25 | <=50 | >50 |
|--------------|------|--------|-----|-----|-----|-----|------|------|------|-----|
| 1000 (100x100) | 117.62 | 60.59 | 332.37 | 416.92 | 23 | 42 | 42 | 42 | 45 | 55 |
| 1500 (150x150) | 99.67 | 61.71 | 252.13 | 390.23 | 18 | 42 | 42 | 42 | 46 | 54 |
| 2000 (200x200) | 99.82 | 59.63 | 297.30 | 399.61 | 13 | 44 | 44 | 47 | 47 | 53 |
| 2500 (250x250) | 97.29 | 35.94 | 269.52 | 526.18 | 8 | 46 | 46 | 46 | 51 | 49 |
| 3000 (300x300) | 94.74 | 29.63 | 275.33 | 612.82 | 7 | 37 | 46 | 49 | 51 | 49 |

## 2. Failure Recoveries and Regressions (Relative to 1000 Baseline)
| Context Size | Baseline Failures Recovered | Baseline Successes Regressed |
|--------------|-----------------------------|------------------------------|
| 1500 | 4 | 3 |
| 2000 | 5 | 3 |
| 2500 | 10 | 4 |
| 3000 | 10 | 4 |

## 3. Failure Classifications (>50px)
| Context Size | PERIODIC_ALIAS | TRUE_AMBIGUITY | GEOMETRY_COARSE_RANK_FAILURE | GEOMETRY_NOT_COVERED | NMS_FAILURE | SCORING_FAILURE | OTHER |
|--------------|----------------|----------------|------------------------------|----------------------|-------------|-----------------|-------|
| 1000 | 27 | 0 | 9 | 10 | 7 | 2 | 0 |
| 1500 | 19 | 0 | 9 | 10 | 16 | 0 | 0 |
| 2000 | 14 | 0 | 7 | 10 | 21 | 1 | 0 |
| 2500 | 7 | 0 | 8 | 9 | 25 | 0 | 0 |
| 3000 | 4 | 0 | 8 | 10 | 27 | 0 | 0 |

## 4. Runtime per case (ms)
| Context Size | Mean Runtime |
|--------------|--------------|
| 1000 | 531.21 |
| 1500 | 514.21 |
| 2000 | 547.48 |
| 2500 | 616.48 |
| 3000 | 667.06 |

## Conclusion
Evaluate whether larger context successfully disambiguates periodic aliases without geometry modifications.
