# DRIFT-SENSE V2 — FINAL V4 OFFLINE EVALUATION

## Dataset

- Total pairs: **200**
- Target-present pairs: **100**
- Target-absent pairs: **100**

## Overall Localization Accuracy

- Mean error: **129.2846 px**
- Median error: **59.2028 px**
- Maximum error: **1174.1407 px**
- Within 1 px: **35.00%**
- Within 2 px: **44.00%**
- Within 5 px: **46.00%**
- Within 10 px: **46.00%**
- Within 25 px: **47.00%**
- Within 50 px: **48.00%**

## Architecture Breakdown

| Architecture | Samples | Mean Error | <2px | <10px | <50px |
|---|---:|---:|---:|---:|---:|
| DRAM | 50 | 169.1079 | 34.00% | 38.00% | 38.00% |
| FinFET | 50 | 89.4614 | 54.00% | 54.00% | 58.00% |

## Difficulty Breakdown

| Difficulty | Samples | Mean Error | <2px | <10px |
|---|---:|---:|---:|---:|
| easy | 36 | 138.6482 | 38.89% | 38.89% |
| hard | 37 | 124.4773 | 48.65% | 54.05% |
| moderate | 27 | 123.3876 | 44.44% | 44.44% |

## Best 10 Present Cases

| Pair | Architecture | GT | Prediction | Error |
|---|---|---|---|---:|
| case_v4_finfet_present_008 | FinFET | (786.90, 232.70) | (786.96, 232.74) | 0.0729 |
| case_v4_finfet_present_027 | FinFET | (784.00, 764.10) | (783.89, 764.18) | 0.1306 |
| case_v4_finfet_present_049 | FinFET | (814.80, 225.40) | (814.68, 225.28) | 0.1645 |
| case_v4_finfet_present_014 | FinFET | (220.40, 240.00) | (220.25, 240.14) | 0.2013 |
| case_v4_finfet_present_039 | FinFET | (159.00, 158.10) | (159.15, 158.26) | 0.2207 |
| case_v4_dram_present_024 | DRAM | (778.40, 196.90) | (778.37, 196.64) | 0.2636 |
| case_v4_finfet_present_001 | FinFET | (774.50, 174.60) | (774.78, 174.53) | 0.2828 |
| case_v4_finfet_present_030 | FinFET | (162.60, 827.70) | (162.39, 827.90) | 0.2910 |
| case_v4_dram_present_028 | DRAM | (792.00, 215.30) | (792.18, 215.58) | 0.3357 |
| case_v4_finfet_present_037 | FinFET | (802.40, 840.40) | (802.41, 840.77) | 0.3717 |

## Worst 10 Present Cases

| Pair | Architecture | GT | Prediction | Error |
|---|---|---|---|---:|
| case_v4_dram_present_031 | DRAM | (845.50, 814.70) | (0.00, 0.00) | 1174.1407 |
| case_v4_dram_present_020 | DRAM | (124.00, 515.10) | (0.00, 0.00) | 529.8151 |
| case_v4_finfet_present_044 | FinFET | (382.20, 842.00) | (782.13, 841.76) | 399.9350 |
| case_v4_dram_present_025 | DRAM | (194.10, 645.30) | (193.80, 255.52) | 389.7765 |
| case_v4_finfet_present_003 | FinFET | (645.90, 353.10) | (785.83, 713.18) | 386.3130 |
| case_v4_dram_present_019 | DRAM | (517.80, 503.40) | (247.62, 773.66) | 382.1483 |
| case_v4_dram_present_005 | DRAM | (534.90, 492.40) | (264.77, 222.48) | 381.8704 |
| case_v4_dram_present_014 | DRAM | (601.40, 424.90) | (751.26, 754.99) | 362.5128 |
| case_v4_dram_present_032 | DRAM | (505.00, 500.80) | (774.82, 741.07) | 361.2973 |
| case_v4_finfet_present_033 | FinFET | (608.00, 760.00) | (248.07, 759.91) | 359.9318 |

## Interpretation

**ACCURACY_THRESHOLD_FAIL**

The majority of target-present cases remain outside the strict 2 px localization threshold.
