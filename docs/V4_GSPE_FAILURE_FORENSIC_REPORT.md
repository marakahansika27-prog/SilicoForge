# V4 GSPE Failure Forensic Report

## Aggregate Statistics
- Positive Cases: 100
- Mean Error: 117.62 px
- Median Error: 60.59 px
- P90 Error: 332.37 px
- Max Error: 416.92 px
- <= 1 px: 23 (23.0%)
- <= 5 px: 42 (42.0%)
- <= 10 px: 42 (42.0%)
- <= 25 px: 42 (42.0%)
- <= 50 px: 45 (45.0%)
- > 50 px: 55 (55.0%)

## Failure Classification
- PERIODIC_ALIAS: 28 (28.0%)
- GEOMETRY_COARSE_RANK_FAILURE: 9 (9.0%)
- SUCCESS: 45 (45.0%)
- GEOMETRY_NOT_COVERED: 10 (10.0%)
- NMS_FAILURE: 7 (7.0%)
- GEOMETRY_PROMOTION_FAILURE: 1 (1.0%)

## Geometry Analysis
- Coarse Stage Failures (GT didn't reach top-3): 26

## Periodicity Analysis
- Periodic Aliases (Lost to identical structural peak at distance % pitch == 0): 28

## NMS Analysis
- GT Suppressed by NMS: 7

## Root Cause
If PERIODIC_ALIAS is the dominant category, the root cause is that the 10x100x100 context window is completely insufficient to break local periodicity inside dense arrays (DRAM/FinFET), resulting in mathematically identical peaks. GSPE is forced to guess randomly among these identical peaks.

## Smallest Justified Fix
The smallest mathematically justified fix is to increase the Phase 2 dataset's `reference_img` context. The underlying `base_img_A` has exactly the same macro-structure, but GSPE needs to 'see' more of it to break local periodic ambiguity. We must increase the spatial context of the reference crop.

NO PRODUCTION CODE WAS MODIFIED.
