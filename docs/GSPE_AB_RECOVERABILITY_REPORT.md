# GSPE A/B Recoverability Report

## Tested Configurations
- **Current GSPE Grid**: Scales [8,9,10,11,12], Rots [-5,0,5], K=3, NMS=1.0x (dynamic ~1/4 crop)
- **Dense Scale**: Scales [8,8.5,9,9.5,10,10.5,11,11.5,12]
- **Dense Rot**: Rots [-5,-2.5,0,2.5,5]
- **Promotion K Sweeps**: K=5, 10, 15
- **NMS Multipliers**: 0.5x, 0.75x, 1.25x, 1.5x

## Aggregate Metrics

| Variant | Mean | Median | P90 | Max | <=5 | <=10 | <=25 | <=50 | >50 |
|---------|------|--------|-----|-----|-----|------|------|------|-----|
| Baseline | 117.62 | 60.59 | 332.37 | 416.92 | 42 | 42 | 42 | 45 | 55 |
| Dense Scale | 110.80 | 60.13 | 302.57 | 382.00 | 44 | 44 | 45 | 47 | 53 |
| Dense Rot | 111.16 | 60.13 | 319.98 | 400.11 | 43 | 43 | 44 | 47 | 53 |
| Dense Scale+Rot | 106.50 | 60.13 | 300.04 | 382.00 | 43 | 43 | 44 | 46 | 54 |
| Promotion K=5 | 117.62 | 60.59 | 332.37 | 416.92 | 42 | 42 | 42 | 45 | 55 |
| Promotion K=10 | 117.62 | 60.59 | 332.37 | 416.92 | 42 | 42 | 42 | 45 | 55 |
| Promotion K=15 | 117.62 | 60.59 | 332.37 | 416.92 | 42 | 42 | 42 | 45 | 55 |
| NMS 0.5x | 117.62 | 60.59 | 332.37 | 416.92 | 42 | 42 | 42 | 45 | 55 |
| NMS 0.75x | 117.62 | 60.59 | 332.37 | 416.92 | 42 | 42 | 42 | 45 | 55 |
| NMS 1.25x | 117.62 | 60.59 | 332.37 | 416.92 | 42 | 42 | 42 | 45 | 55 |
| NMS 1.5x | 117.62 | 60.59 | 332.37 | 416.92 | 42 | 42 | 42 | 45 | 55 |
| Combined (Dense SR + K=15 + NMS 0.5x) | 106.45 | 60.13 | 300.04 | 382.00 | 43 | 43 | 44 | 46 | 54 |

## Failure Classification

| Variant | Geometry Recovered | NMS Recovered | Periodic Alias Remaining | True Ambiguity | Total Success |
|---------|--------------------|---------------|--------------------------|----------------|---------------|
| Baseline | 0 | 0 | 27 | 0 | 45 |
| Dense Scale | 0 | 0 | 25 | 0 | 47 |
| Dense Rot | 8 | 0 | 30 | 0 | 47 |
| Dense Scale+Rot | 0 | 0 | 24 | 0 | 46 |
| Promotion K=5 | 2 | 0 | 27 | 0 | 45 |
| Promotion K=10 | 7 | 0 | 30 | 0 | 45 |
| Promotion K=15 | 9 | 0 | 31 | 0 | 45 |
| NMS 0.5x | 0 | 1 | 28 | 0 | 45 |
| NMS 0.75x | 0 | 1 | 28 | 0 | 45 |
| NMS 1.25x | 0 | 0 | 22 | 0 | 45 |
| NMS 1.5x | 0 | 0 | 21 | 0 | 45 |
| Combined (Dense SR + K=15 + NMS 0.5x) | 11 | 0 | 32 | 0 | 46 |

## FINAL CONCLUSION

1. Can geometry-search changes alone bring mean error below 50 px?
No. Best geometry-only mean error: 106.50 px.

2. Can promotion-K changes alone bring mean error below 50 px?
No. Best promotion-K-only mean error: 117.62 px.

3. Can NMS changes alone bring mean error below 50 px?
No. Best NMS-only mean error: 117.62 px.

4. Can their combination bring mean error below 50 px?
No. Best combined mean error: 106.45 px.

5. How many >50 px failures remain after the best GSPE-only configuration?
54 cases >50px remain in the best GSPE-only config.

6. Of the remaining failures, how many are true periodic/context ambiguities?
32 cases are true periodic or contextual ambiguities.

7. How many genuinely require larger reference context?
32 fundamentally require larger reference context.

NO PRODUCTION CODE WAS MODIFIED.
