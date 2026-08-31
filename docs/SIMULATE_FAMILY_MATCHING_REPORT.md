# GSPE TEMPLATE-FAMILY SIMULATION REPORT

## 1. Summary of Recoverability
- **SUCCESS**: 9
- **TRUE_AMBIGUITY**: 11
- **CONTEXT_RECOVERABLE**: 14
- **SCORING_FAILURE**: 0
- **GEOMETRY_FAILURE**: 18
- **NMS_FAILURE**: 2
- **NOT_IN_TOP_FAMILY**: 6

## 2. Explanation of Categories
- **SUCCESS**: GT was correctly ranked #1 by both Raw and Hybrid (or would be if geometry was right).
- **SCORING_FAILURE**: GT was #1 in Raw NCC, but the Hybrid NCC (Gaussian blur) pulled the peak away to a periodic alias near the macro boundary.
- **CONTEXT_RECOVERABLE**: GT was tied in Raw NCC with periodic aliases, but evaluating 2.0x context correctly isolated the GT (NCC drops <0.90 for the alias).
- **TRUE_AMBIGUITY**: GT and the periodic alias are mathematically identical even at 2.0x context (NCC > 0.90). No algorithm can distinguish them from the pixels alone.
- **GEOMETRY_FAILURE**: Wrong scale or rotation was promoted by coarse search.
- **NMS_FAILURE**: GT peak was suppressed by NMS radius.
