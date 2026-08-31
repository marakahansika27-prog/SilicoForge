# GSPE PERIODIC MATCHING FORENSIC REPORT

## 1. Summary Statistics
- Mean Error: 196.22 px
- Median Error: 196.63 px
- Max Error: 510.99 px

## 2. Periodic Alias Test
- Confirmed Periodic Aliases (<5px pitch err): 44
- Plausible Periodic Aliases (<15px pitch err): 1
- Not Periodic (>15px pitch err): 0

## 3. Informational Ambiguity Test
- True Informational Ambiguity: 11
- Score Function Destroys Unique Info: 1
- Context Starvation (2x context fixes it): 16
- Context Does Not Solve: 18

## 4. Final Root-Cause Classification
**PRIMARY ROOT CAUSE:** PERIODIC_INFORMATION_AMBIGUITY

**SECONDARY ROOT CAUSE:** UNKNOWN

**AFFECTED CASES:** 44 periodic cases > 50px

**SMALLEST MATHEMATICALLY JUSTIFIED CHANGE:** Investigate multi-context/template-family matching.

