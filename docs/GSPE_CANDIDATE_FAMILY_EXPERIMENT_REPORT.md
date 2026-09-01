# GSPE Candidate Family Experiment Report

## Baseline Metrics (Rank 0 only)
- Mean: 117.62
- Median: 60.59
- P90: 332.37
- Max: 416.92
- <= 5: 42
- <= 50: 45
- > 50: 55

## Candidate Family Variant Metrics
- Mean: 119.39
- Median: 59.90
- P90: 317.66
- Max: 400.11
- <= 1: 24
- <= 5: 43
- <= 10: 43
- <= 25: 43
- <= 50: 46
- > 50: 54

## Experiment Analysis
A. How many previous periodic-alias failures become correct? 4
B. How many previous successes regress? 3
C. Does candidate-family retention actually recover the GT when it was already present among strong peaks?
   - GT was present in the retained family in 36 failing cases.
   - Of those, 4 were successfully recovered by LowFreq tie-breaking.
D. How many failures are still caused by geometry? 18
E. How many remain fundamentally ambiguous? 19
F. What is the new mean localization error? 119.39 px
G. What is the runtime impact? Average GSPE runtime is 552.12 ms per case.

NO PRODUCTION CODE WAS MODIFIED.
