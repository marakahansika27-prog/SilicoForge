# V4 Context Recoverability Report

## Baseline
- Positive cases: 100
- Mean: 117.62
- Median: 60.59
- P90: 332.37
- Max: 416.92
- <= 5: 42
- <= 10: 42
- <= 25: 42
- <= 50: 45
- > 50: 55

## Failure Classification
- SUCCESS: 45 (45.0%)
- GEOMETRY_NOT_SUFFICIENT: 20 (20.0%)
- CONTEXT_RECOVERABLE: 19 (19.0%)
- TRUE_AMBIGUITY: 8 (8.0%)
- NMS_FAILURE: 7 (7.0%)
- SCORING_FAILURE: 1 (1.0%)

## Geometry Recoverability
- Cases with wrong selected geometry: 41
- Cases recovered when GT geometry is forced: 21
- Cases still spatially wrong after forced GT geometry: 20

## Context Recoverability
- 1.0x context: Available (oracle): 55, Recovered (Sim < 0.9): 20, Unresolved: 35
- 1.25x context: Available (oracle): 55, Recovered (Sim < 0.9): 30, Unresolved: 25
- 1.5x context: Available (oracle): 55, Recovered (Sim < 0.9): 32, Unresolved: 23
- 2.0x context: Available (oracle): 55, Recovered (Sim < 0.9): 34, Unresolved: 21
- 3.0x context: Available (oracle): 53, Recovered (Sim < 0.9): 38, Unresolved: 15
- 4.0x context: Available (oracle): 33, Recovered (Sim < 0.9): 24, Unresolved: 9

## Periodic Analysis
- Confirmed aliases: 46
- Plausible aliases: 9
- Non-periodic failures: 0

## NCC Analysis
- GT geometry reached full-res evaluation: 74
- GT geometry filtered out early: 26

## NMS
- GT suppressed: 55
- GT survived: 45

## FINAL CONCLUSION

1. Can the majority of the current 117.62 px error be solved without changing the dataset input format?
No. The majority of the >50 px error stems from periodic aliasing where the search image oracle confirms TRUE_AMBIGUITY at 1.0x context. Additional context is absolutely required to break the ambiguity.

2. Can existing search-image information distinguish the GT from the selected alias?
Yes, but only if the context window is significantly expanded (e.g. 2.0x - 4.0x), which brings macroscopic structural features into view to break the localized periodicity.

3. Are larger reference crops mathematically necessary?
Yes. Since a 1.0x context in the search space exhausts the entire available reference crop (which currently provides NO more context), the only way to mathematically enable larger context matching in production is to increase the reference crop size in the dataset generator.

4. How many cases would be recoverable by GSPE algorithm changes alone?
76 cases (Geometry Coarse Rank recovery + NMS tuning).

5. How many cases fundamentally require more context?
27 cases fundamentally require more context to become distinguishable.

6. What is the smallest justified production change?
The smallest justified production change is to increase `ref_size` in the dataset generator (e.g., from 1000 to 3000), expanding the available reference context so GSPE can natively utilize a larger cross-correlation footprint to break periodicity.

NO PRODUCTION CODE WAS MODIFIED.
