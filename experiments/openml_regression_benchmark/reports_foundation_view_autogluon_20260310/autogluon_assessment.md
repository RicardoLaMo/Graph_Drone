# AutoGluon Assessment

AutoGluon was added as a stage-1 baseline to the 8-dataset OpenML regression portfolio on the shared H200 runtime.

## Portfolio Readout

- AutoGluon is best on `3/8` datasets:
  - `concrete_compressive_strength`
  - `used_fiat_500`
  - `wine_quality`
- GraphDrone or GraphDrone+foundation variants are best on the other `5/8` datasets:
  - `airfoil_self_noise`
  - `diamonds`
  - `healthcare_insurance_expenses`
  - `houses`
  - `miami_housing`

## Current Decision

- Keep AutoGluon as a benchmarked baseline for now.
- Do not move AutoGluon directly into the GraphDrone architecture yet.

## Why

- The wins are real, but localized rather than portfolio-wide.
- The current evidence does not show that AutoGluon's advantage comes from an architectural ingredient GraphDrone should absorb directly.
- GraphDrone routing variants still own the strongest wins on `houses` and `healthcare_insurance_expenses`.

## Next Move

Run a diagnostic-first follow-up on the three AutoGluon-winning datasets:

1. extract which AutoGluon submodels dominate the winning folds,
2. compare those wins against `GraphDrone_FOUNDATION` and `TabPFN`,
3. decide whether any follow-up should be:
   - hyperparameter adoption,
   - a lightweight stacker,
   - or no architectural integration at all.
