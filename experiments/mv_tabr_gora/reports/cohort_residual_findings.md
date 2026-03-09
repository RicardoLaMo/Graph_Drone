# Cohort Residual Findings

## What was tested

This branch tested `G0` as a combined intervention on top of `A6f`:

- target residualisation by train-only KMeans-96 geo-segment mean
- `GEO` view augmentation with the same segment mean as an extra feature
- unchanged `A6f` architecture and unchanged default Huber training loss

Important note:

- the proposed `G1` Huber variant is already the baseline in this codebase
- so the actual new change in this branch is `G0`

## Results

### Smoke

- `A6f_raw`: val `0.8292`, test `0.7908`
- `G0_cohort_residual`: val `0.7155`, test `0.6632`

### Full

- `A6f_raw`: val `0.4333`, test `0.4324`
- `G0_cohort_residual`: val `0.5454`, test `0.5329`

## Interpretation

The smoke result was strongly positive, but the full result failed badly.

That means the current *combined* `G0` design is not a valid improvement over
`A6f`.

The most likely reason is not "cohort centering is useless." The branch changed
two things at once:

1. the target definition (`y - segment_mean`)
2. the `GEO` view retrieval geometry by appending a target-derived segment mean

So the negative result should be interpreted as:

- the combined intervention is too aggressive
- not as a clean rejection of cohort-residual targets by themselves

Gemini cross-check added one more likely issue:

3. residual targets were still divided by the original global target standard
   deviation, which may shrink the effective target scale too much and flatten
   optimization

## Strongest next split

If this line continues, split the hypothesis into two orthogonal checks:

- `G0a`: residual target only, no segment-mean feature appended to any view
- `G0b`: segment-mean feature appended to `GEO`, but keep the original target
- `G0c`: residual target only, but normalise by train residual std rather than
  the original global target std

That is the only clean way to tell whether the failure came from:

- residual target centering
- target-derived retrieval geometry
- or their interaction
