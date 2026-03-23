# AFC Phase B California Router-Instability Check

Date: 2026-03-23

Branch lineage:
- branch: `exp/afc-b-frozen-router`
- code commit for local probe: `c2ec1ba`
- code state for explicit fallback patch: working tree after `c2ec1ba`, before next commit

## Question

Why did `california` report null attention and usefulness diagnostics while still producing finite predictions?

## What was checked

1. Direct observer check:
   - `kappa` and `lid` emit divide warnings, but their returned arrays are finite on `california`
2. Focused model probe on GPU:
   - tokens remain finite
   - router weights become non-finite after regression router training
   - defer probabilities stay finite
   - integrated predictions stay finite only because the effective defer path collapses
3. Safety patch:
   - detect non-finite regression-router training
   - preserve validation usefulness diagnostics
   - force explicit anchor-only fallback at prediction time
   - expose `effective_defer_rate=0.0` and `router_nonfinite_fallback=1`

## Key local probe result

Before the explicit fallback patch:
- `token_finite = True`
- `weights_finite = False`
- `defer_finite = True`
- `mean_defer_prob = 0.2620`
- `effective_defer_rate = 0.0`

That means the problem is:
- not non-finite tokens
- not non-finite defer logits
- specifically non-finite learned attention weights after router training

## Benchmark artifact after the fix

Command:

```bash
export CUDA_VISIBLE_DEVICES=5
export GRAPHDRONE_PRESET=v1_20_champion
export GRAPHDRONE_VERSION_OVERRIDE=california-router-stability-fix-v2-20260323
conda run --no-capture-output -n h200_tabpfn python scripts/run_geopoe_benchmark.py \
  --datasets california \
  --folds 0 \
  --cache-dir eval/california_router_stability_fix_v2/cache \
  --output-dir eval/california_router_stability_fix_v2/report \
  --max-samples 5000 \
  --methods graphdrone
```

Artifacts:
- cache row: `eval/california_router_stability_fix_v2/cache/california__fold0__graphdrone.json`
- granular report: `eval/california_router_stability_fix_v2/report/results_granular.csv`

Observed row:
- `rmse = 20.8550`
- `r2 = 0.7436`
- `router_kind = router_training_nonfinite_anchor_only`
- `mean_defer_prob = 0.0`
- `effective_defer_rate = 0.0`
- `router_nonfinite_fallback = 1`
- `validation_best_specialist_advantage_score = +0.0213`
- `validation_weighted_specialist_advantage_score = -0.1884`
- `validation_positive_specialist_mass = 0.2963`

## Interpretation

This clears the numerical ambiguity on `california`.

What is true:
1. The regression router training path can become numerically unstable on finite tokens.
2. The old behavior hid that by surfacing null diagnostics while predictions effectively fell back toward the anchor.
3. The explicit fallback patch preserves the prior task quality while making the failure mode auditably visible.

What it also reveals:
- even on the validation split, specialist value is weak on `california`
- best available specialist advantage is only slightly positive
- realized weighted specialist advantage is strongly negative

So `california` is not just a numerical bug.
It is a numerical bug plus a bad routing signal.

## Best current read

`california` is now best understood as:
- `nonfinite_router_training` plus
- `negative_realized_specialist_value`

The safety fallback fixes the first problem.
It does not solve the second.

## Next checks

1. Rerun the regression mini-full contract on the patched SHA so `california` no longer distorts the aggregate results through silent null diagnostics.
2. Count how many datasets hit `router_training_nonfinite_anchor_only`.
3. Decide whether negative validation-weighted specialist value should itself trigger a conservative anchor-only fallback.
