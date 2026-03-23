# AFC Phase B Residual-Usefulness Diagnostic

Date: 2026-03-23

Branch lineage:
- branch: `exp/afc-b-frozen-router`
- code commit for this check: `fd0f71f`

## Question

When rotor alignment is positive but RMSE still gets worse, is the router putting mass on specialists that are actually better than the anchor?

## Diagnostic added

Added regression fit-time diagnostics that compare specialist usefulness against the anchor on the router-validation split.

Definitions:
- `validation_best_specialist_advantage_score`
  - best available specialist improvement over the anchor on a row, averaged over rows
- `validation_weighted_specialist_advantage_score`
  - attention-weighted specialist advantage, averaged over rows
- `validation_defer_weighted_specialist_advantage_score`
  - the same quantity after multiplying by defer probability
- `validation_positive_specialist_mass`
  - fraction of normalized specialist attention landing on specialists with positive advantage
- `validation_top_specialist_positive_rate`
  - how often the top-attended specialist is better than the anchor

Advantage is measured with a bounded normalized absolute-error score:

```text
(anchor_abs_err - specialist_abs_err) / (anchor_abs_err + specialist_abs_err + 1e-6)
```

## Benchmark contract

Command:

```bash
export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export GRAPHDRONE_ENABLE_LEGITIMACY_GATE=0
export GRAPHDRONE_ROUTER_KIND=contextual_transformer_rotor
export GRAPHDRONE_ALIGNMENT_LAMBDA=0.01
export GRAPHDRONE_ROUTER_SEED=42
export GRAPHDRONE_FREEZE_BASE_ROUTER=1
export GRAPHDRONE_VERSION_OVERRIDE=afc-b-rotor-reg-l001-frozenrouter-residualquick-20260323
conda run --no-capture-output -n h200_tabpfn python scripts/run_champion_challenger.py \
  --task regression \
  --gate quick \
  --champion-version v1.20.0-reg-champion \
  --challenger-version afc-b-rotor-reg-l001-frozenrouter-residualquick \
  --output-dir eval/phaseb_frozen_router_residual_quick \
  --heartbeat-seconds 15
```

Artifacts:
- decision: `eval/phaseb_frozen_router_residual_quick/comparison/promotion_decision.json`
- claim report: `eval/phaseb_frozen_router_residual_quick/comparison/claim_report.json`
- paired deltas: `eval/phaseb_frozen_router_residual_quick/comparison/paired_task_deltas.csv`

## Result

Decision:
- `hold`

Quick summary:
- mean RMSE relative improvement: `-0.000478`
- mean R2 delta: `-0.000204`
- rotor claim status: `supported`
- rotor translation status: `not_translating`

## What the new diagnostic says

### `cpu_act`

Champion:
- best available specialist advantage: `+0.131469`
- weighted specialist advantage: `-0.111031`
- defer-weighted specialist advantage: `-0.004812`
- positive specialist mass: `0.357855`
- top-specialist positive rate: `0.3575`

Challenger:
- best available specialist advantage: `+0.131469`
- weighted specialist advantage: `-0.097987`
- defer-weighted specialist advantage: `-0.004270`
- positive specialist mass: `0.364845`
- top-specialist positive rate: `0.3950`
- alignment cosine gain: `+0.016125`

Outcome:
- challenger RMSE relative improvement: `-0.000956`
- challenger R2 delta: `-0.000407`

Read:
- useful specialists exist on the validation split
- both champion and challenger still allocate net harmful specialist mass on average
- rotor narrows that gap slightly on the validation split, but not enough to improve held-out task quality
- the challenger also collapses attention almost entirely onto `SUB1`, so the small usefulness improvement is not a robust routing policy

### `california`

Both champion and challenger:
- exact same RMSE and R2
- same defer level: `0.262026`
- residual-usefulness diagnostics are still `NaN`

Read:
- this remains a coverage hole
- the model changes policy somewhere, but the current diagnostic path still cannot explain the row-level routing behavior on this dataset

## Updated interpretation

This is the first direct evidence that GraphDrone's current regression router can have:
- positive available specialist value
- negative realized specialist value

That matters because it separates two questions:
1. Are specialists potentially helpful?
2. Is the router using them in a helpful way?

For `cpu_act`, the answer is now:
- yes, useful specialists are available
- no, the learned weighting still makes average specialist use harmful

Rotor improves geometry and slightly improves the validation usefulness score on `cpu_act`, but the test metric still gets worse. That points toward:
- objective mismatch between alignment and residual usefulness
- and/or validation-split overfitting in the routing policy

## Best current read

The remaining blocker is not just lack of geometric alignment.

It is more specifically:
- `specialist_value_exists_but_policy_realization_is_weak`

Rotor may help the geometry, but the current training objective still does not reliably turn that into a generalizing residual-allocation policy.

## Next checks

1. Run the same residual-usefulness diagnostic on the mini-full fold-0 regression contract.
2. Explain why `california` still yields `NaN` usefulness and attention diagnostics even when predictions are finite.
3. Test whether the router validation objective should directly reward positive specialist-advantage mass rather than only task loss plus alignment.
