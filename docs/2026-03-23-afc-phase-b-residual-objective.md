# AFC Phase B Residual-Usefulness Objective

Date: 2026-03-23

Branch lineage:
- branch: `exp/afc-b-residual-objective`
- code commit for this check: `8f7d30e`

## Question

If the main remaining Phase B blocker is negative realized specialist value, does adding an explicit regression objective term for residual-usefulness gap improve translation better than the current `mse + residual penalty + alignment` stack?

## Change

Added a configurable regression-only auxiliary term:

```text
residual_usefulness_gap = relu(positive_specialist_opportunity - realized_specialist_advantage)
```

where:
- `positive_specialist_opportunity`
  is the positive part of the best available specialist advantage over the anchor
- `realized_specialist_advantage`
  is the defer-weighted attention-weighted specialist advantage actually realized by the router

Training objective becomes:

```text
mse + 2 * relu(mse - anchor_mse) + alignment_aux + lambda_usefulness * residual_usefulness_gap
```

This objective is off by default and is enabled by:
- `GRAPHDRONE_RESIDUAL_USEFULNESS_LAMBDA`

## Benchmark contract

This was a diagnostic-grade quick regression contract, not merge-grade evidence.

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
export GRAPHDRONE_RESIDUAL_USEFULNESS_LAMBDA=0.1
export GRAPHDRONE_ROUTER_SEED=42
export GRAPHDRONE_FREEZE_BASE_ROUTER=1
export GRAPHDRONE_VERSION_OVERRIDE=afc-b-rotor-reg-l001-frozenrouter-residualobj-l01-v2-20260323
conda run --no-capture-output -n h200_tabpfn python scripts/run_champion_challenger.py \
  --task regression \
  --gate quick \
  --champion-version v1.20.0-reg-champion \
  --challenger-version afc-b-rotor-reg-l001-frozenrouter-residualobj-l01-v2 \
  --output-dir eval/phaseb_residual_objective_quick_l01_v2 \
  --heartbeat-seconds 15
```

Artifacts:
- decision: `eval/phaseb_residual_objective_quick_l01_v2/comparison/promotion_decision.json`
- claim report: `eval/phaseb_residual_objective_quick_l01_v2/comparison/claim_report.json`
- paired deltas: `eval/phaseb_residual_objective_quick_l01_v2/comparison/paired_task_deltas.csv`
- champion ledger: `eval/phaseb_residual_objective_quick_l01_v2/raw/champion/regression/run_ledger.json`
- challenger ledger: `eval/phaseb_residual_objective_quick_l01_v2/raw/challenger/regression/run_ledger.json`

## Result

Decision:
- `hold`

Quick summary:
- mean RMSE relative improvement: `-0.000341`
- mean R2 delta: `-0.000145`
- mean latency improvement: `-10.0%`
- rotor claim status: `supported`
- rotor translation status: `not_translating`

## What changed relative to the prior residual-usefulness quick probe

Compared against the prior `exp/afc-b-frozen-router` quick probe:

### `cpu_act`

Old rotor challenger:
- RMSE relative improvement: `-0.000956`
- weighted specialist advantage: `-0.097987`
- defer-weighted specialist advantage: `-0.004270`
- positive specialist mass: `0.364845`
- defer: `0.01985`

Residual-objective challenger:
- RMSE relative improvement: `-0.000682`
- weighted specialist advantage: `-0.109587`
- defer-weighted specialist advantage: `-0.000239`
- positive specialist mass: `0.354765`
- residual-usefulness gap: `0.207439`
- defer: `0.00170`

Read:
- held-out RMSE got slightly less bad
- realized negative specialist value shrank sharply
- but this happened mainly because defer collapsed almost to zero
- the underlying weighted specialist advantage did not improve

### `california`

Residual-objective challenger:
- same held-out RMSE and R2 as champion
- router still hits non-finite gradients in the frozen-base pre-router stage
- fallback remains explicit anchor-only behavior
- weighted specialist advantage improved from the earlier opaque path to `-0.1642`
- residual-usefulness gap now surfaces as `0.2236`

Read:
- the new objective does make the failure mode more measurable
- but it does not solve the numerical instability or create usable specialist routing on this dataset

## Interpretation

This objective moved the diagnosed statistic in the intended direction, but not in the way we ultimately want.

The best current read is:
- the residual-usefulness penalty is not inert
- but at `lambda=0.1` it mostly learns “route less” rather than “route better”

That is still useful research signal because it narrows the failure mode:
- plain alignment was too indirect
- usefulness-gap regularization is more targeted
- but the present formulation still allows the model to win the auxiliary term by suppressing defer instead of learning a stronger specialist-allocation policy

## Why this matters

This is another point in favor of the broader hypothesis that single-dataset router fitting is too weak.

If local objectives keep improving internal diagnostics mainly by collapsing routing rather than learning a reusable allocation policy, the next-scale solution is probably not “more per-dataset fitting.”

A better next-scale direction is:
- cross-dataset latent manifold alignment for routing priors
- each dataset contributes view tokens as task-level objects
- a transformer or hyper-router learns reusable view-alignment structure across tasks
- per-dataset routing is conditioned by that prior rather than learned from a tiny validation split from scratch

That idea is not activated in this branch yet. It remains a next framework hypothesis, not the current experiment target.

## Next checks

1. Sweep `GRAPHDRONE_RESIDUAL_USEFULNESS_LAMBDA` over a small set such as `{0.02, 0.05, 0.1}`.
2. If the objective keeps helping mainly by reducing defer, redesign the auxiliary term so it rewards selective positive specialist allocation rather than simple routing suppression.
3. If local objectives keep stalling, open the next branch for cross-dataset latent manifold alignment / hyper-LMA routing priors.

## Lambda sweep follow-up

After the initial quick probe at `lambda=0.1`, the same quick regression contract was rerun with:
- `lambda=0.02`
- `lambda=0.05`
- `lambda=0.10`

Artifacts:
- `eval/phaseb_residual_objective_quick_l002_sweep/comparison/promotion_decision.json`
- `eval/phaseb_residual_objective_quick_l005_sweep/comparison/promotion_decision.json`
- `eval/phaseb_residual_objective_quick_l010_sweep/comparison/promotion_decision.json`

Headline result:
- all three settings remained `hold`
- all three settings stayed slightly worse than the champion on RMSE
- none produced evidence of a genuinely better routed specialist policy

Quick comparison:

| lambda | mean RMSE rel. improvement | mean R2 delta | cpu_act RMSE rel. improvement | cpu_act defer | cpu_act weighted adv. | cpu_act defer-weighted adv. | cpu_act usefulness gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | -0.000357 | -0.000152 | -0.000714 | 0.0058 | -0.1050 | -0.00097 | 0.2082 |
| 0.05 | -0.000337 | -0.000144 | -0.000675 | 0.0022 | -0.1152 | -0.00037 | 0.2076 |
| 0.10 | -0.000341 | -0.000145 | -0.000682 | 0.0017 | -0.1096 | -0.00024 | 0.2074 |

Read:
- lowering `lambda` preserves slightly more routing on `cpu_act`
- but held-out RMSE remains slightly negative at every setting
- the usefulness gap does shrink modestly with larger `lambda`
- the underlying weighted specialist advantage does not become positive

`california` remains dominated by the known instability path:
- challenger defer stays `0.0`
- held-out RMSE stays identical to the champion
- the gap moves from about `0.2212` to `0.2236`, but the model still falls back to anchor-only behavior

Updated interpretation:
- this auxiliary term is active and measurable
- but the current formulation does not translate into a better regression model on the quick contract
- the main effect is still routing suppression, not successful positive specialist allocation

So the strengthened conclusion is:
- the current residual-usefulness-gap objective, as written, is not the right local fix
- a smaller `lambda` only changes the degree of suppression
- the next local step should be a different objective shape, not a bigger benchmark on this exact formulation
