# AFC Phase B Regression Anchor-Exclusion Check

Date: 2026-03-23

Branch lineage:
- branch: `exp/afc-b-reg-anchor-exclusion`
- parent branch: `exp/afc-b-cayley-rotor`
- code commit for this check: `a69c77f`

## Question

Was the Phase B regression underperformance partly caused by the defer integrator mixing anchor mass into the deferred specialist blend?

## Code change

Patched the regression defer path to exclude anchor mass from the deferred specialist blend, matching the classification GeoPOE path.

Files:
- `src/graphdrone_fit/defer_integrator.py`
- `src/graphdrone_fit/model.py`
- `tests/test_defer_integrator.py`

Supporting diagnostic surfacing:
- `src/graphdrone_fit/champion_challenger.py`
- `scripts/run_geopoe_benchmark.py`
- `scripts/run_smart_benchmark.py`

New diagnostics:
- `mean_specialist_mass`
- `mean_anchor_attention_weight`

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
conda run --no-capture-output -n h200_tabpfn python scripts/run_champion_challenger.py \
  --task regression \
  --gate mini-full \
  --champion-version v1.20.0-reg-champion \
  --challenger-version afc-b-rotor-reg-l001-anchorfix \
  --output-dir eval/phaseb_reg_anchorfix_l001_mini \
  --heartbeat-seconds 15
```

Artifacts:
- decision: `eval/phaseb_reg_anchorfix_l001_mini/comparison/promotion_decision.json`
- claim report: `eval/phaseb_reg_anchorfix_l001_mini/comparison/claim_report.json`
- paired deltas: `eval/phaseb_reg_anchorfix_l001_mini/comparison/paired_task_deltas.csv`

## Result against corrected champion

Decision:
- `hold`

Summary:
- mean RMSE relative improvement: `+0.000032`
- mean R2 delta: `+0.000059`
- worst dataset RMSE relative improvement: `-0.001024`
- mean latency improvement: `-2.255128`

Interpretation:
- With the corrected regression circuit, the rotor challenger is no longer clearly negative.
- But it still does not clear the promotion bar. Quality is effectively flat while latency is substantially worse.

## Patch effect on the rotor challenger itself

Compared to the earlier Phase B challenger on the parent branch:
- prior challenger artifact: `eval/phaseb_claim_reg_l001_mini/raw/challenger/regression/report/results_granular.csv`
- patched challenger artifact: `eval/phaseb_reg_anchorfix_l001_mini/raw/challenger/regression/report/results_granular.csv`

Patch-only delta:
- mean RMSE relative improvement vs old challenger: `+0.005111`
- mean R2 delta vs old challenger: `+0.002523`

Per-dataset patch effect:

| dataset | rmse_rel_patch_improvement | r2_patch_delta |
|---|---:|---:|
| california | +0.026630 | +0.014221 |
| diamonds | +0.000000 | +0.000000 |
| house_prices | +0.000000 | +0.000000 |
| elevators | +0.002926 | +0.000460 |
| cpu_act | +0.001050 | +0.000447 |
| kin8nm | +0.000061 | +0.000009 |

This means the anchor-exclusion fix materially helped the rotor branch.

## Patch effect on the champion path

Compared to the earlier champion run on the parent branch:
- mean RMSE relative improvement vs old champion: `+0.000793`
- mean R2 delta vs old champion: `+0.000110`

So the patch helped both paths, but it helped the rotor challenger much more than the champion path.

## Updated interpretation

The regression anchor-in-specialist asymmetry was a real issue and was at least partly causal.

Why that conclusion is justified:
1. The code-path asymmetry was real.
2. Fixing it improved the rotor challenger much more than it improved the champion path.
3. The prior strong negative rotor result was therefore confounded by circuit design.

But the fix did **not** make Phase B good enough:
- the rotor challenger is now roughly quality-neutral against the corrected champion
- latency remains much worse
- some datasets still report non-finite or missing alignment diagnostics, so the rotor evidence is still incomplete

## Best current read

The dominant failure mode is now:
- `mechanism_supported_but_misintegrated`, with the first integration bug partially fixed

The next likely blocker is no longer the anchor-in-specialist blend itself. It is either:
- router/rotor joint-training interference, or
- a mismatch between cosine-to-anchor alignment and label-residual usefulness

## Next checks

1. Add policy-coupling diagnostics on the learned router path:
   - anchor attention weight
   - non-anchor attention entropy
   - per-expert mass shifts

2. Test a frozen-router variant:
   - keep champion router weights fixed
   - train only the rotor
   - evaluate whether joint training is the remaining source of degradation

3. Replace or augment cosine alignment with a residual-usefulness diagnostic:
   - compare `(specialist_pred - anchor_pred)` against `(y - anchor_pred)`
   - this is a better fit for a residual-specialist architecture than anchor similarity alone
