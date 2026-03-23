# AFC Phase B Frozen-Router Regression Check

Date: 2026-03-23

Branch lineage:
- branch: `exp/afc-b-frozen-router`
- parent branch: `exp/afc-b-reg-anchor-exclusion`
- code commit for this check: `1fa3cec`

## Question

After fixing the regression anchor-exclusion asymmetry, is the remaining Phase B problem mainly caused by joint optimization of the base router and the rotor?

## Code change

Added a frozen-base rotor ablation on the learned regression router path.

What changed:
- train a plain contextual base router first
- freeze the base router parameters
- train only the rotor on top of the frozen base router
- surface per-expert attention mass and non-anchor attention entropy in benchmark diagnostics

Files:
- `src/graphdrone_fit/config.py`
- `src/graphdrone_fit/model.py`
- `scripts/run_geopoe_benchmark.py`
- `scripts/run_smart_benchmark.py`
- `src/graphdrone_fit/champion_challenger.py`
- `tests/test_champion_challenger.py`

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
conda run --no-capture-output -n h200_tabpfn python scripts/run_champion_challenger.py \
  --task regression \
  --gate mini-full \
  --champion-version v1.20.0-reg-champion \
  --challenger-version afc-b-rotor-reg-l001-frozenrouter \
  --output-dir eval/phaseb_frozen_router_l001_mini \
  --heartbeat-seconds 15
```

Tracked run:
- `output/experiments/runs/20260323_132350Z_graphdrone-afc-b-frozen-router.json`

Artifacts:
- decision: `eval/phaseb_frozen_router_l001_mini/comparison/promotion_decision.json`
- claim report: `eval/phaseb_frozen_router_l001_mini/comparison/claim_report.json`
- paired deltas: `eval/phaseb_frozen_router_l001_mini/comparison/paired_task_deltas.csv`
- champion ledger: `eval/phaseb_frozen_router_l001_mini/raw/champion/regression/run_ledger.json`
- challenger ledger: `eval/phaseb_frozen_router_l001_mini/raw/challenger/regression/run_ledger.json`

## Result against the same-run champion

Decision:
- `hold`

Summary:
- mean RMSE relative improvement: `+0.000127`
- mean R2 delta: `-0.000023`
- worst dataset RMSE relative improvement: `-0.000956`
- mean latency improvement: `-12.613873`

Claim summary:
- rotor mechanism status: `supported`
- rotor integration status: `translating`
- mean alignment cosine gain on finite tasks: `+0.015696`
- finite gain task fraction: `0.5`

## More trustworthy causal comparison: frozen challenger vs anchor-fix challenger

The branch-local champion changed materially relative to the prior corrected champion run, so the strongest causal comparison here is the frozen challenger against the earlier non-frozen challenger from the anchor-exclusion branch.

Compared runs:
- prior challenger: `eval/phaseb_reg_anchorfix_l001_mini/raw/challenger/regression/report/results_granular.csv`
- frozen challenger: `eval/phaseb_frozen_router_l001_mini/raw/challenger/regression/report/results_granular.csv`

Patch-only delta:
- mean RMSE relative improvement vs prior challenger: `-0.000225`
- mean R2 delta vs prior challenger: `-0.000094`

Per-dataset frozen-vs-prior challenger:

| dataset | rmse_rel_frozen_vs_prior | r2_delta_frozen_vs_prior | read |
|---|---:|---:|---|
| california | +0.000000 | +0.000000 | identical quality; defer increased from `0.1119` to `0.2620` |
| diamonds | +0.000000 | +0.000000 | identical quality |
| house_prices | +0.000000 | +0.000000 | identical quality |
| elevators | +0.000101 | +0.000016 | small improvement with broader non-anchor use |
| cpu_act | -0.001314 | -0.000560 | worse; attention collapsed almost entirely onto `SUB1` |
| kin8nm | -0.000139 | -0.000020 | slightly worse despite much larger non-anchor attention spread |

## Routing evidence

Useful cases:
- `elevators`: `FULL` attention fell to `0.9162`, non-anchor entropy rose to `0.6520`, and RMSE improved slightly

Failure cases:
- `cpu_act`: `FULL` attention was near zero and `SUB1` absorbed about `0.9808` of attention, while RMSE got worse
- `kin8nm`: `FULL` attention dropped from about `0.9830` to `0.5547`, but the broader specialist spread did not translate into a meaningful gain

Coverage gaps:
- `california`, `diamonds`, and `house_prices` still reported non-finite alignment gain diagnostics
- that means half the regression tasks still do not provide usable rotor-activation evidence

## Updated interpretation

The frozen-base ablation weakens the `joint_training_interference_is_the_main_blocker` hypothesis.

Why:
1. Freezing the base router did not improve the rotor challenger over the earlier corrected non-frozen challenger.
2. Some datasets changed routing behavior substantially without improving quality.
3. The most damaging case, `cpu_act`, looks like a policy-allocation failure, not a failure to preserve the pretrained base router.

Best current read:
- the rotor mechanism still appears real on the finite tasks
- the remaining blocker is more likely an objective or circuit mismatch than simple optimizer interference
- specifically, improving anchor-frame similarity does not guarantee that the router allocates mass toward specialists that are more label-useful than the anchor

## Next checks

1. Add a residual-usefulness diagnostic on the regression router fit:
   - compare specialist advantage over the anchor against the router's attention allocation
   - measure whether the rotor increases alignment without increasing weight on label-useful specialists

2. Tighten run provenance further:
   - record branch name and git status alongside git SHA in run ledgers

3. Revisit the missing-coverage tasks:
   - explain why `california`, `diamonds`, and `house_prices` still produce non-finite rotor gain diagnostics
