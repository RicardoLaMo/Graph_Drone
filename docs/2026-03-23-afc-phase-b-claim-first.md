# AFC Phase B Claim-First Review

Date: 2026-03-23

Branch lineage:
- branch: `exp/afc-b-cayley-rotor`
- code commit used for final interpretation: `284ebca991a00341e32e371cf0b8b0d880004e93`
- earlier evaluation harness commit on same branch: `d9e066a73809b84a112c2eab055476d197513dbf`
- follow-up research hygiene fix: explicit router seed was added after this review so future challenger runs are less confounded by router init noise

Experiment tracker records:
- design: `output/experiments/designs/20260323_011330Z_afc-phase-b-claim-first-mini-full.md`
- regression run: `output/experiments/runs/20260323_012037Z_graphdrone-afc-phase-b.json`
- binary run: `output/experiments/runs/20260323_012353Z_graphdrone-afc-phase-b.json`

## Question

If Cayley rotor alignment increases anchor-specialist geometric agreement, why do RMSE or F1 still fail to improve?

## Benchmark contracts

Regression:
- runner: `scripts/run_champion_challenger.py`
- gate: mini-full, fold 0
- champion: `v1_20_champion`
- challenger: `afc_candidate`
- env: `GRAPHDRONE_ROUTER_KIND=contextual_transformer_rotor`
- env: `GRAPHDRONE_ALIGNMENT_LAMBDA=0.01`
- env: `GRAPHDRONE_ENABLE_LEGITIMACY_GATE=0`
- artifacts: `eval/phaseb_claim_reg_l001_mini/`

Binary classification:
- runner: `scripts/run_champion_challenger.py`
- datasets: `diabetes`, `credit_g`
- fold: `0`
- champion: `v1_20_champion`
- challenger: `afc_candidate`
- env: `GRAPHDRONE_ROUTER_KIND=noise_gate_router_rotor`
- env: `GRAPHDRONE_ALIGNMENT_LAMBDA=0.01`
- env: `GRAPHDRONE_ENABLE_LEGITIMACY_GATE=0`
- artifacts: `eval/phaseb_claim_bin_l001_fold0/`

## Headline findings

Regression:
- bottom line is negative: mean RMSE relative improvement `-0.004407`, mean R2 delta `-0.002353`, mean latency improvement `-0.158253`
- component claim is true but incomplete: rotor mean alignment cosine gain `+0.013221`
- interpretation: alignment improved on finite rows, but the win did not carry to prediction error

Binary:
- bottom line is mixed: mean F1 delta `-0.001826`, mean log-loss relative improvement `+0.006898`, mean latency improvement `-0.085595`
- component claim is true: rotor mean alignment cosine gain `+0.045846`
- interpretation: rotor improved probability quality more than thresholded classification quality

## Mechanism observations

Regression finite-claim rows:

| dataset | rmse_rel_improvement | alignment_cosine_gain | defer_delta |
|---|---:|---:|---:|
| california | -0.027488 | 0.005508 | +0.048979 |
| elevators | +0.001122 | 0.015317 | -0.382197 |
| cpu_act | -0.000001 | 0.016404 | +0.000480 |
| kin8nm | -0.000078 | 0.015654 | +0.012613 |

Binary:

| dataset | f1_delta | log_loss_rel_improvement | alignment_cosine_gain | defer_delta |
|---|---:|---:|---:|---:|
| diabetes | -0.003651 | +0.013797 | 0.031600 | +0.003746 |
| credit_g | +0.000000 | -0.000000 | 0.060091 | +0.000076 |

## Interpretation

This looks less like "the rotor math is wrong" and more like "the current objective and routing architecture do not know how to cash out alignment gains."

1. Alignment-to-anchor is not the same objective as predictive usefulness.
   The current auxiliary loss rewards making specialist tokens more anchor-like. That can reduce view mismatch, but it can also suppress precisely the specialist differences that make routing useful.

2. The router and rotor are trained jointly, so defer behavior can change even when the alignment gain is small.
   On `california`, alignment gain is only `0.0055`, yet defer rises materially and RMSE gets worse. This suggests the co-trained router reacts to the altered representation in a way that increases specialist reliance without enough geometric benefit.

3. Positive log-loss with flat or worse F1 is not contradictory.
   On `diabetes`, log-loss improves while F1 slips. That means the challenger likely produces slightly better probabilities or calibration, but not a better default decision boundary.

4. The current claim metric is necessary but not sufficient.
   `alignment_cosine_gain > 0` confirms the rotor moved the token geometry. It does not confirm that the moved axes are the label-relevant axes, nor that the defer policy used that movement correctly.

5. Some regression datasets still produce non-finite rotor diagnostics.
   `diamonds` and `house_prices` report undefined rotor claim diagnostics while leaving metrics unchanged. This should be treated as instability or non-activation, not as evidence that alignment helps.

## Better research assessment flow

Use a 4-layer assessment instead of a single pass/fail benchmark:

1. Component truth
   Ask: did the proposed mechanism move in the intended direction?
   For Phase B this is `alignment_cosine_gain`, `alignment_aux_loss`, and finite-claim coverage.

2. Policy coupling
   Ask: did routing behavior change in the expected way?
   For Phase B this is `defer_delta`, specialist weight entropy, anchor weight, and per-expert mass shifts.
   If alignment changes but routing does not, the representation win is disconnected from decision-making.
   If routing changes strongly when alignment barely moves, the joint objective is unstable.

3. Outcome translation
   Ask: did prediction quality improve under the metric family that matches the claim?
   Regression: RMSE, MAE, R2.
   Classification: separate ranking/calibration from thresholded decisions.
   Keep F1, but pair it with log-loss, PR-AUC, and a threshold sweep.

4. Reproducibility control
   Ask: could the observed difference be explained by random router initialization rather than the intervention?
   Keep router seed fixed, log it in run provenance, and avoid interpreting a one-off challenger delta as a mechanism result.

## Recommended next questions

1. Why does defer move more than alignment on `california`?
   This is the clearest regression failure.

2. Should rotor regularize only the specialist-attention path, not the shared router/defer parameters?
   Right now the auxiliary loss can steer the whole learned router.

3. Should the alignment target be residual-aware rather than anchor-similarity-aware?
   If specialists exist to explain anchor failures, forcing them toward the anchor frame may erase the very residual information they need to contribute.

4. Should binary evaluation headline log-loss and PR-AUC before F1?
   F1 at a fixed threshold can hide gains in probability quality.

## Current decision

Do not promote Phase B as-is.

But the reason is now clearer:
- the rotor mechanism itself is not falsified
- the current GraphDrone training/routing design does not reliably translate that mechanism into better end-task performance

## External critique

Outside-in packet:
- packet dir: `.claude-analysis/20260323T030438Z/`
- task: `.claude-analysis/20260323T030438Z/task.md`
- evidence: `.claude-analysis/20260323T030438Z/evidence.md`
- external output: `.claude-analysis/20260323T030438Z/analysis.claude.txt`

Strongest external conclusion:
- the broad claim "translation failure" is directionally right, but too vague
- the regression path has a more specific architectural asymmetry: `defer_integrator.py` mixes the anchor inside the so-called specialist blend, while binary `geo_ensemble.py` explicitly excludes anchor mass from the specialist distribution
- this makes the regression defer scalar an interpolation between anchor-only and an all-expert mixture that still contains the anchor, rather than a clean anchor-vs-specialist switch

Why that matters:
- Phase B rotor rotates specialists toward the anchor frame
- if the regression specialist blend already contains anchor mass, then making specialists more anchor-like can collapse useful diversity without giving the defer circuit a clean way to exploit specialist residual signal
- that is a circuit-design issue, not simply "rotor math failed"

The external critique also sharpened two secondary points:
- the current rotor loss is anchor-similarity maximizing, not residual-usefulness maximizing
- binary evaluation is probably seeing calibration gains more than threshold gains, so `log_loss` is a better near-term read than fixed-threshold `F1`

## Revised best interpretation

The current best interpretation is:
- `implementation_asymmetry` on regression is the leading hypothesis
- `mechanism_supported_but_misintegrated` remains true at the broader level
- `metric_mismatch` is likely part of the binary story, but not the primary regression failure

This is a better causal ordering than the earlier note:
1. Rotor alignment gain is real on activated rows.
2. The regression circuit does not expose a clean specialist-only path because anchor mass remains inside the deferred mixture.
3. The rotor loss also pushes toward anchor similarity rather than label-residual usefulness, which can further blunt specialist value.

## Ranked next checks

1. Patch regression defer integration to exclude anchor mass from the specialist mixture, mirroring `learned_geo_poe_blend_torch()`, then rerun the smallest regression challenger contract.
   This is the highest-signal falsification test for the asymmetry hypothesis.

2. Add policy-coupling diagnostics before further hyperparameter tuning.
   Most useful additions:
   - mean anchor attention weight
   - non-anchor attention entropy
   - per-expert mass shifts
   Without these, `defer_delta` alone is too blunt.

3. Check whether rotor improves prediction-space residual usefulness rather than only token-space cosine.
   A better specialist diagnostic is whether `(specialist_pred - anchor_pred)` aligns better with `(y - anchor_pred)` after rotation.

4. Investigate why `diamonds` and `house_prices` produce non-finite rotor diagnostics.
   Those rows currently weaken any aggregate regression interpretation.

5. Treat binary `log_loss` as the primary Phase B read until a threshold sweep or PR-AUC check is added.

## Workflow note

To make this repeatable beyond this one branch, a local Codex skill was created:
- skill: `raphael-research-loop`
- path: `/home/wliu23/.codex/skills/raphael-research-loop/`

The loop is:
- inside-out: component truth -> circuit coupling -> outcome translation -> reproducibility
- outside-in: external critique on the exact branch/artifact packet
- persistence: repo note plus packet artifacts, rather than chat-only reasoning
