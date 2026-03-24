# AFC-AI to GraphDrone Research Handoff

Date: 2026-03-23

Prepared for:
- external/internal research team handoff

Prepared from branch:
- `exp/afc-cross-dataset-lma-rankloss`
- head: `6b61b04c4d65c149b45f998877c1adbc080182ca`

## Scope of this handoff

This handoff is not a generic project summary.
It is the current state of the AFC-AI to GraphDrone research line, with git-traceable branches, committed notes, durable findings, and the exact questions that remain open.

The intended use is:
1. another team checks out a specific branch/SHA below
2. they continue from a bounded hypothesis
3. they return a branch name, head SHA, note path, and artifact paths
4. their result can then be merged back into the same research memory

## Branches and roles

### Main LMA line

- branch: `exp/afc-cross-dataset-lma`
- head: `6f31fba`
- role:
  - stable cross-dataset latent-manifold-alignment line
  - contains the task bank, metadata-guided neighborhoods, live task prior, feedback loop, exact reuse, defer regularization, and threshold-sensitivity tooling

Key commits on this branch:
- `45745ca` live task-prior router prototype
- `9b3b2d1` feedback-updated task bank loop
- `413bc48` persistent task key and exact reuse blend
- `f3bf62f` task-prior-aware defer regularization
- `56706d2` threshold-sensitivity tooling
- `6f31fba` recorded threshold-sensitivity finding

Primary note:
- `docs/2026-03-23-afc-cross-dataset-lma-kickoff.md`

### Ranking-loss probe line

- branch: `exp/afc-cross-dataset-lma-rankloss`
- head: `6b61b04`
- role:
  - isolated probe for ordering-aware binary routing
  - currently a negative result, but useful because it falsifies one obvious next step

Key commits:
- `4737ef9` binary rank-loss probe
- `6b61b04` recorded rank-loss finding

Primary note:
- `docs/2026-03-23-afc-cross-dataset-lma-rankloss.md`

## What is established so far

These are the most important local conclusions from the AFC line.

### 1. Rotor / Phase B gave real mechanism signal, but not a promotable model

Earlier branch notes:
- `docs/2026-03-23-afc-phase-b-claim-first.md`
- `docs/2026-03-23-afc-phase-b-anchor-exclusion.md`
- `docs/2026-03-23-afc-phase-b-frozen-router.md`
- `docs/2026-03-23-afc-phase-b-residual-usefulness.md`

Important takeaways:
- rotor alignment is real
- regression had a real anchor-contamination asymmetry
- california had a real router instability path
- the deeper blocker was residual-usefulness / routing mismatch, not “AFC math is fake”

### 2. Cross-dataset task-conditioned LMA is viable as a framework direction

What is now implemented:
- task-conditioned prior scaffold
- persistent prototype bank
- contrastive and metadata-guided task neighborhoods
- feedback-updated bank
- exact task reuse in live GraphDrone

Key interpretation:
- classification is a much better first regime than regression
- task-conditioned or hierarchical prior is better supported than a flat universal prior
- same dataset can now reuse prior memory rather than starting from scratch

### 3. The main live binary blocker moved from prior quality to routing policy

Causal sequence already cleared:
1. task-bank identity and exact reuse are live
2. exact reuse alone did not fix the route
3. binary defer saturation was the next real blocker
4. mild task-prior-aware defer regularization was the first mechanism that changed the live route enough to pass the current binary slice guardrails

Primary artifacts:
- `eval/afc_live_task_prior_binary_deferpen_l02_v10/comparison/promotion_decision.json`
- `eval/afc_live_task_prior_binary_deferpen_l02_v11pred/threshold_analysis/threshold_summary.csv`

### 4. The next bottleneck is score ordering / threshold geometry

Threshold analysis showed:
- `diabetes` is mainly a calibration improvement
- `credit_g` improves at the default threshold, but not on best-threshold ceiling

This means:
- defer regularization helped
- but the remaining gap is no longer “no activation surface” or “no exact reuse”
- it is now the structure of the ranking / decision surface

### 5. One obvious next step is already falsified

The first simple pairwise rank-loss probe is negative:
- branch: `exp/afc-cross-dataset-lma-rankloss`
- note: `docs/2026-03-23-afc-cross-dataset-lma-rankloss.md`

Current read:
- pairwise ranking on the final blended binary score is not the right next architecture step in its current form
- it improves some calibration terms, but not the ranking ceiling we actually wanted

## Current research surface

Source of truth:
- `docs/research/current_hypotheses.md`
- `docs/research/findings.jsonl`

Most important active items:

Partially causal:
- `afc-framework-task-prior-defer-regularization`
- `afc-framework-binary-threshold-sensitivity`
- `afc-framework-live-task-exact-reuse`
- `afc-framework-feedback-updated-task-bank-loop`
- `afc-framework-metadata-guided-task-neighborhoods`

Open:
- `afc-framework-cross-dataset-token-geometry`
- `afc-framework-task-context-heldout-generalization`
- `afc-framework-task-similarity-transfer-space`
- `afc-framework-cross-dataset-hyper-lma`

Falsified:
- `afc-framework-binary-rank-loss-probe`
- `afc-b-residual-objective-gap-penalty`

## Recommended next work for the other team

The other team should not restart broad benchmarking.
They should pick one of these bounded continuation points.

### Option A: Expert-allocation ordering, not final-score ranking

Why:
- the simple rank-loss on final blended score is falsified
- the remaining bottleneck likely lives in expert allocation / attention rather than only in the final score

Good continuation branch:
- branch from `exp/afc-cross-dataset-lma` at `6f31fba`

Suggested focus:
- ordering-aware loss on specialist allocation or attention weights
- preserve the current task-prior + exact-reuse + defer-regularization path
- do not re-open the already falsified final-score pairwise formulation

### Option B: Dataset-conditioned threshold / operating-point modeling

Why:
- threshold analysis showed that `credit_g` still loses on best-threshold ceiling while improving at the default threshold

Good continuation branch:
- branch from `exp/afc-cross-dataset-lma` at `6f31fba`

Suggested focus:
- explicit operating-point modeling from task prior
- threshold-aware diagnostics and calibration curves
- possibly per-task threshold priors rather than only route priors

### Option C: Unseen-neighbor-only prior vs exact-reuse prior

Why:
- exact reuse is now live, but we still need to understand how much of the gain depends on known-task memory vs true similar-task transfer

Good continuation branch:
- branch from `exp/afc-cross-dataset-lma` at `6f31fba`

Suggested focus:
- ablate exact reuse off while preserving neighborhood prior
- compare known-task and unseen-task behavior explicitly

## Exact artifacts worth reading first

Primary branch notes:
- `docs/2026-03-23-afc-cross-dataset-lma-kickoff.md`
- `docs/2026-03-23-afc-cross-dataset-lma-rankloss.md`

Research memory:
- `docs/research/current_hypotheses.md`
- `docs/research/findings.jsonl`

Most important live artifacts:
- `eval/afc_live_task_prior_binary_feedback_exactblend_v8/comparison/promotion_decision.json`
- `eval/afc_live_task_prior_binary_deferpen_l02_v10/comparison/promotion_decision.json`
- `eval/afc_live_task_prior_binary_deferpen_l02_v11pred/threshold_analysis/threshold_summary.csv`
- `eval/afc_live_task_prior_binary_rankloss_v12rank/comparison/promotion_decision.json`
- `eval/afc_live_task_prior_binary_rankloss_v12rank/threshold_analysis/threshold_summary.csv`
- `eval/afc_live_task_prior_binary_rankloss_l003_v12rank03/comparison/promotion_decision.json`

## Handoff protocol back to this team

Please return progress in this exact format:

1. Branch name
2. Head SHA
3. One note path under `docs/`
4. Main artifact paths under `eval/`
5. One durable finding record added to `docs/research/findings.jsonl`
6. Clear status:
   - `cleared`
   - `partially_causal`
   - `open`
   - `confounded`
   - `falsified`
7. One sentence answering:
   - what hypothesis was tested
   - what changed causally
   - why the next team should or should not keep spending on it

Preferred commit pattern:
- code: `exp(afc-lma): <mechanism>`
- docs/research memory: `docs(afc-lma): record <finding>`

## Minimal checkout instructions

Stable base line:

```bash
git checkout exp/afc-cross-dataset-lma
git checkout 6f31fba
```

Negative ranking probe reference:

```bash
git checkout exp/afc-cross-dataset-lma-rankloss
git checkout 6b61b04
```

If they want a clean child branch from the stable LMA line:

```bash
git checkout exp/afc-cross-dataset-lma
git checkout -b exp/afc-cross-dataset-lma-<team-topic>
```

## Final current recommendation

Do not spend the next cycle on:
- broad benchmark expansion without mechanism focus
- the already falsified final-score pairwise rank loss

Do spend the next cycle on:
- expert-allocation ordering
- threshold / operating-point modeling
- exact-reuse vs unseen-neighbor transfer separation
