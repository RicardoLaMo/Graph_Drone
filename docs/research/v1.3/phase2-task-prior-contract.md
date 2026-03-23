# V1.3 Phase 2 — Task-Conditioned Prior Contract

Date: 2026-03-23
Branch: exp/v1.3/phase2-task-prior
Depends on: exp/v1.3/phase1-defer-reg (promoted)

## Hypothesis

Porting the cross-dataset task-conditioned prior framework from the AFC line
into main binary classification router training will enable cross-dataset
knowledge transfer. The TaskConditionedRouter wrapper shifts the anchor token
by a learned projection of the task prior vector, conditioning the routing
policy on neighborhood similarity to previously seen datasets.

With exact reuse blend enabled, the router should benefit from known-dataset
memory (e.g., credit_g seen before) and use a similarity-weighted prior for
unseen or structurally similar datasets.

## Source

- AFC established mechanism: afc-framework-task-conditioned-lma-prior (partially_causal)
- AFC promoted base: defer regularization (phase 1, this branch's dependency)
- AFC bank infrastructure: exp/afc-cross-dataset-lma at 6f31fba
- AFC bank artifacts: eval/afc_task_prototype_bank_*/

## Prerequisite: Bank Training

Before benchmarking, a prototype bank must be trained on the 9 classification
datasets using router tokens from the smart benchmark runs.

Training reference:
- Script: scripts/prototype_task_conditioned_lma.py (from AFC worktree)
- Output: eval/task_prior_banks/binary_v1/transformer_prototype_bank.json
- Output: eval/task_prior_banks/binary_v1/transformer_encoder_state.pt

## Config

- Preset: v1_3_phase2
- `task_prior_bank_dir` = eval/task_prior_banks/binary_v1/ (env: GRAPHDRONE_TASK_PRIOR_BANK_DIR)
- `task_prior_strength` = 0.5 (env: GRAPHDRONE_TASK_PRIOR_STRENGTH)
- `task_prior_exact_reuse_blend` = 0.5 (env: GRAPHDRONE_TASK_PRIOR_EXACT_REUSE_BLEND)
- `task_prior_encoder_kind` = transformer (env: GRAPHDRONE_TASK_PRIOR_ENCODER_KIND)
- `defer_penalty_lambda` = 0.5 (inherited from phase 1)
- Version string: 2026.03.23-clf-v1.3-phase2

## Benchmark Contract

- Runner: scripts/run_smart_benchmark.py
- Champion version: 2026.03.23-clf-v1.3-phase1 (phase 1 promoted result)
- Challenger version: 2026.03.23-clf-v1.3-phase2
- Datasets: 9 classification datasets × 3 folds (54 tasks)
- Cache policy: bump GRAPHDRONE_VERSION_OVERRIDE before each run
- Env: GRAPHDRONE_PRESET=v1_3_phase2, GRAPHDRONE_TASK_PRIOR_BANK_DIR=<path>

## Gate Criteria

MUST (all required for promotion):
- Overall clf ELO >= Phase 1 ELO (no regression)
- No single-dataset F1 regression > 0.005

SHOULD (not hard gates):
- credit_g gap fully closed (F1 >= TabPFN baseline)
- task_prior_exact_reuse_available=True for known datasets (credit_g, diabetes)
- task_prior_top_neighbor_prob > 0.5 (bank is decisive)

Evidence grade: evidence-grade (full benchmark, controlled comparison)

## Failure Protocol

If overall ELO regresses vs Phase 1:
1. Set task_prior_strength=0.0 to isolate TaskConditionedRouter overhead
2. If strength=0 matches Phase 1: bank quality issue → retrain bank
3. If strength=0 also regresses: implementation bug → bisect

If credit_g does not close:
- Record as partially_causal in findings.jsonl
- Proceed to Phase 3 (threshold modeling) as the next mechanism

## Files Changed

- src/graphdrone_fit/task_conditioned_prior.py: new (ported from AFC worktree)
- src/graphdrone_fit/set_router.py: +TaskConditionedRouter class
- src/graphdrone_fit/config.py: +task_prior_bank_dir, encoder_kind, strength, dataset_key, exact_reuse_blend
- src/graphdrone_fit/model.py: +_task_prior_confidence_scale, +_maybe_attach_task_prior_router; call after router build
- src/graphdrone_fit/presets.py: +v1_3_phase2 preset with env-var driven config
- tests/test_task_conditioned_prior.py: 11 unit tests (all pass)
