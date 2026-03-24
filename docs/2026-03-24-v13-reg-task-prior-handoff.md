# v1.3 Regression Task-Prior Handoff

Branch:
- `exp/v13-reg-afc-revisit`

Current branch head:
- `de1037e` before the residual-row-opportunity code now in working tree

This handoff packages the regression task-prior / local-global LMA line for the next team.
It is written for GitHub work, not only local continuation.

## What is established

1. Hard-regime stability is cleared.
- Claim: `v13-reg-hard-regime-router-stability`
- Evidence:
  - `eval/v13_reg_task_prior_hardregimes_quick_v2/comparison/promotion_decision.json`
- Meaning:
  - `california`, `diamonds`, and `house_prices` can stay `clean_routed`
  - regression router nonfinite fallback is no longer the dominant blocker on this slice

2. Task-prior architecture matters.
- Additive anchor-shift is inferior.
- `routing_bias` is the better regression task-prior architecture.
- Evidence:
  - `eval/v13_reg_task_prior_hardregimes_routingbias_v1/comparison/promotion_decision.json`

3. Local-global shaping matters when it is expert-specific.
- Best current local/global result:
  - `eval/v13_reg_task_prior_hardregimes_routingbias_expertlocal_v1/comparison/promotion_decision.json`
- Summary:
  - mean RMSE relative improvement: about `+0.000200`
  - still `hold`, but best among the regression task-prior variants tested on the stabilized hard-regime slice

## What is falsified

1. Stronger additive coupling is not the fix.
- Evidence:
  - `eval/v13_reg_task_prior_hardregimes_coupling_strength1_v1/comparison/promotion_decision.json`
  - `eval/v13_reg_task_prior_hardregimes_coupling_exact1_v1/comparison/promotion_decision.json`

2. Static dataset-level expert opportunity is not the right teacher.
- Evidence:
  - `eval/v13_reg_task_prior_hardregimes_routingbias_expertlocalopp_v1/comparison/promotion_decision.json`
- Claim:
  - `v13-reg-task-prior-expert-opportunity-gate`

3. Raw row-level disagreement is not the right teacher.
- Evidence:
  - `eval/v13_reg_task_prior_hardregimes_routingbias_expertrowopp_v1/comparison/promotion_decision.json`
- Claim:
  - `v13-reg-task-prior-row-opportunity-gate`

4. Residual disagreement modulation is also not the right teacher family.
- Evidence:
  - `eval/v13_reg_task_prior_hardregimes_routingbias_expertrowresid_v1/comparison/promotion_decision.json`
- Meaning:
  - the next team should stop reparameterizing disagreement-derived opportunity and move to a different teacher family

## Best Known Baseline For Next Work

Recommended base for the next branch:
- commit: `1a70b29`
- branch context: `exp/v13-reg-afc-revisit`
- rationale:
  - this is the best current regression task-prior / local-global architecture result
  - it includes:
    - stabilized hard-regime routing
    - `routing_bias`
    - row-level local gate
    - expert-local gate
  - it does **not** include the later falsified opportunity-teacher experiments

If the team prefers the latest full history with all falsified probes recorded, start from:
- current working line on `exp/v13-reg-afc-revisit`

## Recommended Next Branches

1. `exp/v13-reg-task-prior-teacher-family`
- start from `1a70b29`
- mission:
  - test a new teacher family that is not derived from disagreement magnitude
  - examples:
    - non-anchor positive-disagreement teacher with explicit thresholding from validation winners
    - teacher based on specialist rank consistency across splits
    - teacher based on expert-family or regime-conditioned priors rather than row disagreement

2. `exp/v13-reg-task-prior-wider-slice`
- start from `1a70b29`
- mission:
  - test whether the expert-local local/global gain survives a broader clean-routed regression slice beyond the hard-regime trio

3. `exp/v13-reg-task-prior-github-port`
- start from latest recorded branch tip
- mission:
  - port the cleaned regression task-prior stack plus research-memory docs to the shared GitHub repo
  - preserve artifact and claim traceability

## Required GitHub Operation Rules

1. Preserve traceability.
- Report:
  - branch name
  - head SHA
  - base SHA
  - note path
  - artifact paths

2. Do not squash away causal checkpoints unless the causal content is preserved elsewhere.

3. Keep the durable research memory format.
- update:
  - `docs/research/findings.jsonl`
  - `docs/research/current_hypotheses.md`

4. Keep mechanism-first evidence.
- every new branch should report:
  - route-state summary
  - fallback summary
  - paired task deltas
  - the specific task-prior diagnostics relevant to the mechanism

## Exact Local Commands Used On This Line

Focused tests:
```bash
cd /home/wliu23/projects/GraphDrone2/.worktrees/v13-reg-afc-revisit
PYTHONPATH=src pytest -q tests/test_model_diagnostics.py tests/test_model_task_prior_config.py
```

Hard-regime champion/challenger contract:
```bash
cd /home/wliu23/projects/GraphDrone2/.worktrees/v13-reg-afc-revisit
env \
  GRAPHDRONE_TASK_PRIOR_BANK_DIR=eval/v13_reg_task_prior_bank_v2 \
  GRAPHDRONE_TASK_PRIOR_MODE=routing_bias \
  GRAPHDRONE_TASK_PRIOR_STRENGTH=0.5 \
  GRAPHDRONE_TASK_PRIOR_LOCAL_GATE_ALPHA=2.0 \
  GRAPHDRONE_TASK_PRIOR_EXPERT_LOCAL_GATE_ALPHA=2.0 \
  GRAPHDRONE_TASK_PRIOR_EXACT_REUSE_BLEND=0.6 \
  python scripts/run_champion_challenger.py \
    --task regression \
    --datasets california diamonds house_prices \
    --folds 0 1 2 \
    --champion-preset v1_20_champion \
    --challenger-preset afc_candidate \
    --champion-version v13-reg-taskprior-routingbias-champion-v1 \
    --challenger-version <variant-version> \
    --output-dir <variant-output-dir>
```

## Return Format Expected From The Next Team

Reply with:
- branch:
- head SHA:
- base SHA:
- note:
- key artifacts:
- claim status:
- one-sentence causal conclusion:
- next recommended branch:

## Bottom Line

The success to carry forward is not a benchmark win yet. It is the research narrowing:
- regression task priors are live
- hard-regime routing can be stabilized
- `routing_bias + expert-local gating` is the best current local/global architecture
- disagreement-derived opportunity teachers should be deprioritized

That is the correct starting point for the next GitHub team.
