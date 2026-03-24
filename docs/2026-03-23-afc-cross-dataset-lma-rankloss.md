# AFC Cross-Dataset LMA Rank-Loss Probe

Date: 2026-03-23

Branch:
- `exp/afc-cross-dataset-lma-rankloss`

Base:
- `exp/afc-cross-dataset-lma` at `6f31fba`

## Why this branch exists

The prior branch established:
- task-bank identity and exact reuse are live
- binary defer saturation was a real blocker
- mild task-prior-aware defer regularization produced the first promoted binary slice on `credit_g + diabetes`

But the threshold analysis also clarified the remaining gap:
- `diabetes` improved mainly through calibration
- `credit_g` improved at the default operating point but still trailed the champion on best-threshold F1

That moved the next question from defer magnitude to score ordering:

> Can a binary pairwise ranking auxiliary term improve the ranking surface, rather than only calibration and default-threshold behavior?

## Mechanism added

The binary learned-router objective was extended with an optional task-prior-aware pairwise ranking term:
- active only for binary classification
- weighted by task-prior confidence, like the defer penalty
- computed from the blended positive-class score difference

New controls:
- `GRAPHDRONE_TASK_PRIOR_RANK_LOSS_LAMBDA`
- `GRAPHDRONE_TASK_PRIOR_RANK_MARGIN`

The formulation is a simple logistic pairwise loss over validation positives vs negatives:
- encourage positive rows to score above negative rows
- do not change the static multiclass route

## First probe: lambda = 0.10

Run:

```bash
GRAPHDRONE_SAVE_CLASSIFICATION_PREDICTIONS=1 \
GRAPHDRONE_TASK_PRIOR_BANK_DIR=eval/afc_task_prototype_bank_cls_feedback_v1 \
GRAPHDRONE_TASK_PRIOR_ENCODER_KIND=transformer \
GRAPHDRONE_TASK_PRIOR_STRENGTH=0.5 \
GRAPHDRONE_TASK_PRIOR_EXACT_REUSE_BLEND=0.75 \
GRAPHDRONE_TASK_PRIOR_DEFER_PENALTY_LAMBDA=0.2 \
GRAPHDRONE_TASK_PRIOR_DEFER_TARGET=0.85 \
GRAPHDRONE_TASK_PRIOR_RANK_LOSS_LAMBDA=0.1 \
GRAPHDRONE_TASK_PRIOR_RANK_MARGIN=0.0 \
PYTHONPATH=src python scripts/run_champion_challenger.py \
  --task classification \
  --datasets credit_g diabetes \
  --folds 0 \
  --max-samples 384 \
  --champion-preset v1_20_champion \
  --challenger-preset afc_candidate \
  --champion-version champion-bin-no-prior-v12rank \
  --challenger-version challenger-bin-task-prior-rankloss-v12rank \
  --output-dir eval/afc_live_task_prior_binary_rankloss_v12rank
```

Artifacts:
- `eval/afc_live_task_prior_binary_rankloss_v12rank/comparison/promotion_decision.json`
- `eval/afc_live_task_prior_binary_rankloss_v12rank/raw/challenger/classification/report/results_granular.csv`
- `eval/afc_live_task_prior_binary_rankloss_v12rank/threshold_analysis/threshold_summary.csv`

Result:
- decision: `hold`
- `mean_f1_delta = +0.00130`
- `mean_log_loss_delta = -0.00359`
- `credit_g`
  - `f1_macro = 0.5661`
  - `log_loss = 0.5621`
  - `mean_defer_prob = 0.9155`
- `diabetes`
  - `f1_macro = 0.7456`
  - `log_loss = 0.4795`
  - `mean_defer_prob = 0.9561`

Threshold read:
- `credit_g`
  - default F1 improved slightly over champion: `+0.00259`
  - best-threshold F1 still trailed champion: `-0.03918`
  - best challenger threshold stayed at `0.55` vs champion `0.80`
- `diabetes`
  - F1 unchanged across thresholds
  - calibration still improved

Interpretation:
- this formulation improved probability quality more than it improved ranking
- it did not recover the missing best-threshold ceiling on `credit_g`

## Second probe: lambda = 0.03

Run:

```bash
GRAPHDRONE_SAVE_CLASSIFICATION_PREDICTIONS=1 \
GRAPHDRONE_TASK_PRIOR_BANK_DIR=eval/afc_task_prototype_bank_cls_feedback_v1 \
GRAPHDRONE_TASK_PRIOR_ENCODER_KIND=transformer \
GRAPHDRONE_TASK_PRIOR_STRENGTH=0.5 \
GRAPHDRONE_TASK_PRIOR_EXACT_REUSE_BLEND=0.75 \
GRAPHDRONE_TASK_PRIOR_DEFER_PENALTY_LAMBDA=0.2 \
GRAPHDRONE_TASK_PRIOR_DEFER_TARGET=0.85 \
GRAPHDRONE_TASK_PRIOR_RANK_LOSS_LAMBDA=0.03 \
GRAPHDRONE_TASK_PRIOR_RANK_MARGIN=0.0 \
PYTHONPATH=src python scripts/run_champion_challenger.py \
  --task classification \
  --datasets credit_g diabetes \
  --folds 0 \
  --max-samples 384 \
  --champion-preset v1_20_champion \
  --challenger-preset afc_candidate \
  --champion-version champion-bin-no-prior-v12rank03 \
  --challenger-version challenger-bin-task-prior-rankloss-l003-v12rank03 \
  --output-dir eval/afc_live_task_prior_binary_rankloss_l003_v12rank03
```

Artifacts:
- `eval/afc_live_task_prior_binary_rankloss_l003_v12rank03/comparison/promotion_decision.json`
- `eval/afc_live_task_prior_binary_rankloss_l003_v12rank03/raw/challenger/classification/report/results_granular.csv`

Result:
- decision: `hold`
- `worst_dataset_f1_guardrail=FAIL (-0.027956)`
- `credit_g`
  - `f1_macro = 0.5355`
  - `log_loss = 0.5602`
  - `mean_defer_prob = 0.9201`
- `diabetes`
  - `f1_macro = 0.7456`
  - `log_loss = 0.4784`
  - `mean_defer_prob = 0.9497`

Interpretation:
- lowering the rank-loss weight did not rescue the formulation
- `credit_g` became materially worse on F1 while still only modestly improving log-loss
- that makes this first pairwise-rank auxiliary look unstable rather than simply over-tuned

## Current conclusion

This first binary pairwise ranking formulation is **not** the right next architecture step in its current shape.

What it did show:
- the branch is now deep enough that we can distinguish:
  - calibration improvements
  - default-threshold improvements
  - best-threshold / ranking-surface improvements

What it failed to do:
- it did not improve the `credit_g` best-threshold ceiling
- it did not produce a better overall binary challenger than defer regularization alone
- the lower-weight variant regressed F1 badly enough to fail the guardrail

So the current read is:
- simple pairwise ranking on the blended validation score is too weak or too miscoupled
- the next ordering-aware step should probably target:
  - expert allocation / attention ordering directly
  - or dataset-conditioned threshold / operating-point modeling
  - rather than only adding a pairwise loss on the final blended score
