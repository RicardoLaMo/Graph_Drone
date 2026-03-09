# Worktree Lineage

This repo has two different kinds of experiment history:

1. **Tracked experiment families in the current checkout**
   - live under `experiments/`
   - should be treated as the canonical mainline tree for this checkout

2. **Branch-local research lines in sibling worktrees**
   - live under `.worktrees/` or `.claude/worktrees/`
   - may be newer, but are not automatically part of the current checkout

## Current Practical Mapping

### Current checkout — agent home
- path: `/Volumes/MacMini/Projects/Graph_Drone`
- branch: `feature/gora-v5-trust-routing`
- meaning: primary working checkout, agent home directory for all agents
- tracked experiment families:
  - `gora_tabular`
  - `mq_gora_v4`
  - `head_routing_v5`

### ~~Claude worktree: early v3 snapshot~~ [ARCHIVED — do not use as home]
- path: `.claude/worktrees/funny-davinci`
- branch: `claude/funny-davinci`
- role: early `gora_tabular` v3 snapshot at commit `1903f5a`
- status: **archived reference point only** — Claude works from the main checkout

### MV-TabR-GoRA core line
- path: `.worktrees/mv-tabr-gora`
- branch: `feature/mv-tabr-gora`
- result: A6f champion at **0.4063 RMSE**
- role: California branch-local line with `A0..A6f` ablation ladder

### MV-TabR-GoRA B-series (retrieval quality: pool expansion + score biases)
- path: `.worktrees/mv-tabr-gora-rerank`
- branch: `feature/mv-tabr-gora-rerank`
- result: pool expansion K=24→96 **hurts** (+0.006 RMSE); score biases marginal
- finding: raw-space kNN can't find label-predictive far neighbors

### MV-TabR-GoRA A7a (iterative re-indexing: learned embedding-space kNN)
- path: `.worktrees/mv-tabr-gora-a7a-iterative-reindex`
- branch: `feature/mv-tabr-gora-a7a-iterative-reindex`
- result: probe ✅ (+0.011 retrieval quality), neural retrain ≈ neutral-to-negative
- Gemini-validated: weight mismatch bug fixed; post-fix evidence weaker
- finding: encoder z^v space is better for shallow retrieval, but model can't yet
  convert that into stable out-of-sample neural gains; need hybrid graph mixing

### MV-TabR-GoRA A7 candidate (earlier A7 candidate-pool attempt, superseded)
- path: `.worktrees/mv-tabr-gora-a7-candidate-rerank`
- branch: `feature/mv-tabr-gora-a7-candidate-rerank`
- role: exploratory; superseded by A7a iterative reindex

### MV-TabR-GoRA A8a (label-anchor residual prediction)
- path: `.worktrees/mv-tabr-gora-a8a-label-anchor`
- branch: `feature/mv-tabr-gora-a8a-label-anchor`
- result: A6f val=0.4510/test=0.4099 vs A8a val=0.4492/test=0.4156
- finding: val improves (−0.0018) but test worsens (+0.0057); confirms retrieval is the bottleneck

### MV-TabR-GoRA A9a (gated anchor trust)
- path: `.worktrees/mv-tabr-gora-a9a-gated-anchor`
- branch: `feature/mv-tabr-gora-a9a-gated-anchor`
- hypothesis: learned gate down-weights bad anchors; pred_v = sigmoid(gate(ctx_v)) * anchor_v + residual_v
- status: **implemented, ready to run** (smoke test ✅)

### MV-TabR-GoRA A9b (label-guided encoder pretraining)
- path: `.worktrees/mv-tabr-gora-a9b-guided-pretrain`
- branch: `feature/mv-tabr-gora-a9b-guided-pretrain`
- hypothesis: Stage 0 encoder pretraining → better kNN quality → better neural gains
- result: val=0.4882, test=0.4657 — clear regression; Stage 0 collapses direction-encoding metric structure
- finding: pretraining destroys T(z_i^v − z_j^v) space; label-predictive embeddings ≠ retrieval-useful embeddings

### MV-TabR-GoRA A10a (d_model 64→256 scale)
- path: `.worktrees/mv-tabr-gora-a10a-scale256`
- branch: `feature/mv-tabr-gora-a10a-scale256`
- hypothesis: architecture is capacity-bottlenecked; scaling d_model attacks the remaining gap directly
- status: **ready to run** (smoke test ✅, no code changes — use --d_model 256 flag)
- run: `python experiments/mv_tabr_gora/scripts/run_mv_tabr_gora.py --ablation A6f --d_model 256 --output reports/a10a_d256`

### MV-TabR-GoRA A10b (dynamic kNN — co-evolving graph)
- path: `.worktrees/mv-tabr-gora-a10b-dynamic-knn`
- branch: `feature/mv-tabr-gora-a10b-dynamic-knn`
- hypothesis: rebuild per-view kNN from current z^v embeddings every 10 epochs — graph and encoder co-evolve (TabR-like)
- status: **ready to run** (smoke test ✅ — rebuild fires correctly, ~1.4s per rebuild)
- run: `python experiments/mv_tabr_gora/scripts/run_mv_tabr_gora.py --ablation A6f A10b --rebuild_interval 10 --output reports/a10b_dynknn`

### MV-TabR-GoRA C1/E0 (consumption bottleneck diagnosis)
- path: `.worktrees/mv-tabr-gora-c1e0`
- branch: `feature/mv-tabr-gora-c1e0`
- base: `feature/mv-tabr-gora` (A6f, clean static-graph baseline)
- hypothesis: remaining gap to TabR is in consumption, not retrieval
  - C1 tests: does collapsing K=24 neighbours to one ctx_v too early hurt?
    MultiSlotAttention M=2: anchor query + learned per-slot offsets → gated sum
  - E0 tests: does single decoder head variance hurt?
    MultiHeadTaskHead M=4: 4 parallel 2-layer MLPs, averaged output
  - C1E0: both together (interaction check)
- static graph throughout — no retrieval changes
- smoke test: ✅ A6f/C1/E0/C1E0 all pass (param deltas verified)
- status: **scaffolding complete — ready for full run**
- run: `python .worktrees/mv-tabr-gora-c1e0/experiments/mv_tabr_gora/scripts/run_mv_tabr_gora.py --ablation A6f C1 E0 C1E0`

### MV-TabR-GoRA DRST (dynamic candidate-pool retriever + EdgeMLP encoding)
- path: `.worktrees/mv-tabr-gora-drst`
- branch: `feature/mv-tabr-gora-drst`
- base: `feature/mv-tabr-gora` (A6f codebase — avoids A9/A10 cruft)
- hypothesis: Q·K_ema re-ranking inside fixed pool (C=96) + EdgeMLP edge encoding
  fixes the static-direction-encoding / dynamic-graph incompatibility
- ablations: B1 (cand-pool rerank), B2 (+ EdgeMLP), B3 (+ CrossViewMixer)
- smoke test: ✅ B1 and B2 pass (KL loss visible, no shape errors)
- status: **scaffolding complete — ready for full run**
- run: `cd .worktrees/mv-tabr-gora-drst && python experiments/mv_tabr_gora/scripts/run_mv_tabr_gora.py --ablation A6f B1`

## Agent Assignments

| Agent | Home | Writes to | Role |
|---|---|---|---|
| Claude | main checkout (`feature/gora-v5-trust-routing`) | `.claude/` | Implementation |
| Gemini | reads only | `.gemini-cross-checks/` | Cross-validation |
| Codex | TBD | `.codex/` (reserved) | TBD |

For details see `docs/agents.md`.

## How Agents Should Use This

Before inferring experiment lineage:
1. read `docs/worktree_registry.json`
2. read this file
3. distinguish:
   - current checkout facts
   - sibling worktree facts (research lines)

Do not describe sibling worktree experiments as part of the current checkout unless you say so explicitly.

Do not treat `claude/funny-davinci` as the active working location — it is archived.

## Refresh Rule

After adding/removing worktrees or switching the main checkout branch, rerun:

```bash
source .venv/bin/activate
python scripts/audit_worktrees.py --write docs/worktree_registry.json
```

Then commit `docs/worktree_registry.json` with: `chore: refresh worktree registry`.
