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
