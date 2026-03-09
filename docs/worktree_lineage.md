# Worktree Lineage

This repo has two different kinds of experiment history:

1. **Tracked experiment families in the current checkout**
   - live under `experiments/`
   - should be treated as the canonical mainline tree for this checkout

2. **Branch-local research lines in sibling worktrees**
   - live under `.worktrees/` or `.claude/worktrees/`
   - may be newer, but are not automatically part of the current checkout

## Current Practical Mapping

### Current checkout
- path: `/Volumes/MacMini/Projects/Graph_Drone`
- meaning: primary working checkout
- tracked experiment families:
  - `gora_tabular`
  - `mq_gora_v4`
  - `head_routing_v5`

### Claude worktree: early v3 snapshot
- path: `.claude/worktrees/funny-davinci`
- branch: `claude/funny-davinci`
- role: early `gora_tabular` v3 jump at commit `1903f5a`

### MV-TabR-GoRA core line
- path: `.worktrees/mv-tabr-gora`
- branch: `feature/mv-tabr-gora`
- role: newer California branch-local line with `A0..A6f`

### MV-TabR-GoRA follow-ons
- `.worktrees/mv-tabr-gora-rerank`
- `.worktrees/mv-tabr-gora-a7-candidate-rerank`
- `.worktrees/mv-tabr-gora-a7a-iterative-reindex`
- role: follow-on retrieval experiments derived from `feature/mv-tabr-gora`

## How Agents Should Use This

Before inferring experiment lineage:
1. read `docs/worktree_registry.json`
2. read this file
3. distinguish:
   - current checkout facts
   - sibling worktree facts

Do not describe sibling worktree experiments as part of the current checkout unless you say so explicitly.

## Refresh Rule

After adding/removing worktrees or switching the main checkout branch, rerun:

```bash
source .venv/bin/activate
python scripts/audit_worktrees.py --write docs/worktree_registry.json
```
