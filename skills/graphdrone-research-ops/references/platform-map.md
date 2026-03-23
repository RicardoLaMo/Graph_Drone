# GraphDrone Research Platform Map

Use this file when you need the repo-native locations, not generic placeholders.

## Core Repos And Identity

- Main GraphDrone experiment repo: `exp-multiclass-geopoe/`
- Environment facts: `AGENTS.md`, `CLAUDE.md`
- Worktree lineage: `docs/worktree_lineage.md`, `docs/worktree_registry.json`
- Branching guidance: `docs/git_doe_strategy.md`

## Artifact Layers

### 1. Execution lineage

- `output/experiments/designs/`
- `output/experiments/runs/`
- `output/experiments/comparisons/`
- append-only index: `output/experiments/index.jsonl`

### 2. Benchmark and evaluation artifacts

- champion/challenger output roots: `eval/<run_name>/`
- comparison files:
  - `comparison/promotion_decision.json`
  - `comparison/promotion_report.md`
  - `comparison/claim_report.json`
  - `comparison/claim_report.md`
  - `comparison/paired_task_deltas.csv`
- provenance files:
  - `raw/<side>/<task>/run_ledger.json`
  - `raw/<side>/<task>/run_events.jsonl`

### 3. Long-form notes

- branch or experiment notes: `docs/YYYY-MM-DD-*.md`
- strategy and lineage docs: `docs/*.md`, `docs/plans/*.md`

### 4. Durable research memory

- `docs/research/findings.jsonl`
- `docs/research/current_hypotheses.md`
- `docs/research/README.md`

## High-Signal Scripts

- `scripts/run_champion_challenger.py`
- `scripts/run_geopoe_benchmark.py`
- `scripts/run_smart_benchmark.py`
- `scripts/record_research_finding.py`

## High-Signal Diagnostics

Look for these fields before inventing new instrumentation:
- `alignment_cosine_gain`
- `mean_specialist_mass`
- `mean_anchor_attention_weight`
- `effective_defer_rate`
- `router_nonfinite_fallback`
- `validation_best_specialist_advantage_score`
- `validation_weighted_specialist_advantage_score`
- `validation_defer_weighted_specialist_advantage_score`
- `validation_top_specialist_advantage_score`

## Recommended Generic Skill Pairings

- `git-codebase-ops`: repo health and commit readiness
- `git-tree-manager`: tracked tree map before refactors or skill design
- `experiment-design-tracker`: design/run manifests
- `raphael-research-loop`: inside-out plus outside-in diagnosis
- `benchmark-metric-interpretation`: metric-family interpretation
- `h200-gpu-ops`: GPU runtime and execution checks
- `claude-cli-analyzer` or `gemini-validator-reconciler`: independent critique
