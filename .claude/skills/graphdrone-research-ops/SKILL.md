---
name: graphdrone-research-ops
description: "Use when running GraphDrone research as an operating system rather than a single benchmark: branch setup, benchmark contract control, claim-first evaluation, run provenance, long-form notes, and durable finding capture across `eval/`, `output/experiments/`, and `docs/research/`."
---

# GraphDrone Research Ops

Use this as the top-level GraphDrone research skill when the work spans code, experiments, diagnosis, and durable research memory.

This skill assumes the GraphDrone source of truth is the `exp-multiclass-geopoe` repo and that research should leave five artifacts:
- git/worktree snapshot
- execution lineage
- evaluation artifacts
- branch note
- durable finding update

## When To Use

Use this skill when the user asks to:
- plan or run a GraphDrone experiment
- compare champion vs challenger GraphDrone versions
- diagnose why a mechanism helped locally but not globally
- make the research cumulative rather than metric-only
- review branch hygiene, worktrees, or experiment lineage

If the main question is specifically “why did local mechanism truth fail to translate?”, immediately pair this with `$graphdrone-mechanism-diagnosis`.

If the main question is specifically “record or review what we already learned”, pair this with `$graphdrone-research-memory`.

## Operating Model

GraphDrone research now has three memory layers:
1. `output/experiments/`
   execution lineage, design files, run manifests, comparisons
2. `docs/*.md`
   branch notes, causal narratives, design reflections
3. `docs/research/`
   durable claim memory via `findings.jsonl` and `current_hypotheses.md`

Read `references/platform-map.md` when you need the concrete file map.

## Workflow

1. Freeze repo state.
- Confirm repo root, branch, head SHA, dirty files, and active worktree.
- If the repo tree matters for planning, build a tracked-file snapshot with `git-tree-manager`.

2. Frame the research change.
- State the claimed mechanism, target component, expected local signature, expected global signature, and likely failure modes.
- Decide whether this is:
  - efficiency-only
  - component truth
  - integration/circuit diagnosis
  - benchmark-contract validation
  - research-memory consolidation

3. Pick the correct evaluation surface.
- Use `scripts/run_champion_challenger.py` for GraphDrone-vs-GraphDrone comparisons.
- Keep TabPFN as context, not the promotion baseline, unless the user explicitly asks otherwise.
- For local-vs-global gaps, do not stop at the promotion decision; inspect the claim report and paired deltas.

4. Preserve execution evidence.
- Use run ledgers and heartbeat logs under `eval/.../raw/<side>/<task>/`.
- Prefer file-backed proof over one-off terminal observations.
- If the run matters for future reasoning, record it in `output/experiments/`.

5. Externalize what was learned.
- Write a branch note in `docs/YYYY-MM-DD-*.md`.
- Record a durable finding with `scripts/record_research_finding.py`.
- Regenerate `docs/research/current_hypotheses.md`.

## Default Sequence

Use this sequence unless the user asks for a narrower task:

```bash
python ${CODEX_HOME:-$HOME/.codex}/skills/git-codebase-ops/scripts/repo_snapshot.py \
  --output output/repo/git_snapshot.json

python ${CODEX_HOME:-$HOME/.codex}/skills/git-tree-manager/scripts/build_git_tree_snapshot.py \
  --output-dir output/repo_tree
```

Then use the repo-native GraphDrone surfaces:

```bash
python scripts/run_champion_challenger.py --task regression --gate quick

python scripts/record_research_finding.py rebuild
```

For tracked runs, pair this with `experiment-design-tracker`:

```bash
python ${CODEX_HOME:-$HOME/.codex}/skills/experiment-design-tracker/scripts/track_experiment.py \
  record-run \
  --profile graphdrone \
  --seed 42 \
  --command "python scripts/run_champion_challenger.py ..."
```

## Guardrails

- Do not treat `promotion_decision.json` as the whole answer when a mechanism question remains open.
- Do not trust sprint-like canaries for merge decisions without checking the full intended contract.
- Do not overwrite or clean `output/` or `.claude-analysis/` unless the user explicitly asks.
- Do not conflate execution lineage with durable research memory; both are required.

## Resources

- `references/platform-map.md`: where GraphDrone keeps code, benchmark outputs, ledgers, notes, and claim memory
- `references/flow.md`: the recommended GraphDrone research loop from branch creation to durable finding
