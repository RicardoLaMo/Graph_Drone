# Claude Skill Bridge

These skill packs are generated from the repo `skills/` directory.
Treat `skills/` as the source of truth and `.claude/skills/` as the Claude-facing projection.

## How To Use

1. Read the relevant repo context in `CLAUDE.md`.
2. Open this index.
3. Pick the most relevant skill directory below and read its `SKILL.md`.
4. Read any referenced files only when needed.

## Available Skills

- `benchmark-evidence-governance`
  path: `.claude/skills/benchmark-evidence-governance/SKILL.md`
  description: Use when benchmark work needs explicit contract control, fair comparison rules, cache/version boundaries, provenance expectations, and evidence grading rather than ad hoc metric interpretation.
- `graphdrone-benchmark-governance`
  path: `.claude/skills/graphdrone-benchmark-governance/SKILL.md`
  description: Use when planning, running, or reviewing GraphDrone benchmarks and you need strict contract control over datasets, folds, presets, versions, cache reuse, champion/challenger fairness, and the meaning of benchmark evidence.
- `graphdrone-mechanism-diagnosis`
  path: `.claude/skills/graphdrone-mechanism-diagnosis/SKILL.md`
  description: Use when a GraphDrone component or mechanism looks locally promising but end-to-end metrics stay flat or negative, and you need to locate the break between component truth, routing/policy coupling, outcome translation, and architecture fit.
- `graphdrone-research-memory`
  path: `.claude/skills/graphdrone-research-memory/SKILL.md`
  description: Use when recording or reviewing GraphDrone research memory across run manifests, branch notes, durable findings, and current hypotheses so experiments accumulate into reusable project knowledge.
- `graphdrone-research-ops`
  path: `.claude/skills/graphdrone-research-ops/SKILL.md`
  description: Use when running GraphDrone research as an operating system rather than a single benchmark: branch setup, benchmark contract control, claim-first evaluation, run provenance, long-form notes, and durable finding capture across `eval/`, `output/experiments/`, and `docs/research/`.
- `mechanism-first-diagnosis`
  path: `.claude/skills/mechanism-first-diagnosis/SKILL.md`
  description: Use when a new method shows a local or component-level gain but the end-to-end result is flat or negative, and the real task is to explain the break between mechanism truth, integration, and final outcome.
- `research-memory-ledger`
  path: `.claude/skills/research-memory-ledger/SKILL.md`
  description: Use when experiments need to accumulate into durable research knowledge instead of remaining isolated run outputs, with explicit separation between execution lineage, long-form notes, and claim memory.
- `research-platform-ops`
  path: `.claude/skills/research-platform-ops/SKILL.md`
  description: Use when a project needs a research operating system rather than isolated experiments: branch hygiene, contract framing, execution evidence, structured notes, and durable claim memory that compound over time.
