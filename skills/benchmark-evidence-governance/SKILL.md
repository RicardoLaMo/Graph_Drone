---
name: benchmark-evidence-governance
scope: global
description: Use when benchmark work needs explicit contract control, fair comparison rules, cache/version boundaries, provenance expectations, and evidence grading rather than ad hoc metric interpretation.
---

# Benchmark Evidence Governance

Treat benchmark results as governed evidence, not just numbers.

## Freeze The Contract

Before running or comparing, freeze and record:
- runner and task family
- datasets and folds
- method identities (names, versions, presets)
- seeds and max-sample limits
- cache policy (reuse allowed? version key?)
- expected output artifacts
- promotion rule (who is champion, who is challenger, what external baseline represents)

## Champion / Challenger Model

- Compare the system against a known baseline (the champion) rather than against external leaderboard numbers directly.
- Keep external baselines as contextual anchors, not the promotion rule, unless explicitly requested.
- Keep component truth (did the mechanism move?) separate from promotion (should the branch advance?).

## Evidence Grades

Every run should be labeled before interpretation. Read `references/evidence-grades.md`:
- `evidence-grade` — contract is explicit, provenance exists, suitable for a branch or merge decision
- `diagnostic-grade` — intentionally narrow; useful for mechanism understanding, not for promotion
- `non-comparable` — contract drift, cache/version ambiguity, or missing provenance prevents fair interpretation

## Core Rules

- Do not compare across different contracts as if they were equivalent.
- Do not reuse caches when version identity is ambiguous.
- Do not promote from a run that lacks provenance artifacts (ledgers, logs, comparison reports).
- Do not let a diagnostic probe masquerade as branch-grade evidence.
- Prefer file-backed provenance over verbal claims.

## Guardrails

- Do not claim "the method failed" when the run is only diagnostic-grade and the component claim is still supported.
- Do not rerun everything by habit; reuse valid baselines and rerun only the changed side when the contract permits.
- Do not interpret two runs as comparable if they used different dataset subsets, folds, or seeds.

## Adaptation Notes

For a project-specific specialization, bind:
- benchmark runners and gate levels (e.g. quick / mini-full / full)
- cache and version identity mechanism
- comparison artifacts (claim report, paired deltas, promotion decision or equivalent)
- promotion logic
- provenance requirements
