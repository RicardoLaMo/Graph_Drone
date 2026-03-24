---
name: research-platform-ops
scope: global
description: "Use when a project needs a research operating system rather than isolated experiments: branch hygiene, contract framing, execution evidence, structured notes, and durable claim memory that compound over time."
---

# Research Platform Ops

This is the generic parent of project-specific research-operation skills.

Use it when the work spans:
- code changes
- experiments
- diagnosis
- git discipline
- durable research memory

## Core Model

Every serious research change should leave five artifacts:
1. **repository snapshot** — branch, commit, worktree, dirty-state record
2. **execution lineage** — commands, versions, seeds, runtime provenance, output paths
3. **evaluation artifacts** — comparison reports, paired deltas, benchmark tables
4. **long-form note** — where the project stores experiment reflections and causal narratives
5. **durable finding update** — append-only claim ledger plus current hypothesis view

## Memory Layers

Keep three layers distinct:

| Layer | What it records | Update rule |
|---|---|---|
| Execution lineage | what ran | append-only; never delete |
| Long-form notes | why it behaved that way | one note per branch/topic |
| Durable claims | what the project should remember | append or update status; never rewrite |

## When To Use

Use this skill when the user asks to:
- run research systematically instead of ad hoc
- make experiments cumulative rather than metric-only
- build a reusable research workflow for a repo
- separate local mechanism truth from end-to-end promotion

## Workflow

1. **Freeze repository state.**
   - Record branch, head SHA, dirty files, and active contract.

2. **Frame the change.**
   - State: idea, target component, claimed bottleneck, expected local signature, expected global signature, likely failure modes.
   - Decide whether this is: efficiency-only / component truth / integration diagnosis / benchmark-contract validation / research-memory consolidation.

3. **Choose the evaluation surface.**
   - component truth: probe the module directly before running the whole system
   - integration: run the full pipeline but read claim-level artifacts, not just the headline metric
   - governed benchmark: freeze the contract fully before launching
   - research-memory: consolidate prior notes into the claim ledger

4. **Preserve execution evidence.**
   - Prefer file-backed proof over one-off terminal observations.
   - If the run matters for future reasoning, record it in the execution-lineage directory.

5. **Externalize what was learned.**
   - Write a dated long-form note.
   - Record or update a durable finding in the claim ledger.
   - Regenerate the current hypothesis view.

## Pairing

- If the main question is "why did local mechanism truth fail to translate?", pair with `mechanism-first-diagnosis`.
- If the main question is "record or review what we already learned", pair with `research-memory-ledger`.
- If the main question is "is this benchmark run valid evidence?", pair with `benchmark-evidence-governance`.

## Adaptation Notes

To specialize this skill for a project, define:
- the repository source of truth and worktree pattern
- the benchmark runners and gate levels
- the execution-lineage directory
- the notes directory and naming convention
- the durable claim store (format and recorder script)

Read `references/adaptation-guide.md` before specializing for a new repo.
