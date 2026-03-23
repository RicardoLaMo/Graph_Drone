---
name: research-platform-ops
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
- repository snapshot
- execution lineage
- evaluation artifacts
- long-form note
- durable finding update

## When To Use

Use this skill when the user asks to:
- run research systematically instead of ad hoc
- make experiments cumulative rather than metric-only
- build a reusable research workflow for a repo
- separate local mechanism truth from end-to-end promotion

## Workflow

1. Freeze repository state.
- record branch, head SHA, dirty files, and active contract

2. Frame the change.
- idea
- target component
- claimed bottleneck
- expected local signature
- expected global signature
- likely failure modes

3. Choose the right evaluation surface.
- component truth
- integration diagnosis
- governed benchmark
- research-memory update

4. Preserve execution evidence.
- commands
- versions
- seeds
- runtime provenance
- output paths

5. Externalize what was learned.
- branch note
- durable claim or status update

## Adaptation Notes

To specialize this skill for a project, define:
- the repository source of truth
- the benchmark runners
- the evidence artifacts
- the durable claim store
- the branch-note convention

Read `references/adaptation-guide.md` before specializing it for a new repo.
