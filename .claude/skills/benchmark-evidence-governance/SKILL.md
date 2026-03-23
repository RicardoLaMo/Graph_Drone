---
name: benchmark-evidence-governance
description: Use when benchmark work needs explicit contract control, fair comparison rules, cache/version boundaries, provenance expectations, and evidence grading rather than ad hoc metric interpretation.
---

# Benchmark Evidence Governance

This is the generic parent of project-specific benchmark-governance skills.

## Purpose

Treat benchmark results as governed evidence, not just numbers.

## Freeze The Contract

Before running or comparing, define:
- runner
- task family
- datasets
- folds
- method identities
- version identities
- seeds
- max samples or resource profile
- expected output artifacts

## Evidence Grades

Every run should be labeled:
- `evidence-grade`
- `diagnostic-grade`
- `non-comparable`

Read `references/evidence-grades.md`.

## Core Rules

- Do not compare across different contracts as if they were equivalent.
- Do not reuse caches when version identity is ambiguous.
- Do not promote from a run that lacks provenance.
- Do not let a diagnostic probe masquerade as branch-grade evidence.

## Adaptation Notes

For a project-specific specialization, bind:
- benchmark runners
- cache/version identity
- comparison artifacts
- promotion logic
- provenance requirements
