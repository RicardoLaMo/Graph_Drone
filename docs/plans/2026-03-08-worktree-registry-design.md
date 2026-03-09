# Worktree Registry Design

**Date:** 2026-03-08  
**Status:** approved for implementation in current checkout

## Goal
Create one repo-local source of truth that distinguishes:
- the current checkout and its tracked experiment families
- sibling git worktrees and their branch-local experiment lines
- the practical lineage between `gora_tabular`, `mq_gora_v4`, `head_routing_v5`, and `mv_tabr_gora`

This should be easy for Codex, Gemini, and Claude Code to read without relying on memory.

## Problem
The repo currently mixes:
- tracked numbered experiment families under `experiments/`
- branch-local research lines under `.worktrees/`
- older Claude-specific worktrees under `.claude/worktrees/`

That causes repeated confusion about what is:
- mainline and tracked
- branch-local and newer but not merged
- an old snapshot vs an active research branch

## Chosen Approach
Implement a small registry system in the current checkout:

1. `docs/worktree_lineage.md`
   - human-readable lineage and conventions
   - clearly separates tracked experiments from sibling worktrees

2. `docs/worktree_registry.json`
   - machine-readable registry generated from git state
   - meant to be handed to Codex, Gemini, or Claude as the first context artifact

3. `scripts/audit_worktrees.py`
   - enumerates `git worktree list`
   - detects relevant experiment families
   - writes the registry JSON
   - prints a short summary

4. `AGENTS.md` pointer
   - tells future agents to use the registry and lineage doc before inferring branch history

## Registry Schema
Each registry record should contain:
- `path`
- `branch`
- `head`
- `is_current_checkout`
- `purpose`
- `experiment_families`
- `lineage_role`
- `status`

Top-level metadata should contain:
- repo root
- generated timestamp
- current checkout branch/head/path

## Initial Lineage Mapping
- `.claude/worktrees/funny-davinci`
  - early v3 snapshot (`1903f5a`)
- current checkout
  - tracked `gora_tabular`, `mq_gora_v4`, `head_routing_v5`
- `.worktrees/mv-tabr-gora`
  - newer California branch-local line with `A0..A6f`
- `.worktrees/mv-tabr-gora-a7-*`
  - follow-on retrieval experiments derived from `mv_tabr_gora`

## Non-Goals
- no automated branch deletion
- no worktree creation/removal
- no attempt to enforce merge policy
- no hidden synchronization with Gemini or Claude tooling

## Success Criteria
- one command regenerates `docs/worktree_registry.json`
- the lineage doc explains the current repo/worktree split clearly
- future sessions can answer “what is current?” without manually searching multiple worktrees
