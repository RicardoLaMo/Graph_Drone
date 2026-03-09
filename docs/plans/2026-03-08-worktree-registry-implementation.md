# Worktree Registry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a repo-local audit script, generated registry, and lineage doc so all agents can align on current checkout vs sibling worktrees.

**Architecture:** A small Python script reads `git worktree list --porcelain`, scans selected experiment directories, and writes `docs/worktree_registry.json`. A human doc explains the lineage and points agents to the generated registry. `AGENTS.md` gets a short pointer to both.

**Tech Stack:** Python 3.12 standard library, pytest, git CLI, markdown/json docs

---

### Task 1: Add failing tests for the audit script contract

**Files:**
- Create: `tests/test_audit_worktrees.py`
- Create: `scripts/audit_worktrees.py`

**Step 1: Write the failing test**
- Test that a helper can parse `git worktree list --porcelain` output into records.
- Test that family detection marks `gora_tabular`, `mq_gora_v4`, `head_routing_v5`, and `mv_tabr_gora` when those folders exist.

**Step 2: Run test to verify it fails**
- Run: `pytest tests/test_audit_worktrees.py -q`
- Expected: import failure or missing function failure

**Step 3: Write minimal implementation**
- Add parser and family-detection helpers in `scripts/audit_worktrees.py`.

**Step 4: Run test to verify it passes**
- Run: `pytest tests/test_audit_worktrees.py -q`
- Expected: pass

### Task 2: Add registry generation behavior

**Files:**
- Modify: `scripts/audit_worktrees.py`
- Test: `tests/test_audit_worktrees.py`

**Step 1: Write the failing test**
- Test that the script can build a registry dict with:
  - repo metadata
  - current checkout info
  - normalized worktree entries

**Step 2: Run test to verify it fails**
- Run: `pytest tests/test_audit_worktrees.py -q`

**Step 3: Write minimal implementation**
- Add registry assembly and JSON output helpers.

**Step 4: Run test to verify it passes**
- Run: `pytest tests/test_audit_worktrees.py -q`

### Task 3: Add the human lineage doc

**Files:**
- Create: `docs/worktree_lineage.md`

**Step 1: Write the doc**
- Document:
  - current checkout
  - tracked experiment families
  - sibling worktrees
  - specific mapping for `funny-davinci`, `v5`, `mv_tabr_gora`, and A7 branches

**Step 2: Review for consistency with the current git state**
- Compare against `git worktree list` and `experiments/`

### Task 4: Generate and check in the registry JSON

**Files:**
- Create: `docs/worktree_registry.json`

**Step 1: Run generator**
- Run: `python scripts/audit_worktrees.py --write docs/worktree_registry.json`

**Step 2: Verify output**
- Confirm the JSON includes current checkout plus sibling worktrees.

### Task 5: Add AGENTS pointer

**Files:**
- Modify: `AGENTS.md`

**Step 1: Add short guidance**
- Tell future agents to read:
  - `docs/worktree_lineage.md`
  - `docs/worktree_registry.json`
before inferring repo lineage

### Task 6: Verification

**Files:**
- Test: `tests/test_audit_worktrees.py`

**Step 1: Run tests**
- Run: `pytest tests/test_audit_worktrees.py -q`

**Step 2: Run generator**
- Run: `python scripts/audit_worktrees.py --write docs/worktree_registry.json`

**Step 3: Spot-check JSON and lineage doc**
- Open:
  - `docs/worktree_registry.json`
  - `docs/worktree_lineage.md`

### Task 7: Commit

**Step 1: Commit**
```bash
git add docs/plans/2026-03-08-worktree-registry-design.md \
        docs/plans/2026-03-08-worktree-registry-implementation.md \
        docs/worktree_lineage.md docs/worktree_registry.json \
        scripts/audit_worktrees.py tests/test_audit_worktrees.py AGENTS.md
git commit -m "feat: add repo worktree lineage registry"
```
