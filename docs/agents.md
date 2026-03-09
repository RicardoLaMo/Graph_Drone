# Multi-Agent Coordination Protocol

## Purpose

This file defines which AI agent is responsible for what, where each agent's
working directory lives, and how the shared source of truth is maintained.

Before reasoning about experiments, lineage, or making changes, all agents must read:
1. `docs/worktree_registry.json` — live-audited registry of all worktrees
2. `docs/worktree_lineage.md` — human-readable lineage and role descriptions
3. `AGENTS.md` — environment facts and per-experiment anchors

---

## Agent Roles

### Claude (Anthropic Claude Code)

| Property | Value |
|---|---|
| Home directory | `/Volumes/MacMini/Projects/Graph_Drone` (main checkout) |
| Home branch | `feature/gora-v5-trust-routing` |
| Working space | `.claude/` |
| Commit author | `Claude Sonnet 4.6 <noreply@anthropic.com>` (Co-Authored-By) |
| Primary role | Implementation: code, worktrees, commits, experiments |

**Rules:**
- Do NOT treat `.claude/worktrees/funny-davinci` as the active home. It is an archived snapshot.
- Work from the main checkout (`/Volumes/MacMini/Projects/Graph_Drone`) or from the
  active research worktree (currently `.worktrees/mv-tabr-gora-a7a-iterative-reindex`).
- Before creating a new worktree, check `docs/worktree_registry.json` to confirm it
  does not already exist.
- After adding/removing worktrees, regenerate the registry:
  ```bash
  source .venv/bin/activate
  python scripts/audit_worktrees.py --write docs/worktree_registry.json
  ```

### Gemini (Google Gemini CLI)

| Property | Value |
|---|---|
| Working space | `.gemini-cross-checks/` |
| Commit author | N/A (writes findings only, does not commit) |
| Primary role | Cross-validation: read code + reports, flag issues, write review sessions |

**Protocol:**
- Each Gemini review session writes to a timestamped subdirectory under `.gemini-cross-checks/`.
- Directory naming: `<topic>-<YYYYMMDD>-<HHMMSS>/`
- Gemini does NOT modify source files or create commits.
- Claude reads `.gemini-cross-checks/` findings and decides which to incorporate.
- After incorporating Gemini feedback, Claude records this in the relevant commit message.

**Known Gemini reviews:**
- `.gemini-cross-checks/worktree-registry-20260308-205343/` — registry portability review
- `.gemini-cross-checks/v6v7-lineage-20260308-203432/` — experiment lineage audit
- `.gemini-cross-checks/a7-a1-a2-review-20260308-210408/` — A7a/A6f baseline review

### Codex (OpenAI Codex)

| Property | Value |
|---|---|
| Working space | `.codex/` (reserved, not yet active) |
| Primary role | TBD — assign when Codex session is initiated |

**Note:** When starting a Codex session, create a `.codex/session-<date>.md` with:
- The task being assigned
- Which branch/worktree to work in
- What NOT to touch (unrelated dirty files, registry files)

---

## Shared Source of Truth

```
docs/
  worktree_registry.json   ← auto-generated, regenerate after worktree changes
  worktree_lineage.md      ← human-readable, update manually when lineage changes
  agents.md                ← this file
AGENTS.md                  ← environment facts, experiment anchors
```

All agents read these files first. All agents treat the registry as authoritative.

---

## Branch / Worktree Naming Conventions

| Pattern | Meaning |
|---|---|
| `feature/gora-v*` | Tracked GoRA experiment family (in main checkout `experiments/`) |
| `feature/mv-tabr-gora` | MV-TabR-GoRA core California research line |
| `feature/mv-tabr-gora-*` | MV-TabR-GoRA follow-on branches (B-series, A7a, etc.) |
| `claude/funny-davinci` | Archived Claude snapshot — do not use as home |
| `main` | Stable base (at `/private/tmp/graphdrone-main-worktree`) |

---

## Current Active Research Summary (2026-03-08)

### Tracked experiments (main checkout, `feature/gora-v5-trust-routing`):
- `head_routing_v5` — GoRA v5 head-gated prediction, current tracked family
- Best: C8_Routed, CA RMSE 0.5250; E-series targeted fix ran; backbone is the bottleneck

### Branch-local California line (sibling worktrees):
- A6f champion: **0.4063 RMSE** (`feature/mv-tabr-gora`)
- B-series: pool expansion hurts; score biases marginal (`feature/mv-tabr-gora-rerank`)
- A7a: probe ✅ but neural retrain neutral-to-negative (`feature/mv-tabr-gora-a7a-iterative-reindex`)
- **Gap to TabR (0.3829)**: 0.023 — not yet closed; retrieval quality bottleneck remains
- **Next hypothesis**: hybrid graph mixing (part raw, part embedding-space kNN) or view-specific reindexing

### Gemini finding to incorporate (A7a review):
- Weight mismatch between raw Gaussian-normalized graph and rebuilt 1/dist graph → FIXED
- Non-determinism variance (0.0086 RMSE) exceeds some claimed B-series improvements → validate with multi-seed

---

## Registry Refresh Checklist

Run after any of:
- Adding a new worktree (`git worktree add`)
- Removing a worktree (`git worktree remove`)
- Switching the main checkout branch

```bash
cd /Volumes/MacMini/Projects/Graph_Drone
source .venv/bin/activate
python scripts/audit_worktrees.py --write docs/worktree_registry.json
```

Then commit `docs/worktree_registry.json` with message: `chore: refresh worktree registry`.
