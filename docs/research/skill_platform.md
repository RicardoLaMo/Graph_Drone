# GraphDrone Research Skill Platform

This document turns the direction in `doc/for_skill.md` into a GraphDrone-specific research operating model.

## What We Built So Far

The GraphDrone research stack is no longer just “edit code, run benchmark, report metric”:

1. We separated champion/challenger from external leaderboard pressure.
2. We added claim-first evaluation so a component can be tested before the whole branch is judged.
3. We added run provenance (`run_ledger.json`, `run_events.jsonl`) so execution is auditable.
4. We added branch notes and durable claim memory (`docs/research/findings.jsonl`).
5. We used inside-out plus outside-in diagnosis when the bottom line and the local mechanism disagreed.

That means the project already behaves like a research platform; the missing piece was a project-native skill layer that matches those artifacts.

## Reflection On Existing Skills

The currently installed skills are useful but incomplete for GraphDrone:

- `git-codebase-ops`
  good for safe snapshots, but not GraphDrone experiment lineage
- `git-tree-manager`
  good for repo shape, but not benchmark or claim artifacts
- `experiment-design-tracker`
  good for append-only run manifests, but generic and not aware of `docs/research/`
- `eval-driven-iteration-loop`
  good for pass/fail harnesses, but too metric-first for mechanism research
- `raphael-research-loop`
  close to the right research posture, but still intentionally generic

The gap described in `doc/for_skill.md` is real: a generic coding agent tends to wire the new module, run the full metric, and stop at “it did not work” without locating the break between mechanism, circuit, and outcome.

The placeholder skill at `doc/layered-research-orchestrator/layered-research-orchestrator/SKILL.md` also points in the right direction, but it was still a scaffold with TODO sections and no binding to GraphDrone artifacts. The new repo `skills/` suite is the concrete, project-native version of that direction.

## Parent And Specialized Skills

The repo now carries two layers under `skills/`.

Generic parent skills:

1. `research-platform-ops`
   generic research operating-system skill

2. `mechanism-first-diagnosis`
   generic local-win/global-loss diagnosis skill

3. `research-memory-ledger`
   generic durable-knowledge skill

4. `benchmark-evidence-governance`
   generic benchmark-contract and evidence-governance skill

GraphDrone specializations:

1. `graphdrone-research-ops`
   top-level operating skill for branch hygiene, benchmark contract control, evidence capture, and durable finding updates

2. `graphdrone-mechanism-diagnosis`
   mechanism-first local-vs-global diagnosis skill for positive component truth with weak or negative system translation

3. `graphdrone-research-memory`
   project-memory skill for turning runs and notes into durable claim knowledge

4. `graphdrone-benchmark-governance`
   benchmark-contract and evidence-governance skill for champion/challenger fairness, version discipline, and cache/provenance boundaries

The GraphDrone skills now sit on top of both:
- the pre-existing installed generic skills
- the new repo-tracked parent skills that can be carried to other projects

## Scope Of These Skills

There are two scopes at once:

1. Project-level source of truth
   The authoritative versions live in this repo under `skills/`. That makes them reviewable, branchable, and tied to GraphDrone methodology.

2. User-level installation
   After running `python scripts/sync_graphdrone_skills.py`, copies are installed into `~/.codex/skills/` for this user account, which makes them callable across sessions.

That means the skills are both:
- project-native in design and maintenance
- user-level in runtime availability

## Can They Be Carried To Another Project?

Yes, but not all at the same level of portability.

- High portability:
  - the methodology shape in `graphdrone-mechanism-diagnosis`
  - the memory-layer separation in `graphdrone-research-memory`
  - the top-level research operating model in `graphdrone-research-ops`

- Medium portability:
  - `graphdrone-benchmark-governance`, if the new project also uses explicit champion/challenger runners, versioned cache keys, and artifact-grade evaluation outputs

- Low portability without adaptation:
  - any references to GraphDrone-specific files like `docs/research/findings.jsonl`, `claim_report.json`, `promotion_decision.json`, `run_ledger.json`, or `scripts/run_champion_challenger.py`

So the right framing is now:
- the parent skills are directly portable
- the GraphDrone skills are project-specific specializations
- another project can either reuse the parent skills as-is or create its own specialization layer

## Why This Is Better Than A Metric-Only Loop

This platform is explicitly built for research, not only A/B promotion:

- local mechanism truth can be preserved even when end-to-end performance is negative
- partial causality can be recorded without forcing premature rejection
- confounded results can still create reusable knowledge
- branch notes, run ledgers, and durable claims are kept as separate layers

That is the right operating model for scaling GraphDrone, because it compounds understanding rather than only collecting winners.

## Install / Sync

The tracked source of truth is the repo `skills/` directory. To publish the current repo version into the local skill home:

```bash
python scripts/sync_graphdrone_skills.py
```

This copies all repo skill directories, both parent and GraphDrone-specific, into `~/.codex/skills/`.

## Claude Code Bridge

Yes, the same skills can be carried into Claude Code agents through a generated bridge.

- export command:
  ```bash
  python scripts/export_skills_to_claude.py
  ```
- source of truth remains `skills/`
- generated Claude-facing pack lives in `.claude/skills/`
- `CLAUDE.md` points Claude agents to `.claude/skills/INDEX.md`

This is not a native Codex-skill import. It is a repo-local projection of the same methods into a Claude-readable instruction tree.
