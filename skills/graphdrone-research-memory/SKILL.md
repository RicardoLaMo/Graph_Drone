---
name: graphdrone-research-memory
description: Use when recording or reviewing GraphDrone research memory across run manifests, branch notes, durable findings, and current hypotheses so experiments accumulate into reusable project knowledge.
---

# GraphDrone Research Memory

Use this skill to turn experiments into durable research capital.

This skill is for the question:
“What did we actually learn, where is it recorded, and how do we update the project memory without losing lineage?”

## Memory Layers

GraphDrone keeps research memory in three layers:
1. `output/experiments/`
   append-only execution lineage
2. `docs/*.md`
   branch notes and causal interpretations
3. `docs/research/`
   durable claim ledger and current hypothesis surface

Read `references/memory-map.md` when you need the exact update rules.

## When To Use

Use this skill when:
- a run just finished and must be recorded
- a branch note exists but the durable claim ledger is stale
- you need to review what was already cleared, confounded, or still open
- you want to consolidate multiple notes into a current research view

## Workflow

1. Read the current state.
- `docs/research/current_hypotheses.md`
- relevant branch notes in `docs/`
- relevant run manifests in `output/experiments/`

2. Decide whether this is:
- a new durable claim
- a status update to an existing claim
- a note-only entry
- a rebuild of the current map without a new claim

3. Record the finding.

```bash
python scripts/record_research_finding.py record \
  --claim-id <id> \
  --topic <topic> \
  --status <cleared|partially_causal|open|confounded|falsified|note> \
  --question "<question>" \
  --conclusion "<conclusion>" \
  --summary "<summary>" \
  --branch <branch> \
  --commit <sha> \
  --note-path docs/<note>.md \
  --artifact-path eval/.../comparison/promotion_decision.json
```

4. Keep the layers distinct.
- `output/experiments/` records what ran
- `docs/*.md` records why you think it behaved that way
- `docs/research/` records what the project should remember

## Naming Conventions

- branch note: `docs/YYYY-MM-DD-topic.md`
- claim id: `<track>-<mechanism>-<question>`
- topic: short family name such as `afc_phase_b`

## Guardrails

- Do not treat the latest run as the whole story; read prior findings first.
- Do not overwrite old notes to make the story cleaner; append a new finding instead.
- Do not promote a finding from a confounded run without stating the confound explicitly.

## Resources

- `references/memory-map.md`: GraphDrone research-memory rules and status meanings
- `references/note-template.md`: compact note shape for branch-local research notes
