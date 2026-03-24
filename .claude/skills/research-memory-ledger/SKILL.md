---
name: research-memory-ledger
scope: global
description: Use when experiments need to accumulate into durable research knowledge instead of remaining isolated run outputs, with explicit separation between execution lineage, long-form notes, and claim memory.
---

# Research Memory Ledger

## Memory Layers

Keep three layers distinct:

| Layer | What it records | Location pattern | Update rule |
|---|---|---|---|
| Execution lineage | what ran, when, with what args | `[LINEAGE_DIR]/` | append-only; never delete |
| Long-form notes | why it behaved that way | `[NOTES_DIR]/YYYY-MM-DD-topic.md` | one note per topic; append interpretations |
| Durable claims | what the project should remember | `[CLAIM_LEDGER]` (append-only JSONL or equivalent) | append or update status; never rewrite |

## Claim Status Model

Every durable claim has a status. Read `references/status-model.md` for full definitions:
- `cleared` — strong enough to treat as established locally
- `partially_causal` — real and influential, but not the whole story
- `open` — still a live next question
- `confounded` — informative, but interpretation depends on a flaw or missing control
- `falsified` — do not keep spending on this version without a new mechanism
- `note` — informational update without a claim-state change

## Naming Conventions

- Long-form note: `YYYY-MM-DD-topic.md`
- Claim ID: `<track>-<mechanism>-<question>` (e.g., `afc-gora-residual-usefulness`)
- Topic family: short name reused across notes for the same thread

## When To Use

Use this skill when:
- a run just finished and should be recorded
- a branch note exists but the durable claim ledger is stale
- you need to review what is cleared, open, confounded, or falsified
- you want future branches to inherit prior learning

## Workflow

1. **Read the current state.**
   - Open the current hypothesis view (if it exists).
   - Read the relevant branch notes.
   - Read the relevant run manifests.

2. **Classify the update.**
   - new durable claim
   - status update to an existing claim
   - note-only entry
   - rebuild of the hypothesis view without a new claim

3. **Record the finding.**
   Use the project's recorder script or write directly to the claim ledger with:
   - claim ID, topic, status
   - the research question and conclusion
   - branch, commit, artifact paths

4. **Keep the layers distinct.**
   - Execution lineage records what ran.
   - Notes record why you think it behaved that way.
   - Claims record what the project should remember.

5. **Regenerate the current hypothesis view** after any status change.

## Guardrails

- Do not overwrite history to simplify the story; append a new finding instead.
- Do not treat the latest run as the whole story; read prior findings first.
- Do not promote a confounded result without stating the confound explicitly.
- Do not conflate execution lineage with durable research memory; both are required.

## Adaptation Notes

Specialize this skill by binding:
- `[LINEAGE_DIR]` — execution manifest location
- `[NOTES_DIR]` — note storage location and naming convention
- `[CLAIM_LEDGER]` — the append-only claim store format and path
- the current-hypothesis view and how to regenerate it
- the recorder script (or manual format if none exists)

Read `references/status-model.md`.
