# Research Spine

This repo now keeps research memory in three layers:

1. `output/experiments/`
   - append-only execution lineage
   - commands, seeds, profiles, run JSON

2. `docs/*.md`
   - long-form branch or experiment notes
   - mechanism interpretation, causal narrative, artifact links

3. `docs/research/`
   - durable claim memory for scaling and further development
   - not "who beat who", but what was learned and what remains open

## Files

- `findings.jsonl`
  - append-only canonical ledger of research findings
  - each entry records claim id, status, branch, commit, note, artifacts, and next checks

- `current_hypotheses.md`
  - generated current view
  - latest state per claim id, grouped by `cleared`, `partially_causal`, `open`, `confounded`, `falsified`

## Recording a new finding

Use:

```bash
python scripts/record_research_finding.py record \
  --claim-id afc-b-reg-anchor-asymmetry \
  --topic afc_phase_b \
  --status partially_causal \
  --question "Was rotor hurt by anchor mass inside the deferred specialist blend?" \
  --conclusion "The asymmetry was real and partly causal." \
  --summary "Anchor exclusion improved the rotor branch much more than the champion path, but Phase B still does not clear the gate." \
  --branch exp/afc-b-reg-anchor-exclusion \
  --commit 0fbbe59 \
  --note-path docs/2026-03-23-afc-phase-b-anchor-exclusion.md \
  --artifact-path eval/phaseb_reg_anchorfix_l001_mini/comparison/promotion_decision.json \
  --artifact-path eval/phaseb_reg_anchorfix_l001_mini/comparison/claim_report.json \
  --next-check "Test frozen-router rotor training." \
  --next-check "Add per-expert mass diagnostics."
```

This appends to `findings.jsonl` and regenerates `current_hypotheses.md`.

## Status meanings

- `cleared`: strong enough to treat as established locally
- `partially_causal`: real and influential, but not the whole story
- `open`: active next question
- `confounded`: result contained signal, but interpretation depended on a known flaw or missing control
- `falsified`: do not continue this version of the claim without a new mechanism
- `note`: informational record with no claim status change
