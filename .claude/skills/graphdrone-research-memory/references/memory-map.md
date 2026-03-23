# GraphDrone Research Memory Map

## Layer 1: Execution lineage

Use for reproducibility and raw provenance.

- `output/experiments/designs/*.md`
- `output/experiments/runs/*.json`
- `output/experiments/comparisons/*.md`

Questions answered:
- what command ran?
- with which seed, profile, version, branch?
- where are the raw artifacts?

## Layer 2: Long-form interpretation

Use for branch-local or experiment-local reasoning.

- `docs/YYYY-MM-DD-*.md`

Questions answered:
- what hypothesis was tested?
- what happened?
- what is the causal interpretation?

## Layer 3: Durable claim memory

Use for scaling decisions and future work.

- `docs/research/findings.jsonl`
- `docs/research/current_hypotheses.md`

Questions answered:
- what has been cleared?
- what is partly causal?
- what is still open?
- what should not be re-litigated from scratch?

## Status Meanings

- `cleared`: strong enough to treat as established locally
- `partially_causal`: real and influential, but not the whole story
- `open`: active next question
- `confounded`: informative but interpretation depends on a flaw or missing control
- `falsified`: this version of the claim should not keep absorbing time
- `note`: informational update without claim-status change
