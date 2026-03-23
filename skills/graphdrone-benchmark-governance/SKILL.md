---
name: graphdrone-benchmark-governance
description: Use when planning, running, or reviewing GraphDrone benchmarks and you need strict contract control over datasets, folds, presets, versions, cache reuse, champion/challenger fairness, and the meaning of benchmark evidence.
---

# GraphDrone Benchmark Governance

Use this skill when benchmark work needs to be treated as governed evidence rather than an ad hoc run.

This skill is the GraphDrone-specific layer over generic benchmark-control skills. It knows the repo-native benchmark surfaces:
- `scripts/run_champion_challenger.py`
- `scripts/run_geopoe_benchmark.py`
- `scripts/run_smart_benchmark.py`

It also assumes the important outputs are:
- `comparison/promotion_decision.json`
- `comparison/claim_report.json`
- `comparison/paired_task_deltas.csv`
- `raw/.../run_ledger.json`

## When To Use

Use this skill when the user asks to:
- run a fair GraphDrone benchmark
- compare two GraphDrone versions
- reuse baselines without cheating the contract
- decide whether a result is promotable, only diagnostic, or not comparable
- review whether two benchmark reports can be compared at all

## Benchmark Contract

Before running, freeze:
- benchmark family: regression, classification, or both
- runner and gate: `quick`, `mini-full`, or `full`
- explicit datasets if overriding the default contract
- folds
- champion preset/version
- challenger preset/version
- whether TabPFN is an anchor only or part of the decision rule
- max samples
- code commit / branch

Read `references/contract-checklist.md` for the exact checklist.

## GraphDrone Rules

1. Use champion/challenger as the primary promotion surface.
- Compare GraphDrone against GraphDrone first.
- Keep TabPFN as contextual anchor unless the user explicitly wants external ranking.

2. Keep component truth separate from promotion.
- `claim_report.json` answers whether the mechanism moved.
- `promotion_decision.json` answers whether the branch should advance.
- Do not use one as a substitute for the other.

3. Treat cache keys as versioned evidence.
- `GRAPHDRONE_VERSION_OVERRIDE` is part of the benchmark identity.
- Reusing a cache with the wrong version key is not a valid comparison.

4. Keep benchmark and research scopes separate.
- a quick run can justify diagnosis
- a mini-full run can justify serious branch interpretation
- a full run is for merge-grade evidence

5. Prefer file-backed provenance over verbal claims.
- If a run matters, there should be ledgers, events, granular CSVs, and comparison artifacts.

## Standard Commands

Champion/challenger:

```bash
python scripts/run_champion_challenger.py \
  --task regression \
  --gate mini-full \
  --champion-version v1.20.0-reg-champion \
  --challenger-version afc-candidate \
  --output-dir eval/my_run
```

Regression backend only:

```bash
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --tasks regression --folds 0
```

Classification backend only:

```bash
PYTHONPATH=src python scripts/run_smart_benchmark.py --folds 0
```

## Decision Labels

Use these labels explicitly:
- `evidence-grade`: fair enough to support a branch decision
- `diagnostic-grade`: useful for mechanism or failure analysis, but not for promotion
- `non-comparable`: contract drift or missing evidence prevents interpretation

## Guardrails

- Do not compare runs from different dataset or fold contracts as if they were equivalent.
- Do not promote from a run that lacks run ledgers or comparison artifacts.
- Do not claim “the method failed” when the benchmark is only diagnostic-grade and the claim report says the mechanism is supported.
- Do not rerun everything by habit; reuse valid baselines and rerun only the changed side when the contract permits it.

## Resources

- `references/contract-checklist.md`: contract freeze checklist before launch
- `references/evidence-grade.md`: how to classify a GraphDrone run as evidence-grade, diagnostic-grade, or non-comparable
