# TabArena Integration Strategy

## Goal

Add a lightweight, repo-local TabArena bridge so Graph_Drone can:

- benchmark against strong tabular baselines on matched splits
- select datasets that are actually relevant to our retrieval/routing hypotheses
- standardize metrics and reporting before moving the heavier runs to the H200 environment

This branch intentionally does **not** install or vendor TabArena itself yet. It creates the local benchmark contract first.

## Why A Bridge Instead Of Immediate Full Integration

Graph_Drone already has:

- custom branch-local experiment families
- existing aligned baselines for `TabR`, `TabPFN`, `TabM`
- strong protocol sensitivity around split seeds and routing assumptions

So the first need is not another training stack. The first need is:

1. a canonical dataset shortlist
2. a canonical split policy
3. a canonical metrics policy
4. a canonical benchmark matrix

Once those are stable locally, the H200 migration is much cheaper.

## TabArena Features We Want To Reuse

### Data
- OpenML-backed suite structure
- dataset metadata and task identity
- multi-split evaluation instead of one-off anchor runs

### Benchmark Policy
- apples-to-apples split sweeps
- comparison to strong external baselines, not only internal ablations
- dataset families grouped by task and difficulty

### Metrics
- task-primary metrics from the benchmark line
- Graph_Drone secondary diagnostics layered on top

For Graph_Drone regression, the primary metric remains:

- `rmse`

Secondary metrics that should always be carried locally:

- `mae`
- `r2`
- win count across split seeds
- mean and std over split seeds

## Tailoring For Graph_Drone

Graph_Drone should not adopt the full TabArena catalog at once.

It should start with a **regression-first shortlist** chosen for:

- housing / price prediction relevance
- geographically meaningful structure
- enough scale to stress retrieval methods
- diversity beyond geography so we can tell if gains are domain-specific

### Tier 1
- `miami_housing`
- `houses`
- `diamonds`

### Tier 2
- `healthcare_insurance_expenses`
- `concrete_compressive_strength`
- `airfoil_self_noise`
- `wine_quality`

### Tier 3
- `Food_Delivery_Time`
- `physiochemical_protein`
- `superconductivity`

## Benchmark Rules

### Split Policy
- use matched split sweeps
- default split seeds: `42, 43, 44, 45, 46`
- keep split seed separate from training seed

### Baseline Set
Every new challenger should be judged against:

- `TabR`
- `TabPFN_full`
- `TabM`
- current internal champion for the same protocol

### Reporting Rules
Every benchmark row should include:

- dataset name
- split seed
- model name
- primary metric
- secondary metrics
- runtime if available

Every summary should include:

- mean
- std
- paired gain vs baseline
- win count vs baseline

## H200 Handoff

The H200 environment should receive:

1. this manifest
2. the render/validation scripts
3. the exact benchmark branch
4. then the actual heavy baseline/model runs

That keeps the GPU stage focused on compute, not on deciding protocol.
