# GraphDrone

**GraphDrone** is a portfolio architecture for tabular prediction.

It builds a system of **anchor and specialist experts** across multiple feature views, then integrates their predictions with task-aware geometric or learned policies.

GraphDrone is **not** a thin wrapper around TabPFN.

In the current open-source release, TabPFN is the **built-in foundation expert family** used by many GraphDrone portfolios. The GraphDrone contribution is the higher-level system:
- portfolio construction
- anchor vs specialist role assignment
- multi-view subspace design
- geometric or learned integration
- diagnostics, caching, and version-aware benchmarking

If you need one sentence for positioning:

> GraphDrone is a portfolio intelligence layer for tabular prediction: it turns one or more strong base experts into a coordinated anchor-and-specialist system.

## What GraphDrone Is

GraphDrone is a framework for:
- defining a portfolio of experts
- assigning a stable **anchor** expert
- creating multiple specialist views of the same dataset
- integrating predictions with **GeoPOE** or a learned router
- benchmarking those systems under explicit versioned contracts

The product is the **system design**, not a renamed checkpoint.

## What GraphDrone Is Not

GraphDrone is not:
- a renamed TabPFN model
- a single-model inference wrapper
- only a router head attached to one predictor

Today, the default public expert factory builds TabPFN-based specialists for the foundation path. That does **not** make GraphDrone “just TabPFN”. GraphDrone decides:
- how many experts exist
- which feature subsets they see
- which expert is the anchor
- how experts are integrated
- which benchmark version corresponds to which behavior

That system-level layer is the core of the project.

## Current Architecture

The current codebase uses **different default strategies for classification and regression**.

### Classification

The default classification path builds:
- `FULL` anchor expert
- `3 x SUB` specialists over randomized subspaces
- static anchor-boosted **GeoPOE** integration by default

This path avoids learned-router overfitting on small out-of-fold splits and returns valid class probabilities directly.

### Regression

The benchmarked regression path is versioned in the research runners. Depending on the version, it may use:
- `FULL + SUB` specialist portfolios
- GORA geometric observers
- static or learned integration

For regression, treat the benchmark scripts and version ledger as the source of truth rather than assuming the minimal API is identical to a published result.

## Benchmark Snapshot

### Internal benchmark (`v1.19.0`, 2026-03-19)

Evaluated on `6 datasets x 3 folds` against TabPFN v2.5 default.

| Engine | GraphDrone ELO | TabPFN ELO | Result |
|--------|----------------|------------|--------|
| Regression | `1523.2` | `1476.8` | GraphDrone `+46.4` |
| Classification | `1502.2` | `1497.8` | GraphDrone `+4.4` |

Classification per-dataset breakdown:

| Dataset | GraphDrone F1 | TabPFN F1 | Result |
|---------|---------------|-----------|--------|
| diabetes | `0.755` | `0.732` | GraphDrone wins |
| credit_g | `0.679` | `0.694` | TabPFN wins |
| segment | `0.947` | `0.947` | tie |
| mfeat_factors | `0.986` | `0.983` | GraphDrone wins |
| pendigits | `0.995` | `0.996` | near tie |
| optdigits | `0.993` | `0.992` | GraphDrone wins |

### TabArena context

GraphDrone has historical full-suite TabArena results, but those numbers are **version-sensitive** and should not be mixed with newer internal benchmark results without re-running the full contract.

Historical reference:

| Version | ELO | Rank | Notes |
|---------|-----|------|-------|
| `v1.0.0-gora` | `1420.3` | `~19` | initial release |
| `v1-width.1` | `1441.1` | `18.8` | cross-attn fix + BCE loss |
| `v1-width.2` | `1458.9` | `#10` | vectorized GORA observers |

For current benchmark contracts, use the scripts and manifests rather than projected rank claims.

## How To Talk About GraphDrone

Recommended language:
- `GraphDrone is a portfolio architecture for tabular specialists.`
- `GraphDrone uses foundation experts inside a structured anchor-and-specialist system.`
- `GraphDrone improves tabular prediction through multi-view specialization and controlled integration.`
- `In the current OSS release, TabPFN is the built-in foundation expert family for GraphDrone portfolios.`

Avoid:
- `GraphDrone wraps TabPFN`
- `GraphDrone is just a TabPFN router`
- `GraphDrone is the same as TabPFN with a shell on top`

Those descriptions undersell the architecture and confuse what is actually being benchmarked.

## Installation

```bash
git clone https://github.com/RicardoLaMo/Graph_Drone.git
cd Graph_Drone
pip install -e .
```

Requirements:
- Python `>=3.10`
- PyTorch `>=2.0`
- TabPFN `>=2.0`
- scikit-learn `>=1.3`

## Quick Start

```python
import numpy as np
from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig

config = GraphDroneConfig(
    full_expert_id="FULL",
    router=SetRouterConfig(kind="bootstrap_full_only"),
)

model = GraphDrone(config)
model.fit(X_train, y_train)  # auto-detects classification vs regression
predictions = model.predict(X_test)

result = model.predict(X_test, return_diagnostics=True)
print(result.diagnostics)
```

### Regression example

```python
from graphdrone_fit import GraphDrone, GraphDroneConfig, SetRouterConfig

config = GraphDroneConfig(
    full_expert_id="FULL",
    router=SetRouterConfig(kind="contextual_transformer"),
)

model = GraphDrone(config)
model.fit(X_train, y_train, problem_type="regression")
y_pred = model.predict(X_test)
```

### Custom portfolio example

```python
from graphdrone_fit import (
    GraphDrone,
    GraphDroneConfig,
    SetRouterConfig,
    ExpertBuildSpec,
    ViewDescriptor,
    IdentitySelectorAdapter,
)

n_features = X_train.shape[1]
full_idx = tuple(range(n_features))
sub_idx = tuple(range(n_features // 2))

specs = (
    ExpertBuildSpec(
        descriptor=ViewDescriptor(
            expert_id="FULL",
            family="FULL",
            view_name="Foundation Full",
            is_anchor=True,
            input_dim=n_features,
            input_indices=full_idx,
        ),
        model_kind="foundation_classifier",
        input_adapter=IdentitySelectorAdapter(indices=full_idx),
        model_params={"n_estimators": 8, "device": "cuda"},
    ),
    ExpertBuildSpec(
        descriptor=ViewDescriptor(
            expert_id="SUB0",
            family="structural_subspace",
            view_name="Subspace View",
            input_dim=len(sub_idx),
            input_indices=sub_idx,
        ),
        model_kind="foundation_classifier",
        input_adapter=IdentitySelectorAdapter(indices=sub_idx),
        model_params={"n_estimators": 8, "device": "cuda"},
    ),
)

model = GraphDrone(
    GraphDroneConfig(
        full_expert_id="FULL",
        router=SetRouterConfig(kind="bootstrap_full_only"),
    )
)
model.fit(X_train, y_train, expert_specs=specs, problem_type="classification")
```

## Reproducing Results

```bash
# Regression benchmark
PYTHONPATH=src python scripts/run_geopoe_benchmark.py --tasks regression --folds 0 1 2

# Classification benchmark
PYTHONPATH=src python scripts/run_smart_benchmark.py --folds 0 1 2

# Quick smoke test
PYTHONPATH=src python scripts/run_smart_benchmark.py --quick --folds 0
```

Results are written under `eval/`.

## Benchmark Discipline

GraphDrone results are **version-sensitive**.

Do not cite a single number without also naming:
- version tag
- task family
- dataset cohort
- fold protocol
- benchmark script

Use these files as the source of truth:
- [`VERSIONS.md`](VERSIONS.md)
- [`scripts/run_geopoe_benchmark.py`](scripts/run_geopoe_benchmark.py)
- [`scripts/run_smart_benchmark.py`](scripts/run_smart_benchmark.py)

This avoids the common failure mode of comparing a new implementation against stale cached results or incompatible benchmark signatures.

## Research Log

Experiment history lives in [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

See [`VERSIONS.md`](VERSIONS.md) for version history and architecture notes.

## Hardware

The project has been developed and benchmarked on NVIDIA H200 NVL. GPU is strongly preferred for fast foundation-model specialist inference, but the core package can fall back to CPU when needed.

## Short Marketing Copy

**Short**

> GraphDrone is a portfolio architecture for tabular prediction that coordinates anchor and specialist experts across multiple feature views.

**Medium**

> GraphDrone is a system for tabular prediction that builds a portfolio of specialists, assigns a stable anchor, and integrates their outputs with geometric or learned policies. In the current OSS release, TabPFN is the built-in foundation expert family inside that portfolio, but GraphDrone is the portfolio architecture, not a wrapper.

**Direct comparison framing**

> TabPFN is a base expert family used inside GraphDrone. GraphDrone is the higher-level system that constructs and combines specialists.
