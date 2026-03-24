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

## Current Architecture (V1.3)

The current codebase uses **different strategies for binary classification, multiclass classification, and regression**. All three engines dispatch through a single `GraphDrone` class via `_detect_problem_type(y)`.

### Binary Classification (V1.3)

The binary path (`n_classes == 2`) builds:
- Portfolio: `FULL` anchor + `3 × SUB` specialists (fracs 0.8/0.85/0.9); `1 × SUB` at 50% for n_features < 25
- Router: learned `noise_gate_router` with NLL loss + residual anchor penalty
- OOF threshold calibration: F1-maximizing threshold computed on OOF predictions (`calibrate_threshold=True`)
- Task-conditioned prior: cross-dataset task prior injected into anchor token (`task_prior_bank_dir`, `task_prior_strength`)

### Multiclass Classification (V1.3)

The multiclass path (`n_classes > 2`) uses a feature-count-dependent portfolio and routing strategy:
- ≤10 features → `FULL` only → static GeoPOE blend
- ≤14 features → `FULL + 1 × SUB` @ 60% → static GeoPOE blend (2-expert routing is unstable)
- >14 features → `FULL + 3 × SUB` @ 0.8/0.85/0.9 → learned `noise_gate_router` (requires ≥3 experts and ≥150 OOF rows)

The learned router path is gated by `use_learned_router_for_classification=True` (default) in `GraphDroneConfig`. When active:
- Quality tokens come from `foundation_classifier_bagged` (4× TabPFN bags, per-expert variance)
- Per-class OVR threshold calibration via `calibrate_multiclass_thresholds=True` (stored in `class_thresholds_`)
- Falls back to static GeoPOE if expert count or OOF row count guards are not met

### Regression (V1.3)

The regression path uses:
- Portfolio: `FULL` + `SUB0` (70%, seed 0) + `SUB1` (70%, seed 1) + `SUB2` (80%, seed 2)
- Router: `contextual_transformer` with MSE loss + `2.0 × relu(mse − anchor_mse)` residual anchor penalty
- GORA: active (kappa + LID per expert per subspace view)
- Task-conditioned prior infrastructure: `routing_bias` mode + expert-local gating (`task_prior_expert_local_gate_alpha`); research outcome is a hold at current settings — infrastructure is in place for future work

For regression, treat the benchmark scripts and version ledger as the source of truth.

## Benchmark Snapshot

### Internal benchmark (`v1.3.0`, 2026-03-23/24)

Evaluated on `9 datasets × 3 folds` against TabPFN v2.5 default. **Both engines win.**

| Engine | GraphDrone ELO | TabPFN ELO | Δ | Datasets |
|--------|----------------|------------|---|---------|
| Regression | `1523.2` | `1476.8` | `+46.4` | 6 regression datasets |
| Classification | `1512.4` | `1487.6` | `+24.8` | 9 clf datasets (2 binary + 7 multiclass) |

Classification per-dataset (V1.3.0, smart benchmark — 9 datasets × 3 folds):

| Dataset | Type | GraphDrone F1 | TabPFN F1 | Result |
|---------|------|---------------|-----------|--------|
| diabetes | binary | `0.7539` | `0.7320` | **GD wins F1 +0.022** |
| credit_g | binary | `0.7226` | `0.6937` | **GD wins F1 +0.029** |
| segment | 7-class | `0.9474` | `0.9474` | Tie F1, GD wins log_loss |
| mfeat_factors | 10-class | `0.9843` | `0.9826` | **GD wins both** |
| pendigits | 10-class | `0.9949` | `0.9959` | Near-saturation |
| optdigits | 10-class | `0.9927` | `0.9924` | **GD wins F1** |
| maternal_health_risk | 3-class | `0.8644` | `0.8609` | **GD wins F1** |
| website_phishing | 3-class | `0.9230` | `0.9239` | GD wins log_loss |
| SDSS17 | 3-class | `0.9674` | `0.9672` | **GD wins both** |

### TabArena context

GraphDrone has historical full-suite TabArena results, but those numbers are **version-sensitive** and should not be mixed with newer internal benchmark results without re-running the full contract.

Historical reference:

| Version | ELO | Rank | Notes |
|---------|-----|------|-------|
| `v1.0.0-gora` | `1420.3` | `~19` | initial release |
| `v1-width.1` | `1441.1` | `18.8` | cross-attn fix + BCE loss |
| `v1-width.2` | `1458.9` | `#10` | vectorized GORA observers |

For current benchmark contracts, use the scripts and manifests rather than projected rank claims.

## V1.3 Changelog

### Regression V1.3 — Task-Conditioned Prior Research

ELO unchanged at `1523.2`. Research branch `exp/v13-reg-afc-revisit` explored injecting cross-dataset task priors into the regression anchor token.

**Established findings:**
- `routing_bias` architecture outperforms additive `anchor_shift` for regression task priors
- Expert-local gating (`task_prior_expert_local_gate_alpha`) is the best current local/global shaping (+0.000200 mean RMSE improvement)
- Hard-regime routing stability cleared on california/diamonds/house_prices

**Falsified (3 forms of disagreement-derived opportunity):**
- Static dataset-level expert opportunity gate
- Raw row-level disagreement weighting
- Residual/thresholded row-level disagreement modulation

**Outcome:** Research record merged as a hold (not a promotion). Best known regression task-prior baseline: commit `1a70b29`. Infrastructure (`task_prior_mode`, `task_prior_expert_local_gate_alpha`, and related config fields) is in place for the next team to branch from this baseline with a different teacher family.

---

### Binary Classification V1.3 — Three-Phase Upgrade (ELO 1502.2 → 1512.4)

Three sequential phases, each champion/challenger gated:

**Phase 1 — Defer Regularization** (`v1_3_phase1` preset)
Added quadratic defer penalty `(mean_defer − defer_target)²` to prevent defer saturation on small OOF splits. Controlled via `defer_penalty_lambda` and `defer_target`.

**Phase 2 — Task-Conditioned Prior** (`v1_3_phase2` / `v1_3_phase3b` presets)
Cross-dataset task prior injected into the anchor token (`task_prior_bank_dir`, `task_prior_encoder_kind`, `task_prior_strength`, `task_prior_exact_reuse_blend`). Confidence-gated defer penalty on the prior path. credit_g F1 gap narrowed from −0.004 to −0.0014.

**Phase 3B — OOF Threshold Calibration** (`v1_3_phase3b` preset)
F1-maximizing threshold computed on OOF blend predictions (`calibrate_threshold=True`). credit_g threshold shifted to 0.61–0.68 (30% positive rate), closing the gap entirely. **credit_g: GD +0.029 F1 over TabPFN.**

---

### Multiclass Classification V1.3 — MC Pipeline (three phases, net Δ = −0.0001)

Brought the multiclass path to V1.3 infrastructure parity via three MC phases.

**Phase MC-1 — Learned Router Wired for Multiclass** (`v1_3_mc_phase1` preset)
`use_learned_router_for_classification=True` now correctly dispatches to `noise_gate_router` for multiclass. Guards:
- ≥3 experts required (2-expert routing regressed SDSS17 by −0.007 F1)
- ≥150 OOF rows required; else static GeoPOE

**Phase MC-2 — Bagged Quality Tokens** (`v1_3_mc_phase2` preset)
`foundation_classifier_bagged` (4× TabPFN bags) enabled for multiclass when learned router is active, providing per-expert prediction variance as router quality tokens. Result: null (bag variance near-zero on these datasets; router already well-trained on entropy tokens).

**Phase MC-3 — Per-Class OVR Threshold Calibration** (`v1_3_mc_phase3` preset)
`calibrate_multiclass_thresholds=True` computes per-class one-vs-rest F1-maximizing thresholds on OOF predictions. Stored in `class_thresholds_`; applied at label time as `proba / class_thresholds → argmax`. Sparse-class fallback: classes with <30 positive OOF samples default to 0.5.

**Net result:** Mean F1 delta vs static GeoPOE champion = −0.0001 across 7 multiclass datasets (within noise). Learned routing infrastructure is active and ready for MC-4 improvements (optdigits routing instability is the primary remaining opportunity).

---

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
