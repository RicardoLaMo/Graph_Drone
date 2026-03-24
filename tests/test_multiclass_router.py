"""
Unit tests for Phase MC-1: learned router for multiclass classification.

Tests that:
- Multiclass router trains when use_learned_router_for_classification=True
- Guards work: single-expert (FULL-only) falls back to static GeoPOE
- Guards work: tiny dataset (< ~600 rows) falls back due to OOF < 150
- binary_threshold_ stays at 0.5 for multiclass (never touches [:, 1])
- _clf_uses_learned_router is True/False as expected
- Binary path is completely unaffected by use_learned_router_for_classification
"""
from __future__ import annotations

import numpy as np
import pytest

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig


def _make_multiclass_data(n: int = 700, n_features: int = 16, n_classes: int = 3, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y = (rng.rand(n) * n_classes).astype(int)
    return X, y


def _make_binary_data(n: int = 700, n_features: int = 8, seed: int = 0):
    # n=700 ensures OOF split >= 150 rows (700*0.25=175) and skip_subs=False (n>=500)
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y = (rng.rand(n) > 0.5).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Learned router enabled for multiclass
# ---------------------------------------------------------------------------

def test_multiclass_router_trains_with_use_learned_true():
    """When use_learned_router_for_classification=True, multiclass should use learned router."""
    # n=700 → OOF = 700×0.25 = 175 rows > 150 minimum
    X, y = _make_multiclass_data(n=700, n_features=16, n_classes=3)
    cfg = GraphDroneConfig(
        n_classes=3,
        router=SetRouterConfig(kind="noise_gate_router"),
        use_learned_router_for_classification=True,
    )
    gd = GraphDrone(cfg)
    gd.fit(X, y, problem_type="classification")
    assert gd._clf_uses_learned_router is True, "Learned router should be active for multiclass"


def test_multiclass_predictions_shape_with_learned_router():
    """Learned router for multiclass must return [N, C] probability matrix."""
    X, y = _make_multiclass_data(n=500, n_features=16, n_classes=3)
    X_test = np.random.randn(50, 16).astype(np.float32)
    cfg = GraphDroneConfig(
        n_classes=3,
        router=SetRouterConfig(kind="noise_gate_router"),
        use_learned_router_for_classification=True,
    )
    gd = GraphDrone(cfg)
    gd.fit(X, y, problem_type="classification")
    proba = gd.predict(X_test)
    assert proba.shape == (50, 3), f"Expected (50, 3), got {proba.shape}"
    # Probabilities must sum to 1 per row
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(50), atol=1e-5)


def test_multiclass_router_disabled_by_default():
    """Default GraphDroneConfig has use_learned_router_for_classification=True — but
    bootstrap_full_only combined with a single-expert multiclass portfolio should fall back.
    Test separately that the flag=False forces static GeoPOE."""
    X, y = _make_multiclass_data(n=500, n_features=16, n_classes=3)
    cfg = GraphDroneConfig(
        n_classes=3,
        use_learned_router_for_classification=False,
    )
    gd = GraphDrone(cfg)
    gd.fit(X, y, problem_type="classification")
    assert gd._clf_uses_learned_router is False, "Router should be disabled when flag=False"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_single_expert_falls_back_to_static():
    """FULL-only portfolio (<=10 features) must fall back to static GeoPOE."""
    # 10 features → _build_default_specs returns FULL only for multiclass (<3 experts)
    X, y = _make_multiclass_data(n=500, n_features=10, n_classes=3)
    cfg = GraphDroneConfig(
        n_classes=3,
        router=SetRouterConfig(kind="noise_gate_router"),
        use_learned_router_for_classification=True,
    )
    gd = GraphDrone(cfg)
    gd.fit(X, y, problem_type="classification")
    assert gd._clf_uses_learned_router is False, (
        "1-expert (FULL-only) portfolio must fall back to static GeoPOE"
    )


def test_two_expert_multiclass_falls_back_to_static():
    """2-expert multiclass (11–14 features: FULL + 1 SUB) must also fall back.
    Empirically, 2-expert routing is unstable (SDSS17 regression: defer to weak SUB hurts F1)."""
    # 12 features → FULL + 1×SUB@60% = 2 experts → requires >=3 for learned routing
    X, y = _make_multiclass_data(n=700, n_features=12, n_classes=3)
    cfg = GraphDroneConfig(
        n_classes=3,
        router=SetRouterConfig(kind="noise_gate_router"),
        use_learned_router_for_classification=True,
    )
    gd = GraphDrone(cfg)
    gd.fit(X, y, problem_type="classification")
    assert gd._clf_uses_learned_router is False, (
        "2-expert multiclass must fall back to static GeoPOE (requires 3+ experts)"
    )


def test_tiny_dataset_falls_back_to_static():
    """Dataset where OOF split < 150 rows must fall back to static GeoPOE.
    n=500 → 500×0.25=125 OOF rows < 150 → falls back."""
    X, y = _make_multiclass_data(n=500, n_features=16, n_classes=3)
    cfg = GraphDroneConfig(
        n_classes=3,
        router=SetRouterConfig(kind="noise_gate_router"),
        use_learned_router_for_classification=True,
    )
    gd = GraphDrone(cfg)
    gd.fit(X, y, problem_type="classification")
    # 500 × 0.25 = 125 OOF rows < 150 → should fall back
    assert gd._clf_uses_learned_router is False, (
        "Dataset with OOF < 150 rows must fall back to static GeoPOE"
    )


# ---------------------------------------------------------------------------
# Binary path isolation
# ---------------------------------------------------------------------------

def test_binary_path_unaffected():
    """Binary path must be unchanged regardless of use_learned_router_for_classification."""
    # n=700: OOF=175>150; n>=500 avoids skip_subs; binary always uses learned regardless of flag
    X, y = _make_binary_data(n=700, n_features=8)
    # Explicitly set use_learned_router_for_classification=False — binary must still use learned router
    cfg = GraphDroneConfig(
        n_classes=2,
        router=SetRouterConfig(kind="noise_gate_router"),
        use_learned_router_for_classification=False,  # flag=False, but binary always uses learned
    )
    gd = GraphDrone(cfg)
    gd.fit(X, y, problem_type="classification")
    assert gd._clf_uses_learned_router is True, (
        "Binary always uses learned router regardless of use_learned_router_for_classification"
    )


def test_binary_threshold_unchanged_after_multiclass_fit():
    """After a multiclass fit, binary_threshold_ must remain at 0.5 (no [:, 1] indexing)."""
    X, y = _make_multiclass_data(n=500, n_features=16, n_classes=3)
    cfg = GraphDroneConfig(
        n_classes=3,
        router=SetRouterConfig(kind="noise_gate_router", calibrate_threshold=True),
        use_learned_router_for_classification=True,
    )
    gd = GraphDrone(cfg)
    gd.fit(X, y, problem_type="classification")
    assert gd.binary_threshold_ == 0.5, (
        f"binary_threshold_ must stay at 0.5 for multiclass, got {gd.binary_threshold_}"
    )


def test_multiclass_bootstrap_full_only_auto_upgrade():
    """When kind=bootstrap_full_only and use_learned=True, router auto-upgrades to noise_gate_router."""
    X, y = _make_multiclass_data(n=700, n_features=16, n_classes=3)
    cfg = GraphDroneConfig(
        n_classes=3,
        router=SetRouterConfig(kind="bootstrap_full_only"),
        use_learned_router_for_classification=True,
    )
    gd = GraphDrone(cfg)
    gd.fit(X, y, problem_type="classification")
    # Should have auto-upgraded and used learned router
    assert gd._clf_uses_learned_router is True
