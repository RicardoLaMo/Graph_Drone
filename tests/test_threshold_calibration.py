"""
Unit tests for OOF threshold calibration (Phase 3B).

Tests _compute_oof_threshold logic and that binary_threshold_ is set
correctly by the preset when calibrate_threshold is enabled/disabled.
"""
from __future__ import annotations

import numpy as np
import pytest

from graphdrone_fit.model import GraphDrone
from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig


# ---------------------------------------------------------------------------
# _compute_oof_threshold
# ---------------------------------------------------------------------------

def test_threshold_improves_on_imbalanced():
    """Optimal threshold on an imbalanced dataset should not be 0.5."""
    rng = np.random.RandomState(0)
    n = 400
    y = np.concatenate([np.zeros(300), np.ones(100)])  # 3:1 imbalance
    # Noisy but informative classifier: P(pos) ∝ 0.6 if positive, 0.3 otherwise
    proba_pos = np.where(y == 1, rng.uniform(0.4, 0.8, n), rng.uniform(0.1, 0.5, n))
    t = GraphDrone._compute_oof_threshold(y, proba_pos)
    assert 0.0 < t < 1.0
    # For imbalanced data the optimal threshold is expected to be < 0.5
    assert t < 0.5 + 0.15, f"expected threshold close to or below 0.5, got {t:.3f}"


def test_threshold_balanced_near_half():
    """For a perfectly calibrated balanced dataset the threshold should be near 0.5."""
    rng = np.random.RandomState(1)
    n = 400
    y = np.concatenate([np.zeros(200), np.ones(200)])
    proba_pos = np.where(y == 1, rng.uniform(0.55, 0.95, n), rng.uniform(0.05, 0.45, n))
    t = GraphDrone._compute_oof_threshold(y, proba_pos)
    assert 0.3 <= t <= 0.7, f"expected threshold near 0.5 for balanced data, got {t:.3f}"


def test_threshold_falls_back_on_degenerate():
    """All-zero predictions → no threshold improves on baseline → fall back to 0.5."""
    y = np.zeros(50, dtype=int)
    proba_pos = np.zeros(50)
    t = GraphDrone._compute_oof_threshold(y, proba_pos)
    assert t == 0.5

def test_threshold_falls_back_on_marginal_improvement():
    """If OOF improvement < min_improvement, return 0.5 to guard against OOF noise."""
    rng = np.random.RandomState(3)
    n = 200
    y = np.concatenate([np.zeros(100), np.ones(100)])
    # Nearly calibrated: improvement over 0.5 is tiny
    proba_pos = np.where(y == 1, rng.uniform(0.48, 0.55, n), rng.uniform(0.45, 0.52, n))
    t = GraphDrone._compute_oof_threshold(y, proba_pos, min_improvement=0.05)
    assert t == 0.5, f"expected fallback to 0.5 on marginal signal, got {t:.3f}"


def test_threshold_falls_back_on_small_shift():
    """If optimal threshold is too close to 0.5, return 0.5 (guards small OOF splits)."""
    rng = np.random.RandomState(4)
    n = 200
    y = np.concatenate([np.zeros(120), np.ones(80)])  # mild imbalance
    # Best threshold ≈ 0.47 (small shift < min_shift=0.08)
    proba_pos = np.where(y == 1, rng.uniform(0.45, 0.65, n), rng.uniform(0.30, 0.55, n))
    t = GraphDrone._compute_oof_threshold(y, proba_pos, min_improvement=0.005, min_shift=0.15)
    assert t == 0.5, f"expected fallback when shift is too small, got {t:.3f}"


def test_threshold_improves_f1():
    """The returned threshold should achieve higher F1 than 0.5 on the OOF set."""
    from sklearn.metrics import f1_score
    rng = np.random.RandomState(2)
    n = 300
    y = np.concatenate([np.zeros(240), np.ones(60)])  # 4:1 imbalance
    proba_pos = np.where(y == 1, rng.uniform(0.35, 0.75, n), rng.uniform(0.05, 0.45, n))

    t_opt = GraphDrone._compute_oof_threshold(y, proba_pos)
    f1_opt = f1_score(y, (proba_pos >= t_opt).astype(int), average="macro", zero_division=0)
    f1_half = f1_score(y, (proba_pos >= 0.5).astype(int), average="macro", zero_division=0)
    assert f1_opt >= f1_half - 1e-6, (
        f"calibrated threshold F1={f1_opt:.4f} should be >= default F1={f1_half:.4f}"
    )


# ---------------------------------------------------------------------------
# binary_threshold_ attribute on GraphDrone
# ---------------------------------------------------------------------------

def test_binary_threshold_default_is_half():
    """Without calibrate_threshold, binary_threshold_ should default to 0.5."""
    cfg = GraphDroneConfig(
        n_classes=1,
        router=SetRouterConfig(calibrate_threshold=False),
    )
    gd = GraphDrone(cfg)
    assert gd.binary_threshold_ == 0.5


def test_calibrate_threshold_config_flag():
    """calibrate_threshold=True should be accepted without validation error."""
    cfg = GraphDroneConfig(
        n_classes=1,
        router=SetRouterConfig(calibrate_threshold=True),
    )
    validated = cfg.validate()
    assert validated.router.calibrate_threshold is True


def test_calibrate_threshold_false_config_flag():
    """calibrate_threshold=False should be accepted without validation error."""
    cfg = GraphDroneConfig(
        n_classes=1,
        router=SetRouterConfig(calibrate_threshold=False),
    )
    validated = cfg.validate()
    assert validated.router.calibrate_threshold is False
