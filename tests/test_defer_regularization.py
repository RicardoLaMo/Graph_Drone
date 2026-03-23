"""
Unit tests for defer regularization (Phase 1 / V1.3).

Tests that the defer_penalty_lambda / defer_target config fields validate
correctly and that the penalty term drives gradients toward the target.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from graphdrone_fit.config import SetRouterConfig


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

def test_defer_penalty_lambda_default_zero():
    cfg = SetRouterConfig(kind="noise_gate_router")
    validated = cfg.validate()
    assert validated.defer_penalty_lambda == 0.0


def test_defer_target_default():
    cfg = SetRouterConfig(kind="noise_gate_router")
    validated = cfg.validate()
    assert validated.defer_target == 0.8


def test_defer_penalty_lambda_positive():
    cfg = SetRouterConfig(kind="noise_gate_router", defer_penalty_lambda=0.5)
    validated = cfg.validate()
    assert validated.defer_penalty_lambda == 0.5


def test_defer_target_range():
    cfg = SetRouterConfig(kind="noise_gate_router", defer_penalty_lambda=0.5, defer_target=0.6)
    validated = cfg.validate()
    assert validated.defer_target == 0.6


def test_defer_penalty_lambda_negative_raises():
    with pytest.raises(ValueError, match="defer_penalty_lambda must be non-negative"):
        SetRouterConfig(kind="noise_gate_router", defer_penalty_lambda=-0.1).validate()


def test_defer_target_above_one_raises():
    with pytest.raises(ValueError, match="defer_target must be in"):
        SetRouterConfig(kind="noise_gate_router", defer_target=1.5).validate()


def test_defer_target_below_zero_raises():
    with pytest.raises(ValueError, match="defer_target must be in"):
        SetRouterConfig(kind="noise_gate_router", defer_target=-0.1).validate()


# ---------------------------------------------------------------------------
# Gradient test: penalty drives defer_prob toward target
# ---------------------------------------------------------------------------

def test_defer_penalty_gradient_toward_target():
    """
    Simulate the penalty term from _fit_classification_router.
    A defer_prob saturated at ~0.999 should receive a gradient pushing it
    toward defer_target=0.8 when defer_penalty_lambda > 0.
    """
    defer_penalty_lambda = 0.5
    defer_target = 0.8

    # Saturated defer_prob (simulating pathological router output)
    raw_logit = torch.tensor([5.0], requires_grad=True)  # sigmoid → ~0.993
    defer_prob = torch.sigmoid(raw_logit)

    mean_defer = defer_prob.mean()
    penalty = defer_penalty_lambda * (mean_defer - defer_target) ** 2
    penalty.backward()

    # Gradient on raw_logit should be positive → push sigmoid output down toward 0.8
    assert raw_logit.grad is not None
    assert raw_logit.grad.item() > 0, (
        f"Expected positive gradient (pushing defer toward {defer_target}), "
        f"got {raw_logit.grad.item()}"
    )


def test_defer_penalty_zero_when_lambda_zero():
    """
    When defer_penalty_lambda=0, the penalty term should contribute no gradient.
    """
    defer_penalty_lambda = 0.0
    defer_target = 0.8

    raw_logit = torch.tensor([5.0], requires_grad=True)
    defer_prob = torch.sigmoid(raw_logit)
    mean_defer = defer_prob.mean()

    # Replicate conditional from model.py: only add penalty when lambda > 0
    dummy_loss = mean_defer * 0.0  # zero base loss for isolation
    if defer_penalty_lambda > 0:
        dummy_loss = dummy_loss + defer_penalty_lambda * (mean_defer - defer_target) ** 2
    dummy_loss.backward()

    # No penalty added → gradient should be 0
    assert raw_logit.grad is not None
    assert raw_logit.grad.item() == pytest.approx(0.0, abs=1e-9)


def test_defer_penalty_at_target_is_zero():
    """
    When defer_prob is already at defer_target, the penalty should be zero.
    """
    defer_penalty_lambda = 0.5
    defer_target = 0.8

    # defer_prob exactly at target
    defer_prob = torch.tensor([defer_target], requires_grad=False)
    mean_defer = defer_prob.mean()
    penalty = defer_penalty_lambda * (mean_defer - defer_target) ** 2

    assert penalty.item() == pytest.approx(0.0, abs=1e-9)
