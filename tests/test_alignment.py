from __future__ import annotations

import numpy as np
import torch

from graphdrone_fit.alignment import OTNoiseGate, RotorAlignment, sinkhorn_log
from graphdrone_fit.config import LegitimacyGateConfig
from graphdrone_fit.set_router import LegitimacyGate


def test_rotor_alignment_identity_init_is_noop():
    rotor = RotorAlignment(dim=5)
    x = torch.randn(7, 5)
    rotated = rotor(x)
    assert torch.allclose(rotated, x, atol=1e-6)


def test_rotor_alignment_can_learn_known_rotation():
    torch.manual_seed(0)
    rotor = RotorAlignment(dim=3)
    theta = 0.4
    true_rot = torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    anchor = torch.randn(64, 3)
    specialist = anchor @ true_rot.T

    optimizer = torch.optim.Adam(rotor.parameters(), lr=0.1)
    initial_loss = rotor.alignment_loss(specialist, anchor).item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = rotor.alignment_loss(specialist, anchor)
        loss.backward()
        optimizer.step()
    final_loss = rotor.alignment_loss(specialist, anchor).item()

    assert final_loss < initial_loss * 0.1
    rot = rotor.rotation_matrix()
    eye = torch.eye(3)
    assert torch.allclose(rot.T @ rot, eye, atol=1e-4)


def test_sinkhorn_uniform_cost_produces_uniform_plan():
    cost = torch.zeros(3, 3)
    plan = sinkhorn_log(cost, epsilon=0.1, max_iter=100)
    expected = torch.full((3, 3), 1.0 / 9.0)
    assert torch.allclose(plan, expected, atol=1e-4)
    assert torch.allclose(plan.sum(dim=0), torch.full((3,), 1.0 / 3.0), atol=1e-4)
    assert torch.allclose(plan.sum(dim=1), torch.full((3,), 1.0 / 3.0), atol=1e-4)


def test_ot_noise_gate_fits_prototypes_and_returns_expected_shapes():
    gate = OTNoiseGate(token_dim=4, prototype_count=3)
    tokens = torch.randn(12, 3, 4)
    gate.fit_prototypes(tokens, full_index=0)
    validity, ot_costs = gate(tokens, full_index=0)
    assert validity.shape == (12, 3, 1)
    assert ot_costs.shape == (12, 3)
    assert torch.allclose(validity[:, 0], torch.ones_like(validity[:, 0]), atol=1e-6)


def test_ot_noise_gate_downweights_far_specialists():
    gate = OTNoiseGate(token_dim=2, prototype_count=2, threshold=0.2, alpha=20.0)
    train_tokens = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.1]],
            [[0.1, 0.0], [0.1, 0.1]],
            [[-0.1, 0.0], [-0.1, 0.1]],
        ],
        dtype=torch.float32,
    )
    gate.fit_prototypes(train_tokens, full_index=0)

    near_tokens = torch.tensor([[[0.0, 0.0], [0.05, 0.05]]], dtype=torch.float32)
    far_tokens = torch.tensor([[[0.0, 0.0], [3.0, 3.0]]], dtype=torch.float32)
    near_validity, _ = gate(near_tokens, full_index=0)
    far_validity, _ = gate(far_tokens, full_index=0)

    assert near_validity[0, 1, 0] > far_validity[0, 1, 0]


def test_legitimacy_gate_bypasses_confident_rows_for_classification():
    gate = LegitimacyGate(LegitimacyGateConfig(enabled=True, classification_entropy_threshold=0.2))
    anchor_probs = np.array([[0.999, 0.001], [0.55, 0.45]], dtype=np.float32)
    expert_probs = np.stack([anchor_probs, anchor_probs], axis=1)
    decision = gate.evaluate(
        problem_type="classification",
        anchor_predictions=anchor_probs,
        expert_predictions=expert_probs,
        quality_scores=None,
    )
    assert decision.metric == "normalized_entropy"
    assert decision.exit_mask.tolist() == [True, False]


def test_legitimacy_gate_uses_variance_for_regression():
    gate = LegitimacyGate(LegitimacyGateConfig(enabled=True, regression_variance_threshold=0.01))
    expert_preds = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    decision = gate.evaluate(
        problem_type="regression",
        anchor_predictions=expert_preds[:, 0],
        expert_predictions=expert_preds,
        quality_scores=None,
    )
    assert decision.metric == "cross_expert_variance"
    assert decision.exit_mask.tolist() == [True, False]
