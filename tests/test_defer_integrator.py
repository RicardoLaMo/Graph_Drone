from __future__ import annotations

import numpy as np
import torch

from graphdrone_fit.defer_integrator import integrate_predictions
from graphdrone_fit.set_router import RouterOutputs


def test_integrate_predictions_excludes_anchor_from_specialist_blend():
    expert_predictions = np.array([[10.0, 1.0, 3.0]], dtype=np.float32)
    router_outputs = RouterOutputs(
        specialist_weights=torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float32),
        defer_prob=torch.tensor([[1.0]], dtype=torch.float32),
        full_index=0,
        router_kind="contextual_transformer_router",
    )

    out = integrate_predictions(expert_predictions=expert_predictions, router_outputs=router_outputs)

    assert np.allclose(out.normalized_weights, np.array([[0.0, 0.5, 0.5]], dtype=np.float32))
    assert np.allclose(out.predictions, np.array([2.0], dtype=np.float32))
    assert np.isclose(out.diagnostics["mean_specialist_mass"], 0.2)
    assert np.isclose(out.diagnostics["mean_anchor_attention_weight"], 0.8)


def test_integrate_predictions_disables_defer_when_only_anchor_has_mass():
    expert_predictions = np.array([[7.0, 2.0, 4.0]], dtype=np.float32)
    router_outputs = RouterOutputs(
        specialist_weights=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        defer_prob=torch.tensor([[1.0]], dtype=torch.float32),
        full_index=0,
        router_kind="contextual_transformer_router",
    )

    out = integrate_predictions(expert_predictions=expert_predictions, router_outputs=router_outputs)

    assert np.allclose(out.normalized_weights, np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert np.allclose(out.predictions, np.array([7.0], dtype=np.float32))
    assert np.isclose(out.diagnostics["mean_specialist_mass"], 0.0)
