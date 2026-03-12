from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .set_router import RouterOutputs


@dataclass(frozen=True)
class IntegrationOutputs:
    predictions: np.ndarray
    normalized_weights: np.ndarray
    defer_prob: np.ndarray
    diagnostics: dict[str, object]


def integrate_predictions(
    *,
    expert_predictions: np.ndarray,
    router_outputs: RouterOutputs,
) -> IntegrationOutputs:
    pred_tensor = torch.as_tensor(expert_predictions, dtype=torch.float32)
    weights = router_outputs.specialist_weights
    
    # Ensure all on same device
    device = weights.device
    pred_tensor = pred_tensor.to(device)
    defer_prob = router_outputs.defer_prob.to(device)
    
    if weights.shape != pred_tensor.shape:
        raise ValueError(
            f"Expected specialist_weights shape {tuple(pred_tensor.shape)}, got {tuple(weights.shape)}"
        )
    if defer_prob.shape != (pred_tensor.shape[0], 1):
        raise ValueError(
            f"Expected defer_prob shape {(pred_tensor.shape[0], 1)}, got {tuple(defer_prob.shape)}"
        )

    full_pred = pred_tensor[:, router_outputs.full_index : router_outputs.full_index + 1]
    weight_mass = weights.sum(dim=1, keepdim=True)
    safe_mass = torch.where(weight_mass > 0, weight_mass, torch.ones_like(weight_mass))
    normalized_weights = torch.where(weight_mass > 0, weights / safe_mass, torch.zeros_like(weights))
    specialist_pred = (normalized_weights * pred_tensor).sum(dim=1, keepdim=True)
    effective_defer = torch.where(weight_mass > 0, defer_prob, torch.zeros_like(defer_prob))
    blended = (1.0 - effective_defer) * full_pred + effective_defer * specialist_pred

    diagnostics = {
        "router_kind": router_outputs.router_kind,
        "full_index": int(router_outputs.full_index),
        "mean_defer_prob": float(defer_prob.mean().item()),
        "effective_defer_rate": float((effective_defer > 0).float().mean().item()),
        "mean_specialist_mass": float(weight_mass.mean().item()),
    }
    return IntegrationOutputs(
        predictions=blended.squeeze(1).detach().cpu().numpy().astype(np.float32),
        normalized_weights=normalized_weights.detach().cpu().numpy().astype(np.float32),
        defer_prob=defer_prob.detach().cpu().numpy().astype(np.float32),
        diagnostics=diagnostics,
    )
