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


def normalize_non_anchor_weights(
    specialist_weights: torch.Tensor,
    *,
    full_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out anchor mass and renormalize attention over non-anchor experts only."""
    if specialist_weights.ndim != 2:
        raise ValueError(f"Expected 2D specialist weights, got shape {tuple(specialist_weights.shape)}")
    non_anchor = torch.ones(
        specialist_weights.shape[1],
        dtype=torch.bool,
        device=specialist_weights.device,
    )
    non_anchor[full_index] = False
    masked_weights = specialist_weights * non_anchor.unsqueeze(0).to(specialist_weights.dtype)
    specialist_mass = masked_weights.sum(dim=1, keepdim=True)
    safe_mass = torch.where(specialist_mass > 0, specialist_mass, torch.ones_like(specialist_mass))
    normalized_weights = torch.where(
        specialist_mass > 0,
        masked_weights / safe_mass,
        torch.zeros_like(masked_weights),
    )
    return normalized_weights, specialist_mass


def blend_predictions_torch(
    *,
    expert_predictions: torch.Tensor | np.ndarray,
    specialist_weights: torch.Tensor,
    defer_prob: torch.Tensor,
    full_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend anchor and non-anchor specialists with a clean defer path."""
    pred_tensor = torch.as_tensor(expert_predictions, dtype=torch.float32)
    if pred_tensor.ndim == 2:
        pred_tensor = pred_tensor.unsqueeze(-1)
    if pred_tensor.ndim != 3:
        raise ValueError(f"Expected predictions with 2 or 3 dims, got shape {tuple(pred_tensor.shape)}")

    device = specialist_weights.device
    pred_tensor = pred_tensor.to(device)
    defer_prob = defer_prob.to(device)

    if specialist_weights.shape != pred_tensor.shape[:2]:
        raise ValueError(
            f"Expected specialist_weights shape {tuple(pred_tensor.shape[:2])}, got {tuple(specialist_weights.shape)}"
        )

    full_pred = pred_tensor[:, full_index : full_index + 1, :]
    normalized_weights, specialist_mass = normalize_non_anchor_weights(
        specialist_weights,
        full_index=full_index,
    )
    specialist_pred = (normalized_weights.unsqueeze(-1) * pred_tensor).sum(dim=1, keepdim=True)
    effective_defer = torch.where(specialist_mass > 0, defer_prob, torch.zeros_like(defer_prob))
    blended = (1.0 - effective_defer.unsqueeze(-1)) * full_pred + effective_defer.unsqueeze(-1) * specialist_pred
    return blended, normalized_weights, effective_defer, specialist_mass


def integrate_predictions(
    *,
    expert_predictions: np.ndarray,
    router_outputs: RouterOutputs,
) -> IntegrationOutputs:
    weights = router_outputs.specialist_weights
    blended, normalized_weights, effective_defer, specialist_mass = blend_predictions_torch(
        expert_predictions=expert_predictions,
        specialist_weights=weights,
        defer_prob=router_outputs.defer_prob,
        full_index=router_outputs.full_index,
    )

    diagnostics = {
        "router_kind": router_outputs.router_kind,
        "full_index": int(router_outputs.full_index),
        "mean_defer_prob": float(router_outputs.defer_prob.mean().item()),
        "effective_defer_rate": float((effective_defer > 0).float().mean().item()),
        "mean_specialist_mass": float(specialist_mass.mean().item()),
        "mean_anchor_attention_weight": float(weights[:, router_outputs.full_index].mean().item()),
    }

    result = blended.squeeze(1)
    if result.shape[1] == 1:
        result = result.squeeze(1)

    return IntegrationOutputs(
        predictions=result.detach().cpu().numpy().astype(np.float32),
        normalized_weights=normalized_weights.detach().cpu().numpy().astype(np.float32),
        defer_prob=router_outputs.defer_prob.detach().cpu().numpy().astype(np.float32),
        diagnostics=diagnostics,
    )
