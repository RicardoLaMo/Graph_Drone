from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .set_router import RouterOutputs


@dataclass(frozen=True)
class IntegrationOutputs:
    predictions: np.ndarray
    probabilities: np.ndarray | None
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
    if pred_tensor.ndim == 2 and weights.shape != pred_tensor.shape:
        raise ValueError(
            f"Expected specialist_weights shape {tuple(pred_tensor.shape)}, got {tuple(weights.shape)}"
        )
    if pred_tensor.ndim == 3 and weights.shape != pred_tensor.shape[:2]:
        raise ValueError(
            f"Expected specialist_weights shape {tuple(pred_tensor.shape[:2])}, got {tuple(weights.shape)}"
        )
    if pred_tensor.ndim not in {2, 3}:
        raise ValueError(
            f"Expected expert_predictions rank 2 or 3, got {tuple(pred_tensor.shape)}"
        )
    if router_outputs.defer_prob.shape != (pred_tensor.shape[0], 1):
        raise ValueError(
            f"Expected defer_prob shape {(pred_tensor.shape[0], 1)}, got {tuple(router_outputs.defer_prob.shape)}"
        )

    weight_mass = weights.sum(dim=1, keepdim=True)
    safe_mass = torch.where(weight_mass > 0, weight_mass, torch.ones_like(weight_mass))
    normalized_weights = torch.where(weight_mass > 0, weights / safe_mass, torch.zeros_like(weights))
    effective_defer = torch.where(weight_mass > 0, router_outputs.defer_prob, torch.zeros_like(router_outputs.defer_prob))

    if pred_tensor.ndim == 2:
        full_pred = pred_tensor[:, router_outputs.full_index : router_outputs.full_index + 1]
        specialist_pred = (normalized_weights * pred_tensor).sum(dim=1, keepdim=True)
        blended = (1.0 - effective_defer) * full_pred + effective_defer * specialist_pred
        predictions = blended.squeeze(1).detach().cpu().numpy().astype(np.float32)
        probabilities = None
    else:
        full_pred = pred_tensor[:, router_outputs.full_index, :]
        specialist_pred = torch.einsum("ne,nec->nc", normalized_weights, pred_tensor)
        blended = (1.0 - effective_defer) * full_pred + effective_defer * specialist_pred
        blended = blended.clamp_min(1e-8)
        blended = blended / blended.sum(dim=1, keepdim=True).clamp_min(1e-8)
        probabilities = blended.detach().cpu().numpy().astype(np.float32)
        predictions = blended.argmax(dim=1).detach().cpu().numpy().astype(np.int64)

    diagnostics = {
        "router_kind": router_outputs.router_kind,
        "full_index": int(router_outputs.full_index),
        "mean_defer_prob": float(router_outputs.defer_prob.mean().item()),
        "effective_defer_rate": float((effective_defer > 0).float().mean().item()),
        "mean_specialist_mass": float(weight_mass.mean().item()),
    }
    if pred_tensor.ndim == 3:
        diagnostics["mean_prediction_entropy"] = float(
            (-(blended * blended.log()).sum(dim=1).mean().item())
        )
    return IntegrationOutputs(
        predictions=predictions,
        probabilities=probabilities,
        normalized_weights=normalized_weights.detach().cpu().numpy().astype(np.float32),
        defer_prob=router_outputs.defer_prob.detach().cpu().numpy().astype(np.float32),
        diagnostics=diagnostics,
    )
