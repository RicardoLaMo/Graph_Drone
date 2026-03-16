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
    # expert_predictions: [N, E, C] or [N, E]
    pred_tensor = torch.as_tensor(expert_predictions, dtype=torch.float32)
    if pred_tensor.ndim == 2:
        pred_tensor = pred_tensor.unsqueeze(-1) # [N, E, 1]

    weights = router_outputs.specialist_weights # [N, E]
    
    # Ensure all on same device
    device = weights.device
    pred_tensor = pred_tensor.to(device)
    defer_prob = router_outputs.defer_prob.to(device) # [N, 1]
    
    if weights.shape != pred_tensor.shape[:2]:
        raise ValueError(
            f"Expected specialist_weights shape {tuple(pred_tensor.shape[:2])}, got {tuple(weights.shape)}"
        )

    # 1. Anchor (Full) predictions
    full_pred = pred_tensor[:, router_outputs.full_index : router_outputs.full_index + 1, :] # [N, 1, C]
    
    # 2. Specialist Ensemble
    weight_mass = weights.sum(dim=1, keepdim=True)
    safe_mass = torch.where(weight_mass > 0, weight_mass, torch.ones_like(weight_mass))
    normalized_weights = torch.where(weight_mass > 0, weights / safe_mass, torch.zeros_like(weights))
    
    # Specialist Weighted Average across all classes: [N, 1, C]
    # We expand weights [N, E, 1] to multiply [N, E, C]
    specialist_pred = (normalized_weights.unsqueeze(-1) * pred_tensor).sum(dim=1, keepdim=True)
    
    # 3. Defer Integration
    effective_defer = torch.where(weight_mass > 0, defer_prob, torch.zeros_like(defer_prob))
    # effective_defer is [N, 1], expanded to [N, 1, C]
    blended = (1.0 - effective_defer.unsqueeze(-1)) * full_pred + effective_defer.unsqueeze(-1) * specialist_pred

    diagnostics = {
        "router_kind": router_outputs.router_kind,
        "full_index": int(router_outputs.full_index),
        "mean_defer_prob": float(defer_prob.mean().item()),
        "effective_defer_rate": float((effective_defer > 0).float().mean().item()),
        "mean_specialist_mass": float(weight_mass.mean().item()),
    }
    
    # Final shape [N, C] or [N] if C=1
    result = blended.squeeze(1)
    if result.shape[1] == 1:
        result = result.squeeze(1)
        
    return IntegrationOutputs(
        predictions=result.detach().cpu().numpy().astype(np.float32),
        normalized_weights=normalized_weights.detach().cpu().numpy().astype(np.float32),
        defer_prob=defer_prob.detach().cpu().numpy().astype(np.float32),
        diagnostics=diagnostics,
    )
