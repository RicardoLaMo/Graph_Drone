from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from .view_descriptor import ViewDescriptor
from .support_encoder import SupportEncoding

@dataclass(frozen=True)
class QualityEncoding:
    tensor: torch.Tensor
    feature_names: tuple[str, ...]

@dataclass(frozen=True)
class TokenBatch:
    tokens: torch.Tensor  # [B, E, D]
    expert_ids: tuple[str, ...]
    field_slices: dict[str, tuple[int, int]]
    field_names: dict[str, tuple[str, ...]]
    task_token: Optional[torch.Tensor] = None # [B, 1, D_task]

class UniversalTokenBuilder:
    """
    Consolidated Token Builder for GraphDrone.
    Handles Prediction, Quality (Sigma2), Support Moments, SNR, and Geometric Observers.
    """
    def build(
        self,
        *,
        predictions: np.ndarray,
        descriptors: tuple[ViewDescriptor, ...],
        full_expert_id: str,
        support_encoding: Optional[SupportEncoding] = None,
        neural_support: Optional[torch.Tensor] = None,
        prior_alignment: Optional[torch.Tensor] = None,
        geometric_obs: Optional[torch.Tensor] = None,
    ) -> TokenBatch:
        pred_tensor = torch.as_tensor(predictions, dtype=torch.float32) # [N, E, C]
        if pred_tensor.ndim == 2:
            pred_tensor = pred_tensor.unsqueeze(-1) # Ensure [N, E, 1]
            
        device = pred_tensor.device
        expert_ids = tuple(d.expert_id for d in descriptors)
        full_index = expert_ids.index(full_expert_id)
        
        # 1. Prediction Residuals, Consensus & Entropy
        # For multi-class, we focus on the anchor (Full) and the class distributions
        full_pred = pred_tensor[:, full_index : full_index + 1, :] # [N, 1, C]
        row_mean = pred_tensor.mean(dim=1, keepdim=True)           # [N, 1, C]
        
        # Entropy computation: -sum(p * log(p))
        # Helps router detect when experts are uncertain
        eps = 1e-8
        entropy = -(pred_tensor * torch.log(pred_tensor + eps)).sum(dim=-1, keepdim=True) # [N, E, 1]
        
        # Disagreement is the mean variation across experts for each class
        if pred_tensor.shape[1] > 1:
            prediction_consensus = pred_tensor.std(dim=1, keepdim=True).mean(dim=-1, keepdim=True) # [N, 1, 1]
        else:
            prediction_consensus = torch.zeros((pred_tensor.shape[0], 1, 1), device=device)
        
        res_full = (pred_tensor - full_pred).mean(dim=-1, keepdim=True)
        res_mean = (pred_tensor - row_mean).mean(dim=-1, keepdim=True)
        
        max_prob, _ = pred_tensor.max(dim=-1, keepdim=True)

        prediction_fields = torch.cat(
            [max_prob, res_full, res_mean, entropy, prediction_consensus.expand(-1, len(expert_ids), -1)],
            dim=-1
        )

        # 2. Support Moments & SNR
        support_fields = []
        support_names_list = []
        if support_encoding is not None:
            moments = support_encoding.tensor.to(device)
            if moments.ndim == 2:
                moments = moments.unsqueeze(-1)
            support_fields.append(moments)
            support_names_list.extend(support_encoding.feature_names)
            
            if moments.shape[-1] >= 2:
                mean_neighbors = moments[:, :, 0]
                var_neighbors = moments[:, :, 1]
                signal = torch.abs(pred_tensor - mean_neighbors)
                noise = torch.sqrt(var_neighbors + 1e-6)
                log_snr = torch.log1p(signal / noise).unsqueeze(-1)
                support_fields.append(log_snr)
                support_names_list.append("log_snr")

        # 3. Neural Support
        if neural_support is not None:
            support_fields.append(neural_support.to(device))
            support_names_list.extend([f"neural_support_{i}" for i in range(neural_support.shape[-1])])

        # 4. Prior Alignment
        if prior_alignment is not None:
            support_fields.append(prior_alignment.to(device))
            support_names_list.append("prior_alignment")

        # 5. Geometric Observers (GoRA Logic)
        if geometric_obs is not None:
            support_fields.append(geometric_obs.to(device))
            support_names_list.extend(["kappa", "lid"])

        # 6. Static Descriptors
        descriptor_tensor, descriptor_names = self._build_descriptor_tensor(descriptors)
        descriptor_tensor = descriptor_tensor.to(device).unsqueeze(0).expand(pred_tensor.shape[0], -1, -1)

        # Assemble
        parts = [prediction_fields] + support_fields + [descriptor_tensor]
        safe_parts = [p.to(device) for p in parts]
        tokens = torch.cat(safe_parts, dim=-1)
        
        # Slices
        slices = {}
        names = {}
        cursor = 0
        
        slices["prediction"] = (cursor, cursor + prediction_fields.shape[-1])
        names["prediction"] = ("max_val", "residual_full", "residual_mean", "entropy", "disagreement_std")
        cursor += prediction_fields.shape[-1]
        
        if support_fields:
            support_dim = sum(p.shape[-1] for p in support_fields)
            slices["support"] = (cursor, cursor + support_dim)
            names["support"] = tuple(support_names_list)
            cursor += support_dim
            
        slices["descriptor"] = (cursor, cursor + descriptor_tensor.shape[-1])
        names["descriptor"] = descriptor_names
        
        # Prepare Task Token for batch: [B, 1, D_task]
        batch_task_token = None
        if support_encoding is not None and support_encoding.task_token is not None:
            batch_task_token = support_encoding.task_token.to(device).expand(pred_tensor.shape[0], -1, -1)

        return TokenBatch(tokens, expert_ids, slices, names, task_token=batch_task_token)

    def _build_descriptor_tensor(self, descriptors: tuple[ViewDescriptor, ...]) -> tuple[torch.Tensor, tuple[str, ...]]:
        rows = []
        for d in descriptors:
            rows.append([
                float(d.is_anchor),
                float(getattr(d, 'input_dim', 0)),
                float(getattr(d, 'preferred_k', 15)),
                1.0 if d.family == "FULL" else 0.0,
                1.0 if d.family == "structural_subspace" else 0.0,
                1.0 if d.family == "local_support" else 0.0,
            ])
        names = ("is_anchor", "input_dim", "preferred_k", "fam_full", "fam_subspace", "fam_support")
        return torch.tensor(rows, dtype=torch.float32), names
