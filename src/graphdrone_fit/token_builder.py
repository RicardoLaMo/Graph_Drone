from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from .view_descriptor import ViewDescriptor
from .support_encoder import SupportEncoding

@dataclass(frozen=True)
class TokenBatch:
    tokens: torch.Tensor  # [B, E, D]
    expert_ids: tuple[str, ...]
    field_slices: dict[str, tuple[int, int]]
    field_names: dict[str, tuple[str, ...]]

class UniversalTokenBuilder:
    """
    Consolidated Token Builder for GraphDrone.
    Handles Prediction, Quality (Sigma2), Support Moments, and SNR.
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
    ) -> TokenBatch:
        pred_tensor = torch.as_tensor(predictions, dtype=torch.float32)
        device = pred_tensor.device
        expert_ids = tuple(d.expert_id for d in descriptors)
        full_index = expert_ids.index(full_expert_id)
        
        # 1. Prediction Residuals
        full_pred = pred_tensor[:, full_index : full_index + 1]
        row_mean = pred_tensor.mean(dim=1, keepdim=True)
        prediction_fields = torch.stack(
            (pred_tensor, pred_tensor - full_pred, pred_tensor - row_mean),
            dim=-1
        )

        # 2. Support Moments & SNR
        support_fields = []
        support_names_list = []
        if support_encoding is not None:
            moments = support_encoding.tensor.to(device)
            # Ensure moments is [B, E, D]
            if moments.ndim == 2:
                moments = moments.unsqueeze(-1)
            support_fields.append(moments)
            support_names_list.extend(support_encoding.feature_names)
            
            # Integrated SNR calculation - only if we have at least mean and var
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

        # 5. Static Descriptors
        descriptor_tensor, descriptor_names = self._build_descriptor_tensor(descriptors)
        descriptor_tensor = descriptor_tensor.to(device).unsqueeze(0).expand(pred_tensor.shape[0], -1, -1)

        # Assemble
        parts = [prediction_fields] + support_fields + [descriptor_tensor]
        
        # Ensure all parts are on the same device and compatible shape before cat
        device = pred_tensor.device
        safe_parts = [p.to(device) for p in parts]
        tokens = torch.cat(safe_parts, dim=-1)
        
        # Slices
        slices = {}
        names = {}
        cursor = 0
        
        slices["prediction"] = (cursor, cursor + prediction_fields.shape[-1])
        names["prediction"] = ("raw", "residual_full", "residual_mean")
        cursor += prediction_fields.shape[-1]
        
        if support_fields:
            support_dim = sum(p.shape[-1] for p in support_fields)
            slices["support"] = (cursor, cursor + support_dim)
            names["support"] = tuple(support_names_list)
            cursor += support_dim
            
        slices["descriptor"] = (cursor, cursor + descriptor_tensor.shape[-1])
        names["descriptor"] = descriptor_names
        
        return TokenBatch(tokens, expert_ids, slices, names)

    def _build_descriptor_tensor(self, descriptors: tuple[ViewDescriptor, ...]) -> tuple[torch.Tensor, tuple[str, ...]]:
        rows = []
        for d in descriptors:
            rows.append([
                float(d.is_anchor),
                float(d.input_dim),
                float(d.preferred_k),
                # One-hot simplified family (FULL vs Subspace)
                1.0 if d.family == "FULL" else 0.0,
                1.0 if d.family == "structural_subspace" else 0.0,
                1.0 if d.family == "local_support" else 0.0,
            ])
        names = ("is_anchor", "input_dim", "preferred_k", "fam_full", "fam_subspace", "fam_support")
        return torch.tensor(rows, dtype=torch.float32), names
