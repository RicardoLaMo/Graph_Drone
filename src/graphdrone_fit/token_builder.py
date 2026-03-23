from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import HyperbolicDescriptorConfig
from .hyperbolic import HyperbolicDescriptorEncoder
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

class UniversalTokenBuilder(nn.Module):
    """
    Consolidated Token Builder for GraphDrone.
    Handles Prediction, Quality (Sigma2), Support Moments, SNR, and Geometric Observers.
    """
    def __init__(self, hyperbolic_config: HyperbolicDescriptorConfig | None = None):
        super().__init__()
        self.hyperbolic_config = hyperbolic_config.validate() if hyperbolic_config is not None else None
        self.hyperbolic_encoder = (
            HyperbolicDescriptorEncoder(self.hyperbolic_config)
            if self.hyperbolic_config is not None and self.hyperbolic_config.enabled
            else None
        )

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
        quality_encoding: Optional["QualityEncoding"] = None,
    ) -> TokenBatch:
        pred_tensor = torch.as_tensor(predictions, dtype=torch.float32)
        device = pred_tensor.device
        expert_ids = tuple(d.expert_id for d in descriptors)
        full_index = expert_ids.index(full_expert_id)
        
        # 1. Prediction Residuals & Consensus
        full_pred = pred_tensor[:, full_index : full_index + 1]
        row_mean = pred_tensor.mean(dim=1, keepdim=True)
        if pred_tensor.shape[1] > 1:
            prediction_consensus = pred_tensor.std(dim=1, keepdim=True)
        else:
            prediction_consensus = torch.zeros((pred_tensor.shape[0], 1), device=device)
        
        prediction_fields = torch.stack(
            (pred_tensor, pred_tensor - full_pred, pred_tensor - row_mean),
            dim=-1
        )
        consensus_expanded = prediction_consensus.unsqueeze(1).expand(-1, len(expert_ids), 1)
        prediction_fields = torch.cat([prediction_fields, consensus_expanded], dim=-1)

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

        # 5b. Quality Encoding (bagged variance — real uncertainty signal)
        if quality_encoding is not None:
            q_tensor = quality_encoding.tensor.to(device)  # [N, E, D_q]
            support_fields.append(q_tensor)
            support_names_list.extend(quality_encoding.feature_names)

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
        names["prediction"] = ("raw", "residual_full", "residual_mean", "disagreement_std")
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
        if self.hyperbolic_encoder is not None:
            return self.hyperbolic_encoder(descriptors)

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

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    @torch.no_grad()
    def project_hyperbolic_parameters_(self) -> None:
        if self.hyperbolic_encoder is not None:
            self.hyperbolic_encoder.project_parameters_()
