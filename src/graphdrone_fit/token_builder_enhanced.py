from __future__ import annotations
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass

# Borrowing from graphdrone_fit
from graphdrone_fit.token_builder import TokenBatch, QualityEncoding, _build_descriptor_tensor
from graphdrone_fit.view_descriptor import ViewDescriptor
from graphdrone_fit.support_encoder import SupportEncoding

class EnhancedTokenBuilder:
    """
    Challenger Token Builder: 
    Enhances tokens with real kNN-based support moments (mean, var, skew, kurtosis).
    """
    def build(
        self,
        *,
        predictions: np.ndarray,
        descriptors: tuple[ViewDescriptor, ...],
        full_expert_id: str,
        quality_features: np.ndarray | torch.Tensor | QualityEncoding | None = None,
        support_encoding: SupportEncoding | None = None,
    ) -> TokenBatch:
        pred_tensor = torch.as_tensor(predictions, dtype=torch.float32)
        expert_ids = tuple(descriptor.expert_id for descriptor in descriptors)
        full_index = expert_ids.index(full_expert_id)
        
        full_pred = pred_tensor[:, full_index : full_index + 1]
        row_mean = pred_tensor.mean(dim=1, keepdim=True)
        prediction_fields = torch.stack(
            (
                pred_tensor,
                pred_tensor - full_pred,
                pred_tensor - row_mean,
            ),
            dim=-1,
        )

        if isinstance(quality_features, QualityEncoding):
            quality_tensor = quality_features.tensor
            quality_names = quality_features.feature_names
        else:
            quality_tensor = torch.zeros((pred_tensor.shape[0], pred_tensor.shape[1], 0), dtype=torch.float32)
            quality_names = ()

        if support_encoding is not None:
            support_tensor = support_encoding.tensor
            support_names = support_encoding.feature_names
        else:
            support_tensor = torch.zeros((pred_tensor.shape[0], pred_tensor.shape[1], 0), dtype=torch.float32)
            support_names = ()

        # 4. Descriptor Fields
        descriptor_tensor, descriptor_names = _build_descriptor_tensor(descriptors)
        descriptor_tensor = descriptor_tensor.unsqueeze(0).expand(pred_tensor.shape[0], -1, -1)

        # Combine - Ensure device parity
        parts = [prediction_fields, quality_tensor, support_tensor, descriptor_tensor]
        device = prediction_fields.device
        parts = [p.to(device) for p in parts]
        tokens = torch.cat(parts, dim=-1)
        
        field_slices = {}
        field_names = {}
        cursor = 0
        
        field_slices["prediction"] = (cursor, cursor + prediction_fields.shape[-1])
        field_names["prediction"] = ("prediction_raw", "prediction_minus_full", "prediction_minus_row_mean")
        cursor += prediction_fields.shape[-1]
        
        field_slices["quality"] = (cursor, cursor + quality_tensor.shape[-1])
        field_names["quality"] = quality_names
        cursor += quality_tensor.shape[-1]
        
        field_slices["support"] = (cursor, cursor + support_tensor.shape[-1])
        field_names["support"] = support_names
        cursor += support_tensor.shape[-1]
        
        field_slices["descriptor"] = (cursor, cursor + descriptor_tensor.shape[-1])
        field_names["descriptor"] = descriptor_names
        
        return TokenBatch(
            tokens=tokens,
            expert_ids=expert_ids,
            field_slices=field_slices,
            field_names=field_names
        )

def compute_real_support_encoding(X_train_views, X_test_views, y_train, ks=None):
    """
    Computes real kNN moments for each view.
    X_train_views: list of [N_tr, F_v]
    X_test_views: list of [N_te, F_v]
    y_train: [N_tr]
    """
    n_test = X_test_views[0].shape[0]
    n_experts = len(X_train_views)
    if ks is None:
        ks = [15] * n_experts
        
    all_moments = []

    for v_idx in range(n_experts):
        k = ks[v_idx]
        knn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train_views[v_idx])
        indices = knn.kneighbors(X_test_views[v_idx], return_distance=False)
        
        neighbor_labels = y_train[indices] # [N_te, K]
        
        mean = neighbor_labels.mean(axis=1)
        var = neighbor_labels.var(axis=1)
        
        # Skewness proxy
        diff = neighbor_labels - mean[:, None]
        skew = (diff**3).mean(axis=1)
        kurt = (diff**4).mean(axis=1)
        
        moments = np.stack([mean, var, skew, kurt], axis=1) # [N_te, 4]
        all_moments.append(moments)
        
    return torch.as_tensor(np.stack(all_moments, axis=1), dtype=torch.float32)
