from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .view_descriptor import ViewDescriptor


@dataclass(frozen=True)
class SupportEncoding:
    tensor: torch.Tensor
    feature_names: tuple[str, ...]
    task_token: torch.Tensor | None = None # [1, D_task] Global summary


class MomentSupportEncoder:
    def encode(
        self,
        *,
        n_rows: int,
        descriptors: tuple[ViewDescriptor, ...],
        support_tensor: np.ndarray | torch.Tensor | SupportEncoding | None = None,
        full_matrix: np.ndarray | torch.Tensor | None = None,
    ) -> SupportEncoding:
        n_experts = len(descriptors)
        
        # 1. Global Task Token (Bayesian Prior)
        task_token = None
        if full_matrix is not None:
            X = torch.as_tensor(full_matrix, dtype=torch.float32)
            # Stats: Mean, Std, Sparsity, Dim
            g_mean = X.mean(dim=0).mean().unsqueeze(0)
            g_std = X.std(dim=0).mean().unsqueeze(0)
            g_sparsity = (X == 0).float().mean().unsqueeze(0)
            g_dim = torch.tensor([float(X.shape[1])], device=X.device)
            task_token = torch.cat([g_mean, g_std, g_sparsity, g_dim]).unsqueeze(0) # [1, 4]

        if isinstance(support_tensor, SupportEncoding):
            tensor = torch.as_tensor(support_tensor.tensor, dtype=torch.float32)
            self._validate_shape(tensor, n_rows=n_rows, n_experts=n_experts)
            return SupportEncoding(
                tensor=tensor,
                feature_names=tuple(support_tensor.feature_names),
                task_token=task_token if task_token is not None else support_tensor.task_token
            )

        if support_tensor is None:
            return SupportEncoding(
                tensor=torch.zeros((n_rows, n_experts, 0), dtype=torch.float32),
                feature_names=(),
                task_token=task_token
            )

        tensor = torch.as_tensor(support_tensor, dtype=torch.float32)
        if tensor.ndim == 3:
            self._validate_shape(tensor, n_rows=n_rows, n_experts=n_experts)
            feature_names = tuple(f"support_{idx}" for idx in range(tensor.shape[2]))
            return SupportEncoding(tensor=tensor, feature_names=feature_names)

        if tensor.ndim == 4 and tensor.shape[0] == n_rows and tensor.shape[1] == n_experts:
            mean = tensor.mean(dim=2)
            std = tensor.std(dim=2, unbiased=False)
            abs_max = tensor.abs().max(dim=2).values
            support_count = torch.full(
                (n_rows, n_experts, 1),
                float(tensor.shape[2]),
                dtype=torch.float32,
                device=tensor.device,
            )
            summary = torch.cat([mean, std, abs_max, support_count], dim=-1)
            n_support_features = tensor.shape[-1]
            feature_names = (
                *(f"support_mean_{idx}" for idx in range(n_support_features)),
                *(f"support_std_{idx}" for idx in range(n_support_features)),
                *(f"support_absmax_{idx}" for idx in range(n_support_features)),
                "support_count",
            )
            return SupportEncoding(
                tensor=summary,
                feature_names=tuple(feature_names),
                task_token=task_token
            )

        raise ValueError(
            f"Expected support tensor shape [{n_rows}, {n_experts}, S] or "
            f"[{n_rows}, {n_experts}, K, F], got {tuple(tensor.shape)}"
        )

    @staticmethod
    def _validate_shape(tensor: torch.Tensor, *, n_rows: int, n_experts: int) -> None:
        if tensor.ndim != 3 or tensor.shape[0] != n_rows or tensor.shape[1] != n_experts:
            raise ValueError(
                f"Expected support tensor shape [{n_rows}, {n_experts}, S], got {tuple(tensor.shape)}"
            )


class ZeroSupportEncoder(MomentSupportEncoder):
    """
    Backwards-compatible alias kept so the Phase I-A surface does not break while Phase I-C
    upgrades the actual encoding behavior from zero-width placeholders to pooled summaries.
    """
