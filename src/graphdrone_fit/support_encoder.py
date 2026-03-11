from __future__ import annotations

from dataclasses import dataclass

import torch

from .view_descriptor import ViewDescriptor


@dataclass(frozen=True)
class SupportEncoding:
    tensor: torch.Tensor
    feature_names: tuple[str, ...]


class ZeroSupportEncoder:
    def encode(
        self,
        *,
        n_rows: int,
        descriptors: tuple[ViewDescriptor, ...],
        support_tensor: torch.Tensor | None = None,
    ) -> SupportEncoding:
        """
        Phase I-A placeholder encoder.

        This keeps the support path explicit in the API without pretending support summaries
        are implemented yet. When no support tensor is provided, the returned width is zero.
        """
        n_experts = len(descriptors)
        if support_tensor is None:
            return SupportEncoding(
                tensor=torch.zeros((n_rows, n_experts, 0), dtype=torch.float32),
                feature_names=(),
            )
        tensor = torch.as_tensor(support_tensor, dtype=torch.float32)
        if tensor.ndim != 3 or tensor.shape[0] != n_rows or tensor.shape[1] != n_experts:
            raise ValueError(
                f"Expected support tensor shape [{n_rows}, {n_experts}, S], got {tuple(tensor.shape)}"
            )
        feature_names = tuple(f"support_{idx}" for idx in range(tensor.shape[2]))
        return SupportEncoding(tensor=tensor, feature_names=feature_names)
