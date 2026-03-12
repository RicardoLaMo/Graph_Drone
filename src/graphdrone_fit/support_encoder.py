from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .view_descriptor import ViewDescriptor


@dataclass(frozen=True)
class SupportEncoding:
    tensor: torch.Tensor
    feature_names: tuple[str, ...]


class MomentSupportEncoder:
    def encode(
        self,
        *,
        n_rows: int,
        descriptors: tuple[ViewDescriptor, ...],
        support_tensor: np.ndarray | torch.Tensor | SupportEncoding | None = None,
    ) -> SupportEncoding:
        n_experts = len(descriptors)
        if isinstance(support_tensor, SupportEncoding):
            tensor = torch.as_tensor(support_tensor.tensor, dtype=torch.float32)
            self._validate_shape(tensor, n_rows=n_rows, n_experts=n_experts)
            return SupportEncoding(
                tensor=tensor,
                feature_names=tuple(support_tensor.feature_names),
            )

        if support_tensor is None:
            return SupportEncoding(
                tensor=torch.zeros((n_rows, n_experts, 0), dtype=torch.float32),
                feature_names=(),
            )

        tensor = torch.as_tensor(support_tensor, dtype=torch.float32)
        if tensor.ndim == 3:
            self._validate_shape(tensor, n_rows=n_rows, n_experts=n_experts)
            feature_names = tuple(f"support_{idx}" for idx in range(tensor.shape[2]))
            return SupportEncoding(tensor=tensor, feature_names=feature_names)

        if tensor.ndim == 4 and tensor.shape[0] == n_rows and tensor.shape[1] == n_experts:
            anchor_index = _resolve_anchor_index(descriptors)
            mean = tensor.mean(dim=2)
            std = tensor.std(dim=2, unbiased=False)
            abs_max = tensor.abs().max(dim=2).values
            anchor_mean = mean[:, anchor_index : anchor_index + 1, :]
            anchor_std = std[:, anchor_index : anchor_index + 1, :]
            mean_minus_anchor = mean - anchor_mean
            std_minus_anchor = std - anchor_std
            mean_anchor_l2 = torch.linalg.vector_norm(mean_minus_anchor, dim=-1, keepdim=True)
            std_anchor_l2 = torch.linalg.vector_norm(std_minus_anchor, dim=-1, keepdim=True)
            mean_anchor_cosine = _cosine_against_anchor(mean, anchor_mean)
            weighted_mean, support_radius_mean, support_effective_fraction = _weighted_support_summary(
                tensor=tensor,
                centroid=mean,
            )
            weighted_mean_minus_anchor = weighted_mean - anchor_mean
            weighted_anchor_l2 = torch.linalg.vector_norm(
                weighted_mean_minus_anchor,
                dim=-1,
                keepdim=True,
            )
            weighted_anchor_cosine = _cosine_against_anchor(weighted_mean, anchor_mean)
            support_count = torch.full(
                (n_rows, n_experts, 1),
                float(tensor.shape[2]),
                dtype=torch.float32,
                device=tensor.device,
            )
            summary = torch.cat(
                [
                    mean,
                    std,
                    abs_max,
                    mean_minus_anchor,
                    std_minus_anchor,
                    mean_anchor_l2,
                    std_anchor_l2,
                    mean_anchor_cosine,
                    weighted_mean,
                    weighted_mean_minus_anchor,
                    weighted_anchor_l2,
                    weighted_anchor_cosine,
                    support_radius_mean,
                    support_effective_fraction,
                    support_count,
                ],
                dim=-1,
            )
            n_support_features = tensor.shape[-1]
            feature_names = (
                *(f"support_mean_{idx}" for idx in range(n_support_features)),
                *(f"support_std_{idx}" for idx in range(n_support_features)),
                *(f"support_absmax_{idx}" for idx in range(n_support_features)),
                *(f"support_mean_minus_anchor_{idx}" for idx in range(n_support_features)),
                *(f"support_std_minus_anchor_{idx}" for idx in range(n_support_features)),
                "support_anchor_mean_l2",
                "support_anchor_std_l2",
                "support_anchor_mean_cosine",
                *(f"support_weighted_mean_{idx}" for idx in range(n_support_features)),
                *(f"support_weighted_mean_minus_anchor_{idx}" for idx in range(n_support_features)),
                "support_weighted_anchor_l2",
                "support_weighted_anchor_cosine",
                "support_radius_mean",
                "support_effective_fraction",
                "support_count",
            )
            return SupportEncoding(
                tensor=summary,
                feature_names=tuple(feature_names),
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


def _resolve_anchor_index(descriptors: tuple[ViewDescriptor, ...]) -> int:
    anchor_indices = [idx for idx, descriptor in enumerate(descriptors) if descriptor.is_anchor]
    if not anchor_indices:
        raise ValueError("SupportEncoding requires at least one anchor descriptor")
    return anchor_indices[0]


def _cosine_against_anchor(
    tensor: torch.Tensor,
    anchor_tensor: torch.Tensor,
) -> torch.Tensor:
    cosine = torch.nn.functional.cosine_similarity(
        tensor,
        anchor_tensor.expand_as(tensor),
        dim=-1,
        eps=1e-6,
    )
    return cosine.unsqueeze(-1).to(dtype=torch.float32)


def _weighted_support_summary(
    *,
    tensor: torch.Tensor,
    centroid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centered = tensor - centroid.unsqueeze(2)
    item_radius = torch.linalg.vector_norm(centered, dim=-1)
    weights = torch.softmax(-item_radius, dim=2)
    weighted_mean = (weights.unsqueeze(-1) * tensor).sum(dim=2)
    support_radius_mean = item_radius.mean(dim=2, keepdim=True).to(dtype=torch.float32)
    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=2, keepdim=True)
    support_effective_fraction = (torch.exp(entropy) / float(tensor.shape[2])).to(dtype=torch.float32)
    return weighted_mean.to(dtype=torch.float32), support_radius_mean, support_effective_fraction
