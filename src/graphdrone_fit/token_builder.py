from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .support_encoder import SupportEncoding
from .view_descriptor import ViewDescriptor


@dataclass(frozen=True)
class TokenBatch:
    tokens: torch.Tensor
    expert_ids: tuple[str, ...]
    field_slices: dict[str, tuple[int, int]]


class PerViewTokenBuilder:
    def build(
        self,
        *,
        predictions: np.ndarray,
        descriptors: tuple[ViewDescriptor, ...],
        full_expert_id: str,
        quality_features: np.ndarray | torch.Tensor | None = None,
        support_encoding: SupportEncoding | None = None,
    ) -> TokenBatch:
        """
        Build per-row, per-expert tokens from prediction-side and optional prior/support fields.

        The current residual fields are intentionally row-conditioned:
        - raw expert prediction
        - prediction relative to the FULL anchor
        - prediction relative to the row-wise portfolio mean

        The row-wise mean is used as a neutral within-row baseline for Phase I-A tokenization.
        It is not yet a learned family baseline or a final routing objective.
        """
        pred_tensor = torch.as_tensor(predictions, dtype=torch.float32)
        if pred_tensor.ndim != 2:
            raise ValueError(f"Expected predictions with shape [N, E], got {tuple(pred_tensor.shape)}")
        expert_ids = tuple(descriptor.expert_id for descriptor in descriptors)
        if full_expert_id not in expert_ids:
            raise ValueError(f"full_expert_id={full_expert_id!r} is missing from {expert_ids!r}")
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

        quality_tensor = _coerce_optional_tensor(
            quality_features,
            n_rows=pred_tensor.shape[0],
            n_experts=pred_tensor.shape[1],
        )
        if support_encoding is None:
            support_tensor = torch.zeros((pred_tensor.shape[0], pred_tensor.shape[1], 0), dtype=torch.float32)
        else:
            support_tensor = support_encoding.tensor
        descriptor_tensor = _build_descriptor_tensor(descriptors).unsqueeze(0).expand(pred_tensor.shape[0], -1, -1)

        parts = [prediction_fields, quality_tensor, support_tensor, descriptor_tensor]
        field_slices: dict[str, tuple[int, int]] = {}
        cursor = 0
        names = ("prediction", "quality", "support", "descriptor")
        for name, tensor in zip(names, parts):
            width = tensor.shape[-1]
            field_slices[name] = (cursor, cursor + width)
            cursor += width
        tokens = torch.cat(parts, dim=-1)
        return TokenBatch(tokens=tokens, expert_ids=expert_ids, field_slices=field_slices)


def _coerce_optional_tensor(
    quality_features: np.ndarray | torch.Tensor | None,
    *,
    n_rows: int,
    n_experts: int,
) -> torch.Tensor:
    if quality_features is None:
        return torch.zeros((n_rows, n_experts, 0), dtype=torch.float32)
    tensor = torch.as_tensor(quality_features, dtype=torch.float32)
    if tensor.ndim != 3 or tensor.shape[0] != n_rows or tensor.shape[1] != n_experts:
        raise ValueError(
            f"Expected quality tensor shape [{n_rows}, {n_experts}, Q], got {tuple(tensor.shape)}"
        )
    return tensor


def _build_descriptor_tensor(descriptors: tuple[ViewDescriptor, ...]) -> torch.Tensor:
    families = sorted({descriptor.family for descriptor in descriptors})
    family_to_index = {family: idx for idx, family in enumerate(families)}
    projection_kinds = sorted({descriptor.projection_kind for descriptor in descriptors})
    projection_to_index = {kind: idx for idx, kind in enumerate(projection_kinds)}

    rows = []
    for descriptor in descriptors:
        family_one_hot = np.zeros(len(families), dtype=np.float32)
        family_one_hot[family_to_index[descriptor.family]] = 1.0
        projection_one_hot = np.zeros(len(projection_kinds), dtype=np.float32)
        projection_one_hot[projection_to_index[descriptor.projection_kind]] = 1.0
        rows.append(
            np.concatenate(
                [
                    np.array(
                        [
                            float(descriptor.is_anchor),
                            float(descriptor.input_dim),
                            float(len(descriptor.tags)),
                        ],
                        dtype=np.float32,
                    ),
                    family_one_hot,
                    projection_one_hot,
                ]
            )
        )
    return torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32)
