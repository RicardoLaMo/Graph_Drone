from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .support_encoder import SupportEncoding
from .view_descriptor import ViewDescriptor


@dataclass(frozen=True)
class QualityEncoding:
    tensor: torch.Tensor
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class TokenBatch:
    tokens: torch.Tensor
    expert_ids: tuple[str, ...]
    field_slices: dict[str, tuple[int, int]]
    field_names: dict[str, tuple[str, ...]]


class PerViewTokenBuilder:
    def build(
        self,
        *,
        predictions: np.ndarray,
        descriptors: tuple[ViewDescriptor, ...],
        full_expert_id: str,
        quality_features: np.ndarray | torch.Tensor | QualityEncoding | None = None,
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

        Support fields are expected to already be anchor-aware summaries produced by
        `support_encoder.py`. This keeps the token builder focused on assembling
        semantically named token groups rather than inventing support semantics ad hoc.
        """
        pred_tensor = torch.as_tensor(predictions, dtype=torch.float32)
        if pred_tensor.ndim not in {2, 3}:
            raise ValueError(
                f"Expected predictions with shape [N, E] or [N, E, C], got {tuple(pred_tensor.shape)}"
            )
        expert_ids = tuple(descriptor.expert_id for descriptor in descriptors)
        if full_expert_id not in expert_ids:
            raise ValueError(f"full_expert_id={full_expert_id!r} is missing from {expert_ids!r}")
        full_index = expert_ids.index(full_expert_id)
        prediction_fields, prediction_feature_names = _build_prediction_fields(
            pred_tensor=pred_tensor,
            full_index=full_index,
        )

        quality_encoding = _coerce_quality_encoding(
            quality_features,
            n_rows=pred_tensor.shape[0],
            n_experts=pred_tensor.shape[1],
        )
        quality_tensor, quality_feature_names = _augment_quality_tensor(
            quality_encoding.tensor,
            feature_names=quality_encoding.feature_names,
            full_index=full_index,
        )
        if support_encoding is None:
            support_tensor = torch.zeros((pred_tensor.shape[0], pred_tensor.shape[1], 0), dtype=torch.float32)
            support_feature_names: tuple[str, ...] = ()
        else:
            support_tensor, support_feature_names = _augment_support_tensor(
                support_encoding.tensor,
                feature_names=tuple(support_encoding.feature_names),
            )
        descriptor_tensor, descriptor_feature_names = _build_descriptor_tensor(descriptors)
        descriptor_tensor = descriptor_tensor.unsqueeze(0).expand(pred_tensor.shape[0], -1, -1)

        parts = [prediction_fields, quality_tensor, support_tensor, descriptor_tensor]
        field_slices: dict[str, tuple[int, int]] = {}
        field_names: dict[str, tuple[str, ...]] = {}
        cursor = 0
        names = ("prediction", "quality", "support", "descriptor")
        for name, tensor in zip(names, parts):
            width = tensor.shape[-1]
            field_slices[name] = (cursor, cursor + width)
            if name == "prediction":
                field_names[name] = prediction_feature_names
            elif name == "quality":
                field_names[name] = quality_feature_names
            elif name == "support":
                field_names[name] = support_feature_names
            else:
                field_names[name] = descriptor_feature_names
            cursor += width
        tokens = torch.cat(parts, dim=-1)
        return TokenBatch(
            tokens=tokens,
            expert_ids=expert_ids,
            field_slices=field_slices,
            field_names=field_names,
        )


def _build_prediction_fields(
    *,
    pred_tensor: torch.Tensor,
    full_index: int,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    if pred_tensor.ndim == 2:
        full_pred = pred_tensor[:, full_index : full_index + 1]
        row_mean = pred_tensor.mean(dim=1, keepdim=True)
        fields = torch.stack(
            (
                pred_tensor,
                pred_tensor - full_pred,
                pred_tensor - row_mean,
            ),
            dim=-1,
        )
        names = (
            "prediction_raw",
            "prediction_minus_full",
            "prediction_minus_row_mean",
        )
        return fields, names

    if pred_tensor.ndim != 3:
        raise ValueError(f"Expected prediction tensor rank 2 or 3, got {pred_tensor.ndim}")
    full_pred = pred_tensor[:, full_index : full_index + 1, :]
    row_mean = pred_tensor.mean(dim=1, keepdim=True)
    safe_pred = pred_tensor.clamp_min(1e-8)
    full_safe_pred = full_pred.clamp_min(1e-8)

    max_prob = pred_tensor.max(dim=-1, keepdim=True).values
    full_max_prob = full_pred.max(dim=-1, keepdim=True).values
    entropy = -(safe_pred * safe_pred.log()).sum(dim=-1, keepdim=True)
    full_entropy = -(full_safe_pred * full_safe_pred.log()).sum(dim=-1, keepdim=True)

    top_k = min(2, pred_tensor.shape[-1])
    top_probs = torch.topk(pred_tensor, k=top_k, dim=-1).values
    if top_k == 1:
        margin = top_probs[..., :1]
    else:
        margin = (top_probs[..., :1] - top_probs[..., 1:2])

    fields = torch.cat(
        [
            pred_tensor,
            pred_tensor - full_pred,
            pred_tensor - row_mean,
            max_prob,
            margin,
            entropy,
            max_prob - full_max_prob,
            entropy - full_entropy,
        ],
        dim=-1,
    )
    n_classes = pred_tensor.shape[-1]
    names = (
        *(f"prediction_proba_class_{idx}" for idx in range(n_classes)),
        *(f"prediction_proba_minus_full_class_{idx}" for idx in range(n_classes)),
        *(f"prediction_proba_minus_row_mean_class_{idx}" for idx in range(n_classes)),
        "prediction_max_prob",
        "prediction_margin_top2",
        "prediction_entropy",
        "prediction_max_prob_minus_full",
        "prediction_entropy_minus_full",
    )
    return fields, tuple(names)


def _coerce_quality_encoding(
    quality_features: np.ndarray | torch.Tensor | QualityEncoding | None,
    *,
    n_rows: int,
    n_experts: int,
) -> QualityEncoding:
    if quality_features is None:
        return QualityEncoding(
            tensor=torch.zeros((n_rows, n_experts, 0), dtype=torch.float32),
            feature_names=(),
        )
    if isinstance(quality_features, QualityEncoding):
        tensor = torch.as_tensor(quality_features.tensor, dtype=torch.float32)
        if tensor.ndim != 3 or tensor.shape[0] != n_rows or tensor.shape[1] != n_experts:
            raise ValueError(
                f"Expected quality tensor shape [{n_rows}, {n_experts}, Q], got {tuple(tensor.shape)}"
            )
        return QualityEncoding(
            tensor=tensor,
            feature_names=tuple(quality_features.feature_names),
        )
    tensor = torch.as_tensor(quality_features, dtype=torch.float32)
    if tensor.ndim != 3 or tensor.shape[0] != n_rows or tensor.shape[1] != n_experts:
        raise ValueError(
            f"Expected quality tensor shape [{n_rows}, {n_experts}, Q], got {tuple(tensor.shape)}"
        )
    return QualityEncoding(
        tensor=tensor,
        feature_names=tuple(f"quality_{idx}" for idx in range(tensor.shape[2])),
    )


def _build_descriptor_tensor(descriptors: tuple[ViewDescriptor, ...]) -> tuple[torch.Tensor, tuple[str, ...]]:
    families = sorted({descriptor.family for descriptor in descriptors})
    family_to_index = {family: idx for idx, family in enumerate(families)}
    projection_kinds = sorted({descriptor.projection_kind for descriptor in descriptors})
    projection_to_index = {kind: idx for idx, kind in enumerate(projection_kinds)}
    max_input_dim = max((descriptor.input_dim for descriptor in descriptors), default=1)
    max_tag_count = max((len(descriptor.tags) for descriptor in descriptors), default=1)

    rows = []
    for descriptor in descriptors:
        family_one_hot = np.zeros(len(families), dtype=np.float32)
        family_one_hot[family_to_index[descriptor.family]] = 1.0
        projection_one_hot = np.zeros(len(projection_kinds), dtype=np.float32)
        projection_one_hot[projection_to_index[descriptor.projection_kind]] = 1.0
        normalized_input_dim = 0.0 if max_input_dim <= 0 else float(np.log1p(descriptor.input_dim) / np.log1p(max_input_dim))
        normalized_tag_count = 0.0 if max_tag_count <= 0 else float(len(descriptor.tags) / max_tag_count)
        rows.append(
            np.concatenate(
                [
                    np.array(
                        [
                            float(descriptor.is_anchor),
                            normalized_input_dim,
                            normalized_tag_count,
                        ],
                        dtype=np.float32,
                    ),
                    family_one_hot,
                    projection_one_hot,
                ]
            )
        )
    feature_names = (
        "descriptor_is_anchor",
        "descriptor_input_dim",
        "descriptor_tag_count",
        *(f"descriptor_family_{family}" for family in families),
        *(f"descriptor_projection_{kind}" for kind in projection_kinds),
    )
    return torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32), tuple(feature_names)


def _augment_quality_tensor(
    tensor: torch.Tensor,
    *,
    feature_names: tuple[str, ...],
    full_index: int,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    if tensor.shape[-1] == 0:
        return tensor, feature_names
    # `full_index` is the anchor expert index inside the current expert set.
    # These residuals are expert-relative, not row-index based.
    anchor = tensor[:, full_index : full_index + 1, :]
    row_mean = tensor.mean(dim=1, keepdim=True)
    augmented = torch.cat(
        [
            tensor,
            tensor - anchor,
            tensor - row_mean,
        ],
        dim=-1,
    )
    names = (
        *feature_names,
        *(f"{name}_minus_anchor" for name in feature_names),
        *(f"{name}_minus_row_mean" for name in feature_names),
    )
    return augmented, tuple(names)


def _augment_support_tensor(
    tensor: torch.Tensor,
    *,
    feature_names: tuple[str, ...],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    if tensor.shape[-1] == 0:
        return tensor, feature_names
    row_mean = tensor.mean(dim=1, keepdim=True)
    augmented = torch.cat(
        [
            tensor,
            tensor - row_mean,
        ],
        dim=-1,
    )
    names = (
        *feature_names,
        *(f"{name}_minus_row_mean" for name in feature_names),
    )
    return augmented, tuple(names)


def build_legacy_quality_encoding(
    *,
    view_names: list[str] | tuple[str, ...],
    sigma2_v: np.ndarray,
    j_flat: np.ndarray | None = None,
    mean_j: np.ndarray | None = None,
) -> QualityEncoding:
    sigma2 = np.asarray(sigma2_v, dtype=np.float32)
    if sigma2.ndim != 2:
        raise ValueError(f"Expected sigma2_v shape [N, V], got {sigma2.shape}")
    n_rows, n_experts = sigma2.shape
    if len(view_names) != n_experts:
        raise ValueError(f"Expected {n_experts} view names, got {len(view_names)}")

    if j_flat is None:
        j_mean = np.zeros((n_rows, n_experts), dtype=np.float32)
        j_max = np.zeros((n_rows, n_experts), dtype=np.float32)
    else:
        j_matrix = np.zeros((n_rows, n_experts, n_experts), dtype=np.float32)
        pairs = [(i, j) for i in range(n_experts) for j in range(i + 1, n_experts)]
        flat = np.asarray(j_flat, dtype=np.float32)
        if flat.shape != (n_rows, len(pairs)):
            raise ValueError(
                f"Expected j_flat shape [{n_rows}, {len(pairs)}], got {tuple(flat.shape)}"
            )
        for idx, (i, j) in enumerate(pairs):
            j_matrix[:, i, j] = flat[:, idx]
            j_matrix[:, j, i] = flat[:, idx]
        if n_experts == 1:
            j_mean = np.zeros((n_rows, 1), dtype=np.float32)
            j_max = np.zeros((n_rows, 1), dtype=np.float32)
        else:
            j_mean = j_matrix.sum(axis=2) / float(n_experts - 1)
            j_max = j_matrix.max(axis=2)

    mean_sigma2 = sigma2.mean(axis=1, keepdims=True)
    sigma2_centered = sigma2 - mean_sigma2
    if mean_j is None:
        mean_j_global = j_mean.mean(axis=1, keepdims=True) if n_experts > 0 else np.zeros((n_rows, 1), dtype=np.float32)
    else:
        mean_j_global = np.asarray(mean_j, dtype=np.float32).reshape(n_rows, 1)
    mean_j_broadcast = np.repeat(mean_j_global, n_experts, axis=1)

    tensor = np.stack(
        [
            sigma2,
            sigma2_centered,
            j_mean.astype(np.float32),
            j_max.astype(np.float32),
            mean_j_broadcast.astype(np.float32),
        ],
        axis=-1,
    )
    return QualityEncoding(
        tensor=torch.as_tensor(tensor, dtype=torch.float32),
        feature_names=legacy_quality_feature_names(),
    )


def build_legacy_quality_encoding_from_flat(
    *,
    view_names: list[str] | tuple[str, ...],
    flat_quality: np.ndarray,
) -> QualityEncoding:
    quality = np.asarray(flat_quality, dtype=np.float32)
    if quality.ndim != 2:
        raise ValueError(f"Expected flat quality shape [N, F], got {quality.shape}")
    n_experts = len(view_names)
    pair_count = n_experts * (n_experts - 1) // 2
    expected_width = n_experts + pair_count + 1
    if quality.shape[1] != expected_width:
        raise ValueError(
            f"Expected flat quality width {expected_width} for {n_experts} experts, got {quality.shape[1]}"
        )
    sigma2 = quality[:, :n_experts]
    j_flat = quality[:, n_experts : n_experts + pair_count]
    mean_j = quality[:, -1]
    return build_legacy_quality_encoding(
        view_names=view_names,
        sigma2_v=sigma2,
        j_flat=j_flat,
        mean_j=mean_j,
    )


def legacy_quality_feature_names() -> tuple[str, ...]:
    return (
        "quality_sigma2_self",
        "quality_sigma2_centered",
        "quality_pair_overlap_mean",
        "quality_pair_overlap_max",
        "quality_mean_J_global",
    )
