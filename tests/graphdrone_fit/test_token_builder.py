from __future__ import annotations

import numpy as np

from src.graphdrone_fit.support_encoder import MomentSupportEncoder
from src.graphdrone_fit.token_builder import (
    PerViewTokenBuilder,
    build_legacy_quality_encoding,
    build_legacy_quality_encoding_from_flat,
)
from src.graphdrone_fit.view_descriptor import ViewDescriptor, normalize_descriptor_set


def _descriptors() -> tuple[ViewDescriptor, ...]:
    return normalize_descriptor_set(
        [
            ViewDescriptor(
                expert_id="FULL",
                family="FULL",
                view_name="FULL",
                projection_kind="identity_subselect",
                input_dim=2,
                input_indices=(0, 1),
                is_anchor=True,
                tags=("anchor",),
            ),
            ViewDescriptor(
                expert_id="GEO",
                family="domain_semantic",
                view_name="GEO",
                projection_kind="identity_subselect",
                input_dim=1,
                input_indices=(1,),
                tags=("registry",),
            ),
            ViewDescriptor(
                expert_id="LOWRANK",
                family="structural_subspace",
                view_name="LOWRANK",
                projection_kind="external_transform",
                input_dim=1,
                tags=("transformed",),
            ),
        ],
        required_anchor_id="FULL",
    )


def test_build_legacy_quality_encoding_maps_scalar_priors_to_per_expert_tokens() -> None:
    encoding = build_legacy_quality_encoding(
        view_names=["FULL", "GEO", "LOWRANK"],
        sigma2_v=np.array([[0.1, 0.4, 0.9], [0.2, 0.3, 0.5]], dtype=np.float32),
        j_flat=np.array([[0.8, 0.4, 0.2], [0.7, 0.5, 0.1]], dtype=np.float32),
        mean_j=np.array([0.4667, 0.4333], dtype=np.float32),
    )
    assert encoding.tensor.shape == (2, 3, 5)
    assert encoding.feature_names == (
        "quality_sigma2_self",
        "quality_sigma2_centered",
        "quality_pair_overlap_mean",
        "quality_pair_overlap_max",
        "quality_mean_J_global",
    )


def test_build_legacy_quality_encoding_from_flat_parses_expected_layout() -> None:
    flat = np.array([[0.2, 0.4, 0.6, 0.9, 0.5, 0.3, 0.5667]], dtype=np.float32)
    encoding = build_legacy_quality_encoding_from_flat(
        view_names=["FULL", "GEO", "LOWRANK"],
        flat_quality=flat,
    )
    assert encoding.tensor.shape == (1, 3, 5)
    assert np.isclose(float(encoding.tensor[0, 1, 0]), 0.4)


def test_support_encoder_summarizes_neighbor_tensors() -> None:
    support = np.array(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[0.5, 1.5], [1.5, 2.5]],
                [[2.0, 3.0], [4.0, 5.0]],
            ]
        ],
        dtype=np.float32,
    )
    encoding = MomentSupportEncoder().encode(
        n_rows=1,
        descriptors=_descriptors(),
        support_tensor=support,
    )
    assert encoding.tensor.shape == (1, 3, 22)
    assert encoding.feature_names[-5:] == (
        "support_weighted_anchor_l2",
        "support_weighted_anchor_cosine",
        "support_radius_mean",
        "support_effective_fraction",
        "support_count",
    )
    anchor_slice = encoding.tensor[0, 0]
    assert float(anchor_slice[6]) == 0.0
    assert float(anchor_slice[10]) == 0.0
    assert float(anchor_slice[11]) == 0.0
    assert np.isclose(float(anchor_slice[12]), 1.0)
    assert float(anchor_slice[-5]) == 0.0
    assert np.isclose(float(anchor_slice[-4]), 1.0)
    assert 0.0 < float(anchor_slice[-2]) <= 1.0


def test_token_builder_records_field_names_for_quality_support_and_descriptor() -> None:
    predictions = np.array([[1.0, 1.5, 0.5]], dtype=np.float32)
    quality = build_legacy_quality_encoding_from_flat(
        view_names=["FULL", "GEO", "LOWRANK"],
        flat_quality=np.array([[0.2, 0.4, 0.6, 0.9, 0.5, 0.3, 0.5667]], dtype=np.float32),
    )
    support = MomentSupportEncoder().encode(
        n_rows=1,
        descriptors=_descriptors(),
        support_tensor=np.array(
            [
                [
                    [[1.0], [2.0]],
                    [[0.5], [1.5]],
                    [[2.0], [3.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )
    batch = PerViewTokenBuilder().build(
        predictions=predictions,
        descriptors=_descriptors(),
        full_expert_id="FULL",
        quality_features=quality,
        support_encoding=support,
    )
    assert batch.tokens.shape[0] == 1
    assert batch.field_names["quality"][0] == "quality_sigma2_self"
    assert "quality_sigma2_self_minus_anchor" in batch.field_names["quality"]
    assert "quality_sigma2_self_minus_row_mean" in batch.field_names["quality"]
    assert batch.field_names["support"][-1] == "support_count_minus_row_mean"
    assert "support_anchor_mean_l2" in batch.field_names["support"]
    assert "support_weighted_anchor_cosine" in batch.field_names["support"]
    assert "support_weighted_anchor_cosine_minus_row_mean" in batch.field_names["support"]
    assert batch.field_names["descriptor"][0] == "descriptor_is_anchor"
    descriptor_index = batch.field_names["descriptor"].index("descriptor_family_domain_semantic")
    descriptor_slice = batch.field_slices["descriptor"]
    descriptor_values = batch.tokens[0, 1, descriptor_slice[0] : descriptor_slice[1]]
    assert float(descriptor_values[descriptor_index]) == 1.0
