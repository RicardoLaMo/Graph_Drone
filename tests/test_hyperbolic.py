from __future__ import annotations

import torch

from graphdrone_fit.config import HyperbolicDescriptorConfig
from graphdrone_fit.hyperbolic import HyperbolicDescriptorEncoder, PoincareBallEmbedding
from graphdrone_fit.view_descriptor import ViewDescriptor


def test_poincare_ball_embedding_stays_inside_ball():
    embedding = PoincareBallEmbedding(num_embeddings=5, embedding_dim=3, curvature=1.0, max_norm=0.9)
    vectors = embedding(torch.tensor([0, 1, 2]))
    norms = torch.linalg.norm(vectors, dim=-1)
    assert torch.all(norms < 0.9 + 1e-6)


def test_log_exp_maps_round_trip_at_origin():
    embedding = PoincareBallEmbedding(num_embeddings=2, embedding_dim=4)
    tangent = torch.tensor([[0.01, -0.02, 0.03, -0.01]], dtype=torch.float32)
    point = embedding.exp_map_zero(tangent)
    recovered = embedding.log_map_zero(point)
    assert torch.allclose(recovered, tangent, atol=1e-4)


def test_hyperbolic_distance_is_symmetric_and_finite():
    embedding = PoincareBallEmbedding(num_embeddings=3, embedding_dim=2)
    x = embedding(torch.tensor([0]))
    y = embedding(torch.tensor([1]))
    d_xy = embedding.distance(x, y)
    d_yx = embedding.distance(y, x)
    assert torch.isfinite(d_xy).all()
    assert torch.allclose(d_xy, d_yx, atol=1e-6)


def test_hyperbolic_descriptor_encoder_emits_expected_shape():
    encoder = HyperbolicDescriptorEncoder(HyperbolicDescriptorConfig(enabled=True, embedding_dim=3))
    descriptors = (
        ViewDescriptor(
            expert_id="FULL",
            family="FULL",
            view_name="Full",
            projection_kind="identity_subselect",
            input_dim=8,
            input_indices=tuple(range(8)),
            is_anchor=True,
        ),
        ViewDescriptor(
            expert_id="SUB0",
            family="structural_subspace",
            view_name="Sub",
            projection_kind="structural_subspace",
            input_dim=4,
            input_indices=(0, 1, 2, 3),
        ),
    )
    encoded, names = encoder(descriptors)
    assert encoded.shape == (2, 12)
    assert len(names) == 12
