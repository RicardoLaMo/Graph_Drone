from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import HyperbolicDescriptorConfig
from .view_descriptor import VALID_EXPERT_FAMILIES, VALID_PROJECTION_KINDS, ViewDescriptor


class PoincareBallEmbedding(nn.Module):
    """Embedding table constrained to the Poincare ball."""

    def __init__(self, num_embeddings: int, embedding_dim: int, *, curvature: float = 1.0, max_norm: float = 0.95):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.max_norm = max_norm
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.project_parameters_()

    def project(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-9)
        max_radius = self.max_norm / math.sqrt(self.curvature)
        scale = torch.clamp(max_radius / norm, max=1.0)
        return x * scale

    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)
        sqrt_c = math.sqrt(self.curvature)
        factor = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return self.project(factor * v)

    def log_map_zero(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        x_norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-9)
        sqrt_c = math.sqrt(self.curvature)
        factor = torch.atanh(sqrt_c * x_norm) / (sqrt_c * x_norm)
        return factor * x

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        y = self.project(y)
        x_sq = torch.sum(x * x, dim=-1)
        y_sq = torch.sum(y * y, dim=-1)
        diff_sq = torch.sum((x - y) * (x - y), dim=-1)
        denom = (1.0 - self.curvature * x_sq).clamp_min(1e-6) * (1.0 - self.curvature * y_sq).clamp_min(1e-6)
        z = 1.0 + 2.0 * self.curvature * diff_sq / denom
        return torch.arccosh(z.clamp_min(1.0 + 1e-6))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.project(self.weight[indices])

    @torch.no_grad()
    def project_parameters_(self) -> None:
        self.weight.copy_(self.project(self.weight))


class HyperbolicDescriptorEncoder(nn.Module):
    """Learned descriptor encoder that maps categorical descriptor fields into a tangent-space representation."""

    def __init__(self, config: HyperbolicDescriptorConfig):
        super().__init__()
        self.config = config
        self.family_vocab = {"__unk__": 0, **{name: i + 1 for i, name in enumerate(sorted(VALID_EXPERT_FAMILIES))}}
        self.projection_vocab = {"__unk__": 0, **{name: i + 1 for i, name in enumerate(sorted(VALID_PROJECTION_KINDS))}}
        self.anchor_vocab = {"False": 0, "True": 1}
        dim = config.embedding_dim
        kwargs = {"curvature": config.curvature, "max_norm": config.max_norm}
        self.family_embed = PoincareBallEmbedding(len(self.family_vocab), dim, **kwargs)
        self.projection_embed = PoincareBallEmbedding(len(self.projection_vocab), dim, **kwargs)
        self.anchor_embed = PoincareBallEmbedding(len(self.anchor_vocab), dim, **kwargs)

    def _lookup_indices(self, descriptors: tuple[ViewDescriptor, ...], attr: str, vocab: dict[str, int]) -> torch.Tensor:
        values = [str(getattr(descriptor, attr, "")) for descriptor in descriptors]
        indices = [vocab.get(value, 0) for value in values]
        return torch.tensor(indices, dtype=torch.long, device=self.family_embed.weight.device)

    def forward(self, descriptors: tuple[ViewDescriptor, ...]) -> tuple[torch.Tensor, tuple[str, ...]]:
        family_idx = self._lookup_indices(descriptors, "family", self.family_vocab)
        projection_idx = self._lookup_indices(descriptors, "projection_kind", self.projection_vocab)
        anchor_idx = torch.tensor(
            [self.anchor_vocab[str(bool(descriptor.is_anchor))] for descriptor in descriptors],
            dtype=torch.long,
            device=self.family_embed.weight.device,
        )
        numeric = torch.tensor(
            [[float(descriptor.input_dim), float(descriptor.preferred_k), float(descriptor.is_anchor)] for descriptor in descriptors],
            dtype=torch.float32,
            device=self.family_embed.weight.device,
        )
        family_tangent = self.family_embed.log_map_zero(self.family_embed(family_idx))
        projection_tangent = self.projection_embed.log_map_zero(self.projection_embed(projection_idx))
        anchor_tangent = self.anchor_embed.log_map_zero(self.anchor_embed(anchor_idx))
        descriptor_tensor = torch.cat([numeric, family_tangent, projection_tangent, anchor_tangent], dim=-1)
        dim = self.config.embedding_dim
        names = (
            "input_dim",
            "preferred_k",
            "is_anchor",
            *[f"family_h{i}" for i in range(dim)],
            *[f"projection_h{i}" for i in range(dim)],
            *[f"anchor_h{i}" for i in range(dim)],
        )
        return descriptor_tensor, names

    @torch.no_grad()
    def project_parameters_(self) -> None:
        self.family_embed.project_parameters_()
        self.projection_embed.project_parameters_()
        self.anchor_embed.project_parameters_()
