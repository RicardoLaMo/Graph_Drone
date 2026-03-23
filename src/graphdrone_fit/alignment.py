from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class RotorAlignment(nn.Module):
    """Learnable orthogonal transform via the Cayley parameterization."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.n_params = dim * (dim - 1) // 2
        self.params = nn.Parameter(torch.zeros(self.n_params))

    def _build_skew(self) -> torch.Tensor:
        A = torch.zeros(self.dim, self.dim, device=self.params.device, dtype=self.params.dtype)
        idx = torch.triu_indices(self.dim, self.dim, offset=1, device=self.params.device)
        A[idx[0], idx[1]] = self.params
        A[idx[1], idx[0]] = -self.params
        return A

    def rotation_matrix(self) -> torch.Tensor:
        A = self._build_skew()
        eye = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        return torch.linalg.solve(eye + A, eye - A)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.rotation_matrix().T

    def alignment_loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self.forward(X), Y)


def sinkhorn_log(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    max_iter: int = 100,
    threshold: float = 1e-6,
) -> torch.Tensor:
    """Batched log-domain Sinkhorn transport plan."""

    if cost.ndim < 2:
        raise ValueError(f"cost must have at least 2 dimensions, got shape={tuple(cost.shape)}")

    *batch_shape, n_source, n_target = cost.shape
    device = cost.device
    dtype = cost.dtype
    log_mu = torch.full((*batch_shape, n_source), -math.log(float(n_source)), device=device, dtype=dtype)
    log_nu = torch.full((*batch_shape, n_target), -math.log(float(n_target)), device=device, dtype=dtype)

    log_k = -cost / epsilon
    log_u = torch.zeros_like(log_mu)
    log_v = torch.zeros_like(log_nu)

    for _ in range(max_iter):
        log_u_prev = log_u
        log_u = log_mu - torch.logsumexp(log_k + log_v.unsqueeze(-2), dim=-1)
        log_v = log_nu - torch.logsumexp(log_k + log_u.unsqueeze(-1), dim=-2)
        if torch.max(torch.abs(log_u - log_u_prev)).item() < threshold:
            break

    log_t = log_k + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return torch.exp(log_t)


def _select_prototypes(token_bank: torch.Tensor, prototype_count: int) -> torch.Tensor:
    """Reduce a token bank to a fixed prototype budget with deterministic k-means."""

    if token_bank.shape[0] <= prototype_count:
        return token_bank

    from sklearn.cluster import KMeans

    bank_np = token_bank.detach().cpu().numpy().astype(np.float32)
    model = KMeans(n_clusters=prototype_count, n_init=10, random_state=0)
    centers = model.fit(bank_np).cluster_centers_
    return torch.as_tensor(centers, dtype=token_bank.dtype)


class OTNoiseGate(nn.Module):
    """Noise gate driven by Sinkhorn transport cost to per-expert prototypes."""

    def __init__(
        self,
        token_dim: int,
        prototype_count: int = 32,
        epsilon: float = 0.05,
        max_iter: int = 50,
        alpha: float = 6.0,
        threshold: float = 0.25,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.prototype_count = prototype_count
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.threshold = threshold
        self.register_buffer("_prototypes", torch.empty(0), persistent=False)
        self.register_buffer("_prototype_counts", torch.empty(0, dtype=torch.long), persistent=False)

    @property
    def has_prototypes(self) -> bool:
        return self._prototypes.numel() > 0

    @torch.no_grad()
    def fit_prototypes(self, tokens: torch.Tensor, *, full_index: int) -> None:
        if tokens.ndim != 3:
            raise ValueError(f"Expected [N, E, D] tokens, got shape={tuple(tokens.shape)}")

        prototype_sets: list[torch.Tensor] = []
        counts: list[int] = []
        for expert_idx in range(tokens.shape[1]):
            bank = tokens[:, expert_idx, :].detach().cpu()
            prototypes = _select_prototypes(bank, self.prototype_count)
            prototype_sets.append(prototypes)
            counts.append(prototypes.shape[0])

        max_count = max(counts)
        padded = []
        for prototypes in prototype_sets:
            if prototypes.shape[0] < max_count:
                pad = prototypes[-1:].repeat(max_count - prototypes.shape[0], 1)
                padded.append(torch.cat([prototypes, pad], dim=0))
            else:
                padded.append(prototypes)

        self._prototypes = torch.stack(padded, dim=0)
        self._prototype_counts = torch.tensor(counts, dtype=torch.long)

    def forward(self, tokens: torch.Tensor, *, full_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_prototypes:
            validity = torch.ones((*tokens.shape[:2], 1), device=tokens.device, dtype=tokens.dtype)
            ot_cost = torch.zeros(tokens.shape[:2], device=tokens.device, dtype=tokens.dtype)
            return validity, ot_cost

        batch_size, n_experts, _ = tokens.shape
        anchor = tokens[:, full_index, :]
        validity_parts: list[torch.Tensor] = []
        cost_parts: list[torch.Tensor] = []
        prototypes = self._prototypes.to(tokens.device, dtype=tokens.dtype)
        prototype_counts = self._prototype_counts.to(tokens.device)

        for expert_idx in range(n_experts):
            if expert_idx == full_index:
                validity_parts.append(torch.ones((batch_size, 1), device=tokens.device, dtype=tokens.dtype))
                cost_parts.append(torch.zeros((batch_size,), device=tokens.device, dtype=tokens.dtype))
                continue

            count = int(prototype_counts[expert_idx].item())
            prototype_bank = prototypes[expert_idx, :count, :]
            source = torch.stack([tokens[:, expert_idx, :], anchor], dim=1)
            cost = torch.cdist(source.float(), prototype_bank.unsqueeze(0).expand(batch_size, -1, -1).float(), p=2).pow(2)
            transport = sinkhorn_log(cost, epsilon=self.epsilon, max_iter=self.max_iter)
            ot_cost = (transport * cost).sum(dim=(-1, -2)).to(tokens.dtype)
            validity = torch.sigmoid(-self.alpha * (ot_cost - self.threshold)).unsqueeze(-1)
            validity_parts.append(validity)
            cost_parts.append(ot_cost)

        return torch.stack(validity_parts, dim=1), torch.stack(cost_parts, dim=1)
