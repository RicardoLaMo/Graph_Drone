from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SetRouterConfig


@dataclass(frozen=True)
class RouterOutputs:
    specialist_weights: torch.Tensor
    defer_prob: torch.Tensor
    full_index: int
    router_kind: str


class BootstrapFullRouter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
        """
        Explicit Phase I-A bootstrap router.

        This is intentionally a non-learning placeholder so the package surface, token shape,
        and defer contract can be validated before contextual routing is implemented.
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor [N, E, D], got {tuple(tokens.shape)}")
        n_rows, n_experts, _ = tokens.shape
        specialist_weights = torch.zeros((n_rows, n_experts), dtype=tokens.dtype, device=tokens.device)
        defer_prob = torch.zeros((n_rows, 1), dtype=tokens.dtype, device=tokens.device)
        return RouterOutputs(
            specialist_weights=specialist_weights,
            defer_prob=defer_prob,
            full_index=full_index,
            router_kind="bootstrap_full_only",
        )


def build_set_router(config: SetRouterConfig) -> torch.nn.Module:
    config.validate()
    if config.kind == "bootstrap_full_only":
        return BootstrapFullRouter()
    raise ValueError(f"Unsupported router kind={config.kind!r}")
