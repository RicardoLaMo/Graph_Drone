from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment import OTNoiseGate, RotorAlignment
from .config import LegitimacyGateConfig, SetRouterConfig


@dataclass(frozen=True)
class RouterOutputs:
    specialist_weights: torch.Tensor
    defer_prob: torch.Tensor
    full_index: int
    router_kind: str
    aux_loss: torch.Tensor | None = None
    ot_costs: torch.Tensor | None = None
    extra_diagnostics: dict[str, float] | None = None


@dataclass(frozen=True)
class LegitimacyGateDecision:
    exit_mask: np.ndarray
    scores: np.ndarray
    threshold: float
    metric: str


class LegitimacyGate:
    """Entropy / variance gate that can bypass routing for confident anchor rows."""

    def __init__(self, config: LegitimacyGateConfig):
        self.config = config.validate()

    def evaluate(
        self,
        *,
        problem_type: str,
        anchor_predictions: np.ndarray,
        expert_predictions: np.ndarray,
        quality_scores: np.ndarray | None,
    ) -> LegitimacyGateDecision:
        if problem_type == "classification":
            probs = np.clip(np.asarray(anchor_predictions, dtype=np.float32), 1e-9, 1.0)
            entropy = -np.sum(probs * np.log(probs), axis=-1).astype(np.float32)
            max_entropy = float(np.log(probs.shape[-1])) if probs.shape[-1] > 1 else 1.0
            max_entropy = max(max_entropy, 1e-6)
            scores = (entropy / max_entropy).astype(np.float32)
            threshold = self.config.classification_entropy_threshold
            metric = "normalized_entropy"
        else:
            if quality_scores is not None:
                scores = np.asarray(quality_scores[:, 0, 0], dtype=np.float32)
                metric = "anchor_bag_variance"
            else:
                scores = np.var(np.asarray(expert_predictions, dtype=np.float32), axis=1).astype(np.float32)
                metric = "cross_expert_variance"
            threshold = self.config.regression_variance_threshold
        return LegitimacyGateDecision(
            exit_mask=scores <= threshold,
            scores=scores,
            threshold=threshold,
            metric=metric,
        )


class BootstrapFullRouter(nn.Module):
    def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
        n_rows, n_experts, _ = tokens.shape
        specialist_weights = torch.zeros((n_rows, n_experts), dtype=tokens.dtype, device=tokens.device)
        defer_prob = torch.zeros((n_rows, 1), dtype=tokens.dtype, device=tokens.device)
        diagnostics = {
            "n_experts": float(n_experts),
            "n_specialists": float(max(n_experts - 1, 0)),
        }
        return RouterOutputs(
            specialist_weights,
            defer_prob,
            full_index,
            "bootstrap_full_only",
            extra_diagnostics=diagnostics,
        )


class LearnedNoiseGate(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(token_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, *, full_index: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.model(tokens), None


class ContextualTransformerRouter(nn.Module):
    """
    Anchor-query router with pluggable validity gating.

    The router operates on [B, E, D] token sets. Optional validity gates can
    suppress unreliable specialists before the anchor attends over the set.
    """

    def __init__(
        self,
        token_dim: int,
        *,
        hidden_dim: int = 64,
        noise_gate_module: nn.Module | None = None,
        router_kind: str = "contextual_transformer_router",
    ):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5
        self.noise_gate_module = noise_gate_module
        self.router_kind = router_kind

        self.q_proj = nn.Linear(token_dim, hidden_dim)
        self.k_proj = nn.Linear(token_dim, hidden_dim)
        self.defer_head = nn.Sequential(nn.Linear(token_dim, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.constant_(self.defer_head[-1].bias, -3.0)

    @torch.no_grad()
    def fit_auxiliary_state(self, tokens: torch.Tensor, *, full_index: int) -> None:
        if hasattr(self.noise_gate_module, "fit_prototypes"):
            self.noise_gate_module.fit_prototypes(tokens, full_index=full_index)

    def _apply_noise_gate(
        self,
        tokens: torch.Tensor,
        *,
        full_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.noise_gate_module is None:
            return tokens, None, None
        validity, ot_costs = self.noise_gate_module(tokens, full_index=full_index)
        return tokens * validity, ot_costs, validity

    def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
        gated_tokens, ot_costs, validity = self._apply_noise_gate(tokens, full_index=full_index)
        anchor_token = gated_tokens[:, full_index, :]
        q = self.q_proj(anchor_token)
        k = self.k_proj(gated_tokens)
        attn_scores = torch.einsum("bh,beh->be", q, k) * self.scale
        specialist_weights = torch.softmax(attn_scores, dim=-1)
        defer_prob = torch.sigmoid(self.defer_head(anchor_token))
        diagnostics = {
            "n_experts": float(tokens.shape[1]),
            "n_specialists": float(max(tokens.shape[1] - 1, 0)),
        }
        if validity is not None and tokens.shape[1] > 1:
            specialist_validity = torch.cat(
                [
                    validity[:, :full_index, :],
                    validity[:, full_index + 1 :, :],
                ],
                dim=1,
            )
            diagnostics.update(
                {
                    "mean_specialist_validity": float(specialist_validity.mean().item()),
                    "closed_specialist_frac": float((specialist_validity < 0.5).float().mean().item()),
                }
            )
        return RouterOutputs(
            specialist_weights=specialist_weights,
            defer_prob=defer_prob,
            full_index=full_index,
            router_kind=self.router_kind,
            ot_costs=ot_costs,
            extra_diagnostics=diagnostics,
        )


class RotorAlignedRouter(nn.Module):
    """Wrap a contextual router with per-specialist Cayley rotors."""

    def __init__(
        self,
        token_dim: int,
        *,
        n_experts: int,
        base_router: ContextualTransformerRouter,
        alignment_lambda: float,
        router_kind: str,
    ):
        super().__init__()
        self.base_router = base_router
        self.alignment_lambda = alignment_lambda
        self.router_kind = router_kind
        self.rotors = nn.ModuleList([RotorAlignment(token_dim) for _ in range(max(n_experts - 1, 0))])

    @torch.no_grad()
    def fit_auxiliary_state(self, tokens: torch.Tensor, *, full_index: int) -> None:
        self.base_router.fit_auxiliary_state(tokens, full_index=full_index)

    def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
        aligned = tokens.clone()
        anchor = tokens[:, full_index, :]
        losses: list[torch.Tensor] = []
        pre_cosines: list[torch.Tensor] = []
        post_cosines: list[torch.Tensor] = []
        rotor_idx = 0
        for expert_idx in range(tokens.shape[1]):
            if expert_idx == full_index:
                continue
            rotor = self.rotors[rotor_idx]
            rotated = rotor(tokens[:, expert_idx, :])
            aligned[:, expert_idx, :] = rotated
            losses.append(rotor.alignment_loss(tokens[:, expert_idx, :], anchor.detach()))
            pre_cosines.append(F.cosine_similarity(tokens[:, expert_idx, :], anchor, dim=-1).mean())
            post_cosines.append(F.cosine_similarity(rotated, anchor, dim=-1).mean())
            rotor_idx += 1

        outputs = self.base_router(aligned, full_index=full_index)
        aux_loss = None
        diagnostics = dict(outputs.extra_diagnostics or {})
        if losses:
            aux_loss = self.alignment_lambda * torch.stack(losses).mean()
            pre = torch.stack(pre_cosines).mean()
            post = torch.stack(post_cosines).mean()
            diagnostics.update(
                {
                    "alignment_cosine_pre": float(pre.item()),
                    "alignment_cosine_post": float(post.item()),
                    "alignment_cosine_gain": float((post - pre).item()),
                }
            )
        else:
            diagnostics.update(
                {
                    "alignment_cosine_pre": float("nan"),
                    "alignment_cosine_post": float("nan"),
                    "alignment_cosine_gain": float("nan"),
                }
            )
        return replace(
            outputs,
            router_kind=self.router_kind,
            aux_loss=aux_loss,
            extra_diagnostics=diagnostics,
        )


def _make_noise_gate(config: SetRouterConfig, token_dim: int) -> nn.Module | None:
    if config.kind == "noise_gate_router" or config.kind == "noise_gate_router_rotor":
        return LearnedNoiseGate(token_dim)
    if config.kind == "ot_noise_gate_router" or config.kind == "contextual_transformer_ot_gate":
        return OTNoiseGate(
            token_dim=token_dim,
            prototype_count=config.ot_prototype_count,
            epsilon=config.ot_epsilon,
            max_iter=config.ot_max_iter,
            alpha=config.ot_alpha,
            threshold=config.ot_threshold,
        )
    return None


def build_set_router(config: SetRouterConfig, token_dim: int = 14, n_experts: int = 1) -> torch.nn.Module:
    config = config.validate()
    if config.kind == "bootstrap_full_only":
        return BootstrapFullRouter()

    router_kind = {
        "contextual_transformer": "contextual_transformer_router",
        "noise_gate_router": "noise_gate_router",
        "contextual_transformer_ot_gate": "contextual_transformer_ot_gate",
        "ot_noise_gate_router": "ot_noise_gate_router",
    }.get(config.kind, config.kind)

    base_router = ContextualTransformerRouter(
        token_dim=token_dim,
        noise_gate_module=_make_noise_gate(config, token_dim),
        router_kind=router_kind,
    )

    if config.kind in {"contextual_transformer_rotor", "noise_gate_router_rotor"}:
        return RotorAlignedRouter(
            token_dim=token_dim,
            n_experts=n_experts,
            base_router=base_router,
            alignment_lambda=config.alignment_lambda,
            router_kind=config.kind,
        )
    return base_router


class TaskConditionedRouter(nn.Module):
    """Inject a dataset-level prior embedding into the anchor token before routing.

    This is a clean wrapper around any base router: when no task prior has been
    set (or strength=0), it delegates to the base router unmodified.  When a
    prior context is set via set_task_prior_context(), it shifts the anchor token
    by a learned linear projection of the prior before passing to the base router.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        prior_dim: int,
        base_router: nn.Module,
        strength: float,
        router_kind: str,
    ):
        super().__init__()
        self.base_router = base_router
        self.strength = strength
        self.router_kind = router_kind
        self.prior_to_token = nn.Linear(prior_dim, token_dim)
        self.register_buffer("_task_prior_context", torch.zeros(prior_dim), persistent=False)
        self._has_task_prior = False

    @torch.no_grad()
    def set_task_prior_context(self, context: torch.Tensor) -> None:
        context = torch.as_tensor(context, dtype=self._task_prior_context.dtype, device=self._task_prior_context.device)
        if context.ndim != 1:
            raise ValueError(f"task prior context must be 1D, got shape={tuple(context.shape)}")
        if context.shape[0] != self._task_prior_context.shape[0]:
            raise ValueError(
                f"task prior context dim mismatch: expected {self._task_prior_context.shape[0]}, got {context.shape[0]}"
            )
        self._task_prior_context.copy_(context)
        self._has_task_prior = True

    @torch.no_grad()
    def fit_auxiliary_state(self, tokens: torch.Tensor, *, full_index: int) -> None:
        if hasattr(self.base_router, "fit_auxiliary_state"):
            self.base_router.fit_auxiliary_state(tokens, full_index=full_index)

    def forward(self, tokens: torch.Tensor, *, full_index: int) -> RouterOutputs:
        if not self._has_task_prior or self.strength <= 0:
            outputs = self.base_router(tokens, full_index=full_index)
            diagnostics = dict(outputs.extra_diagnostics or {})
            diagnostics.update({"task_prior_enabled": 0.0})
            return replace(outputs, router_kind=self.router_kind, extra_diagnostics=diagnostics)

        conditioned = tokens.clone()
        prior_shift = self.prior_to_token(self._task_prior_context).to(tokens.device)
        conditioned[:, full_index, :] = conditioned[:, full_index, :] + self.strength * prior_shift.unsqueeze(0)
        outputs = self.base_router(conditioned, full_index=full_index)
        diagnostics = dict(outputs.extra_diagnostics or {})
        diagnostics.update(
            {
                "task_prior_enabled": 1.0,
                "task_prior_strength": float(self.strength),
                "task_prior_norm": float(self._task_prior_context.norm().item()),
            }
        )
        return replace(outputs, router_kind=self.router_kind, extra_diagnostics=diagnostics)
