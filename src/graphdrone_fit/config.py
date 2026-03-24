from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal


ROUTER_KIND_ALIASES = {
    "contextual_transformer_router": "contextual_transformer",
    "cross_attention_set_router": "contextual_transformer",
    "hyper_set_router": "contextual_transformer",
}
VALID_ROUTER_KINDS = frozenset(
    {
        "bootstrap_full_only",
        "contextual_transformer",
        "noise_gate_router",
        "contextual_transformer_rotor",
        "noise_gate_router_rotor",
        "contextual_transformer_ot_gate",
        "ot_noise_gate_router",
    }
)


@dataclass(frozen=True)
class PortfolioLoadConfig:
    manifest_path: Path
    strict: bool = True

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path.expanduser().resolve()


@dataclass(frozen=True)
class SetRouterConfig:
    kind: Literal[
        "bootstrap_full_only",
        "contextual_transformer",
        "noise_gate_router",
        "contextual_transformer_rotor",
        "noise_gate_router_rotor",
        "contextual_transformer_ot_gate",
        "ot_noise_gate_router",
    ] = "bootstrap_full_only"
    sparse_top_k: int = 1
    alignment_lambda: float = 0.0
    residual_usefulness_lambda: float = 0.0
    allocation_usefulness_lambda: float = 0.0
    conservative_allocation_lambda: float = 0.0
    conservative_allocation_opportunity_threshold: float = 0.1
    robust_allocation_usefulness_lambda: float = 0.0
    router_seed: int = 42
    freeze_base_router: bool = False
    ot_prototype_count: int = 32
    ot_epsilon: float = 0.05
    ot_max_iter: int = 50
    ot_alpha: float = 6.0
    ot_threshold: float = 0.25
    task_prior_bank_dir: str | None = None
    task_prior_encoder_kind: Literal["transformer", "gru"] = "transformer"
    task_prior_mode: Literal["anchor_shift", "routing_bias"] = "anchor_shift"
    task_prior_strength: float = 0.5
    task_prior_local_gate_alpha: float = 0.0
    task_prior_expert_local_gate_alpha: float = 0.0
    task_prior_row_expert_opportunity_alpha: float = 0.0
    task_prior_dataset_key: str | None = None
    task_prior_exact_reuse_blend: float = 0.5
    task_prior_defer_penalty_lambda: float = 0.0
    task_prior_defer_target: float = 0.8
    task_prior_rank_loss_lambda: float = 0.0
    task_prior_rank_margin: float = 0.0

    def validate(self) -> "SetRouterConfig":
        normalized_kind = ROUTER_KIND_ALIASES.get(self.kind, self.kind)
        if normalized_kind not in VALID_ROUTER_KINDS:
            raise ValueError(
                f"Unsupported router kind={self.kind!r}; expected one of "
                f"{sorted(VALID_ROUTER_KINDS | set(ROUTER_KIND_ALIASES))}"
            )
        if self.sparse_top_k < 1:
            raise ValueError(f"sparse_top_k must be positive, got {self.sparse_top_k}")
        if self.alignment_lambda < 0:
            raise ValueError(f"alignment_lambda must be non-negative, got {self.alignment_lambda}")
        if self.residual_usefulness_lambda < 0:
            raise ValueError(
                f"residual_usefulness_lambda must be non-negative, got {self.residual_usefulness_lambda}"
            )
        if self.allocation_usefulness_lambda < 0:
            raise ValueError(
                f"allocation_usefulness_lambda must be non-negative, got {self.allocation_usefulness_lambda}"
            )
        if self.conservative_allocation_lambda < 0:
            raise ValueError(
                f"conservative_allocation_lambda must be non-negative, got {self.conservative_allocation_lambda}"
            )
        if self.conservative_allocation_opportunity_threshold < 0:
            raise ValueError(
                "conservative_allocation_opportunity_threshold must be non-negative, got "
                f"{self.conservative_allocation_opportunity_threshold}"
            )
        if self.robust_allocation_usefulness_lambda < 0:
            raise ValueError(
                f"robust_allocation_usefulness_lambda must be non-negative, got {self.robust_allocation_usefulness_lambda}"
            )
        if self.router_seed < 0:
            raise ValueError(f"router_seed must be non-negative, got {self.router_seed}")
        if self.ot_prototype_count < 1:
            raise ValueError(f"ot_prototype_count must be positive, got {self.ot_prototype_count}")
        if self.ot_epsilon <= 0:
            raise ValueError(f"ot_epsilon must be positive, got {self.ot_epsilon}")
        if self.ot_max_iter < 1:
            raise ValueError(f"ot_max_iter must be positive, got {self.ot_max_iter}")
        if self.task_prior_strength < 0:
            raise ValueError(f"task_prior_strength must be non-negative, got {self.task_prior_strength}")
        if self.task_prior_local_gate_alpha < 0:
            raise ValueError(
                f"task_prior_local_gate_alpha must be non-negative, got {self.task_prior_local_gate_alpha}"
            )
        if self.task_prior_expert_local_gate_alpha < 0:
            raise ValueError(
                "task_prior_expert_local_gate_alpha must be non-negative, got "
                f"{self.task_prior_expert_local_gate_alpha}"
            )
        if self.task_prior_row_expert_opportunity_alpha < 0:
            raise ValueError(
                "task_prior_row_expert_opportunity_alpha must be non-negative, got "
                f"{self.task_prior_row_expert_opportunity_alpha}"
            )
        if self.task_prior_mode not in {"anchor_shift", "routing_bias"}:
            raise ValueError(
                f"task_prior_mode must be one of ['anchor_shift', 'routing_bias'], got {self.task_prior_mode!r}"
            )
        if not 0.0 <= self.task_prior_exact_reuse_blend <= 1.0:
            raise ValueError(
                f"task_prior_exact_reuse_blend must be in [0, 1], got {self.task_prior_exact_reuse_blend}"
            )
        if self.task_prior_defer_penalty_lambda < 0:
            raise ValueError(
                f"task_prior_defer_penalty_lambda must be non-negative, got {self.task_prior_defer_penalty_lambda}"
            )
        if not 0.0 <= self.task_prior_defer_target <= 1.0:
            raise ValueError(
                f"task_prior_defer_target must be in [0, 1], got {self.task_prior_defer_target}"
            )
        if self.task_prior_rank_loss_lambda < 0:
            raise ValueError(
                f"task_prior_rank_loss_lambda must be non-negative, got {self.task_prior_rank_loss_lambda}"
            )
        if self.task_prior_rank_margin < 0:
            raise ValueError(
                f"task_prior_rank_margin must be non-negative, got {self.task_prior_rank_margin}"
            )
        if self.task_prior_bank_dir is not None and not str(self.task_prior_bank_dir).strip():
            raise ValueError("task_prior_bank_dir must be non-empty when provided")
        if self.task_prior_dataset_key is not None and not str(self.task_prior_dataset_key).strip():
            raise ValueError("task_prior_dataset_key must be non-empty when provided")
        return replace(self, kind=normalized_kind)


@dataclass(frozen=True)
class LegitimacyGateConfig:
    enabled: bool = True
    regression_enabled: bool = True
    binary_enabled: bool = False
    multiclass_enabled: bool = False
    classification_entropy_threshold: float = 0.15
    regression_variance_threshold: float = 0.005

    def validate(self) -> "LegitimacyGateConfig":
        if not 0.0 <= self.classification_entropy_threshold <= 1.0:
            raise ValueError(
                "classification_entropy_threshold must be in [0, 1], got "
                f"{self.classification_entropy_threshold}"
            )
        if self.regression_variance_threshold < 0.0:
            raise ValueError(
                f"regression_variance_threshold must be non-negative, got {self.regression_variance_threshold}"
            )
        return self


@dataclass(frozen=True)
class HyperbolicDescriptorConfig:
    enabled: bool = False
    embedding_dim: int = 4
    curvature: float = 1.0
    max_norm: float = 0.95

    def validate(self) -> "HyperbolicDescriptorConfig":
        if self.embedding_dim < 1:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.curvature <= 0:
            raise ValueError(f"curvature must be positive, got {self.curvature}")
        if not 0.0 < self.max_norm < 1.0:
            raise ValueError(f"max_norm must be in (0, 1), got {self.max_norm}")
        return self


@dataclass(frozen=True)
class GraphDroneConfig:
    portfolio: PortfolioLoadConfig | None = None
    full_expert_id: str = "FULL"
    router: SetRouterConfig = field(default_factory=SetRouterConfig)
    legitimacy_gate: LegitimacyGateConfig = field(default_factory=LegitimacyGateConfig)
    hyperbolic_descriptors: HyperbolicDescriptorConfig = field(default_factory=HyperbolicDescriptorConfig)
    # n_classes > 1 forces classification mode and pins the output dimension.
    # Leave at 1 for regression or binary auto-detection.
    n_classes: int = 1
    # When True, trains a ContextualTransformerRouter for classification with
    # NLL loss + residual anchor penalty (same protective mechanism as regression).
    # When False, falls back to static anchor_geo_poe_blend.
    use_learned_router_for_classification: bool = True

    def validate(self) -> "GraphDroneConfig":
        if not self.full_expert_id.strip():
            raise ValueError("full_expert_id must be non-empty")
        router = self.router.validate()
        legitimacy_gate = self.legitimacy_gate.validate()
        hyperbolic_descriptors = self.hyperbolic_descriptors.validate()
        if self.portfolio is not None:
            self.portfolio.resolved_manifest_path()
        return replace(
            self,
            router=router,
            legitimacy_gate=legitimacy_gate,
            hyperbolic_descriptors=hyperbolic_descriptors,
        )
