from __future__ import annotations

import os

from .config import (
    GraphDroneConfig,
    HyperbolicDescriptorConfig,
    LegitimacyGateConfig,
    SetRouterConfig,
)


_PRESET_ALIASES = {
    "current_env": "afc_candidate",
}
_VALID_PRESETS = frozenset({"v1_20_champion", "afc_candidate"})


def available_graphdrone_presets() -> tuple[str, ...]:
    return tuple(sorted(_VALID_PRESETS | set(_PRESET_ALIASES)))


def resolve_graphdrone_preset_name(preset: str) -> str:
    normalized = _PRESET_ALIASES.get(preset, preset)
    if normalized not in _VALID_PRESETS:
        raise ValueError(
            f"Unsupported GraphDrone preset={preset!r}; expected one of "
            f"{sorted(_VALID_PRESETS | set(_PRESET_ALIASES))}"
        )
    return normalized


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def build_graphdrone_config_from_preset(
    *,
    preset: str,
    n_classes: int = 1,
    default_router_kind: str,
    task_prior_dataset_key: str | None = None,
) -> GraphDroneConfig:
    resolved_preset = resolve_graphdrone_preset_name(preset)

    if resolved_preset == "v1_20_champion":
        return GraphDroneConfig(
            n_classes=n_classes,
            router=SetRouterConfig(kind=default_router_kind, router_seed=42),
            legitimacy_gate=LegitimacyGateConfig(
                enabled=False,
                regression_enabled=False,
                binary_enabled=False,
                multiclass_enabled=False,
            ),
            hyperbolic_descriptors=HyperbolicDescriptorConfig(enabled=False),
        )

    return GraphDroneConfig(
        n_classes=n_classes,
        router=SetRouterConfig(
            kind=os.getenv("GRAPHDRONE_ROUTER_KIND", default_router_kind),
            alignment_lambda=_env_float("GRAPHDRONE_ALIGNMENT_LAMBDA", 0.1),
            residual_usefulness_lambda=_env_float("GRAPHDRONE_RESIDUAL_USEFULNESS_LAMBDA", 0.0),
            allocation_usefulness_lambda=_env_float("GRAPHDRONE_ALLOCATION_USEFULNESS_LAMBDA", 0.0),
            conservative_allocation_lambda=_env_float("GRAPHDRONE_CONSERVATIVE_ALLOCATION_LAMBDA", 0.0),
            conservative_allocation_opportunity_threshold=_env_float(
                "GRAPHDRONE_CONSERVATIVE_ALLOCATION_OPPORTUNITY_THRESHOLD", 0.1
            ),
            robust_allocation_usefulness_lambda=_env_float("GRAPHDRONE_ROBUST_ALLOCATION_USEFULNESS_LAMBDA", 0.0),
            router_seed=_env_int("GRAPHDRONE_ROUTER_SEED", 42),
            freeze_base_router=_env_flag("GRAPHDRONE_FREEZE_BASE_ROUTER", False),
            ot_prototype_count=_env_int("GRAPHDRONE_OT_PROTOTYPE_COUNT", 32),
            ot_epsilon=_env_float("GRAPHDRONE_OT_EPSILON", 0.05),
            ot_max_iter=_env_int("GRAPHDRONE_OT_MAX_ITER", 50),
            ot_alpha=_env_float("GRAPHDRONE_OT_ALPHA", 6.0),
            ot_threshold=_env_float("GRAPHDRONE_OT_THRESHOLD", 0.25),
            task_prior_bank_dir=_env_str("GRAPHDRONE_TASK_PRIOR_BANK_DIR", None),
            task_prior_encoder_kind=_env_str("GRAPHDRONE_TASK_PRIOR_ENCODER_KIND", "transformer") or "transformer",
            task_prior_mode=_env_str("GRAPHDRONE_TASK_PRIOR_MODE", "anchor_shift") or "anchor_shift",
            task_prior_strength=_env_float("GRAPHDRONE_TASK_PRIOR_STRENGTH", 0.5),
            task_prior_local_gate_alpha=_env_float("GRAPHDRONE_TASK_PRIOR_LOCAL_GATE_ALPHA", 0.0),
            task_prior_expert_local_gate_alpha=_env_float("GRAPHDRONE_TASK_PRIOR_EXPERT_LOCAL_GATE_ALPHA", 0.0),
            task_prior_dataset_key=task_prior_dataset_key or _env_str("GRAPHDRONE_TASK_PRIOR_DATASET_KEY", None),
            task_prior_exact_reuse_blend=_env_float("GRAPHDRONE_TASK_PRIOR_EXACT_REUSE_BLEND", 0.5),
            task_prior_defer_penalty_lambda=_env_float("GRAPHDRONE_TASK_PRIOR_DEFER_PENALTY_LAMBDA", 0.0),
            task_prior_defer_target=_env_float("GRAPHDRONE_TASK_PRIOR_DEFER_TARGET", 0.8),
            task_prior_rank_loss_lambda=_env_float("GRAPHDRONE_TASK_PRIOR_RANK_LOSS_LAMBDA", 0.0),
            task_prior_rank_margin=_env_float("GRAPHDRONE_TASK_PRIOR_RANK_MARGIN", 0.0),
        ),
        legitimacy_gate=LegitimacyGateConfig(
            enabled=_env_flag("GRAPHDRONE_ENABLE_LEGITIMACY_GATE", True),
            regression_enabled=_env_flag("GRAPHDRONE_ENABLE_GATE_REGRESSION", True),
            binary_enabled=_env_flag("GRAPHDRONE_ENABLE_GATE_BINARY", False),
            multiclass_enabled=_env_flag("GRAPHDRONE_ENABLE_GATE_MULTICLASS", False),
            classification_entropy_threshold=_env_float("GRAPHDRONE_GATE_ENTROPY_THRESHOLD", 0.15),
            regression_variance_threshold=_env_float("GRAPHDRONE_GATE_VARIANCE_THRESHOLD", 0.005),
        ),
        hyperbolic_descriptors=HyperbolicDescriptorConfig(
            enabled=_env_flag("GRAPHDRONE_ENABLE_HYPERBOLIC_DESCRIPTORS", False),
            embedding_dim=_env_int("GRAPHDRONE_HYPERBOLIC_EMBEDDING_DIM", 4),
            curvature=_env_float("GRAPHDRONE_HYPERBOLIC_CURVATURE", 1.0),
            max_norm=_env_float("GRAPHDRONE_HYPERBOLIC_MAX_NORM", 0.95),
        ),
    )
