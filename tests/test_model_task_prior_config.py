from __future__ import annotations

from graphdrone_fit.config import GraphDroneConfig, SetRouterConfig
from graphdrone_fit.model import GraphDrone


def test_binary_classification_router_config_preserves_task_prior_fields() -> None:
    model = GraphDrone(
        GraphDroneConfig(
            router=SetRouterConfig(
                kind="bootstrap_full_only",
                task_prior_bank_dir="/tmp/task-bank",
                task_prior_encoder_kind="transformer",
                task_prior_mode="routing_bias",
                task_prior_strength=0.7,
                task_prior_local_gate_alpha=3.0,
                task_prior_dataset_key="credit_g",
                task_prior_exact_reuse_blend=0.8,
                task_prior_defer_penalty_lambda=0.6,
                task_prior_defer_target=0.65,
                task_prior_rank_loss_lambda=0.15,
                task_prior_rank_margin=0.05,
            )
        )
    )
    use_learned, cfg = model._classification_router_config(is_binary=True)
    assert use_learned is True
    assert cfg is not None
    assert cfg.kind == "noise_gate_router"
    assert cfg.task_prior_bank_dir == "/tmp/task-bank"
    assert cfg.task_prior_encoder_kind == "transformer"
    assert cfg.task_prior_mode == "routing_bias"
    assert cfg.task_prior_strength == 0.7
    assert cfg.task_prior_local_gate_alpha == 3.0
    assert cfg.task_prior_dataset_key == "credit_g"
    assert cfg.task_prior_exact_reuse_blend == 0.8
    assert cfg.task_prior_defer_penalty_lambda == 0.6
    assert cfg.task_prior_defer_target == 0.65
    assert cfg.task_prior_rank_loss_lambda == 0.15
    assert cfg.task_prior_rank_margin == 0.05
