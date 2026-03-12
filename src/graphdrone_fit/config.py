from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


TaskType = Literal["regression", "classification"]
ClassificationBlendMode = Literal["probability"]


@dataclass(frozen=True)
class PortfolioLoadConfig:
    manifest_path: Path
    strict: bool = True

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path.expanduser().resolve()


@dataclass(frozen=True)
class SetRouterConfig:
    kind: Literal["bootstrap_full_only", "contextual_sparse_mlp"] = "bootstrap_full_only"
    field_aware_tokens: bool = True
    sparse_top_k: int = 1
    hidden_dim: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    validation_fraction: float = 0.2
    defer_bias: float = -2.0
    random_seed: int = 42

    def validate(self) -> "SetRouterConfig":
        if self.sparse_top_k < 1:
            raise ValueError(f"sparse_top_k must be positive, got {self.sparse_top_k}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.patience < 1:
            raise ValueError(f"patience must be positive, got {self.patience}")
        if not 0.0 <= self.validation_fraction < 0.5:
            raise ValueError(
                f"validation_fraction must be in [0.0, 0.5), got {self.validation_fraction}"
            )
        return self


@dataclass(frozen=True)
class GraphDroneConfig:
    portfolio: PortfolioLoadConfig | None = None
    full_expert_id: str = "FULL"
    task_type: TaskType = "regression"
    class_labels: tuple[str | int, ...] | None = None
    classification_blend_mode: ClassificationBlendMode = "probability"
    router: SetRouterConfig = field(default_factory=SetRouterConfig)

    def validate(self) -> "GraphDroneConfig":
        if not self.full_expert_id.strip():
            raise ValueError("full_expert_id must be non-empty")
        if self.task_type not in {"regression", "classification"}:
            raise ValueError(f"Unsupported task_type={self.task_type!r}")
        if self.task_type == "classification" and self.class_labels is not None and len(self.class_labels) < 2:
            raise ValueError("classification GraphDroneConfig requires at least two class_labels")
        if self.classification_blend_mode != "probability":
            raise ValueError(
                "classification_blend_mode currently supports only 'probability'"
            )
        self.router.validate()
        if self.portfolio is not None:
            self.portfolio.resolved_manifest_path()
        return self
