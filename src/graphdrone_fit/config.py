from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PortfolioLoadConfig:
    manifest_path: Path
    strict: bool = True

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path.expanduser().resolve()


@dataclass(frozen=True)
class SetRouterConfig:
    kind: Literal["bootstrap_full_only", "contextual_transformer_router", "noise_gate_router"] = "bootstrap_full_only"
    sparse_top_k: int = 1

    def validate(self) -> "SetRouterConfig":
        if self.sparse_top_k < 1:
            raise ValueError(f"sparse_top_k must be positive, got {self.sparse_top_k}")
        return self


@dataclass(frozen=True)
class GraphDroneConfig:
    portfolio: PortfolioLoadConfig | None = None
    full_expert_id: str = "FULL"
    router: SetRouterConfig = field(default_factory=SetRouterConfig)
    problem_type: Literal["regression", "classification"] = "regression"
    n_classes: int = 1

    def validate(self) -> "GraphDroneConfig":
        if not self.full_expert_id.strip():
            raise ValueError("full_expert_id must be non-empty")
        self.router.validate()
        if self.problem_type == "classification" and self.n_classes < 2:
            raise ValueError(f"Classification requires at least 2 classes, got {self.n_classes}")
        if self.portfolio is not None:
            self.portfolio.resolved_manifest_path()
        return self
