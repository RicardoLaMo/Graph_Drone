from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .config import PortfolioLoadConfig
from .view_descriptor import ViewDescriptor, normalize_descriptor_set


class ExpertPredictor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ConstantPredictor:
    value: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float32)


@dataclass(frozen=True)
class LinearPredictor:
    coefficients: np.ndarray
    bias: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        coef = np.asarray(self.coefficients, dtype=np.float32).reshape(-1)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != coef.shape[0]:
            raise ValueError(
                f"Linear predictor expected X shape [N,{coef.shape[0]}], got {tuple(X.shape)}"
            )
        return (X @ coef + self.bias).astype(np.float32)


def _default_input_adapter(descriptor: ViewDescriptor) -> Callable[[np.ndarray], np.ndarray]:
    if descriptor.projection_kind != "identity_subselect":
        raise ValueError(
            f"Expert {descriptor.expert_id!r} requires an explicit adapter for projection_kind="
            f"{descriptor.projection_kind!r}"
        )

    indices = np.asarray(descriptor.input_indices, dtype=np.int64)

    def project(X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")
        if len(indices) == 0:
            raise ValueError(f"Descriptor {descriptor.expert_id!r} has no input indices")
        if indices.max(initial=-1) >= matrix.shape[1]:
            raise ValueError(
                f"Descriptor {descriptor.expert_id!r} expects at least {indices.max() + 1} features, "
                f"got {matrix.shape[1]}"
            )
        return matrix[:, indices]

    return project


@dataclass
class LoadedExpert:
    descriptor: ViewDescriptor
    predictor: ExpertPredictor
    artifact_kind: str
    input_adapter: Callable[[np.ndarray], np.ndarray] | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        adapter = self.input_adapter or _default_input_adapter(self.descriptor)
        view_matrix = adapter(X)
        
        # Classification vs Regression branch
        if self.artifact_kind == "foundation_classifier" and hasattr(self.predictor, "predict_proba"):
            # Multi-class [N, C] or Binary [N, 2]
            pred = self.predictor.predict_proba(view_matrix)
        elif self.artifact_kind == "tabpfn_classifier" and hasattr(self.predictor, "predict_proba"):
            pred = self.predictor.predict_proba(view_matrix)
        else:
            pred = self.predictor.predict(view_matrix)
            
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
            
        if pred.shape[0] != len(X):
            raise ValueError(
                f"Expert {self.descriptor.expert_id!r} returned {pred.shape[0]} predictions for {len(X)} rows"
            )
        return pred


@dataclass(frozen=True)
class LoadedPortfolio:
    expert_order: tuple[str, ...]
    experts: dict[str, LoadedExpert]
    full_expert_id: str
    manifest_path: Path | None = None

    @property
    def descriptors(self) -> tuple[ViewDescriptor, ...]:
        return tuple(self.experts[expert_id].descriptor for expert_id in self.expert_order)

    def validate(self) -> "LoadedPortfolio":
        if not self.expert_order:
            raise ValueError("LoadedPortfolio requires at least one expert")
        if self.full_expert_id not in self.experts:
            raise ValueError(f"full_expert_id={self.full_expert_id!r} is missing from experts")
        normalize_descriptor_set(list(self.descriptors), required_anchor_id=self.full_expert_id)
        return self


def load_portfolio(config: PortfolioLoadConfig, *, full_expert_id: str | None = None) -> LoadedPortfolio:
    manifest_path = config.resolved_manifest_path()
    payload = json.loads(manifest_path.read_text())
    manifest_full_id = str(payload.get("full_expert_id", full_expert_id or "FULL"))
    effective_full_id = full_expert_id or manifest_full_id
    expert_payloads = payload.get("experts", [])
    if not expert_payloads:
        raise ValueError(f"No experts found in manifest {manifest_path}")

    experts: dict[str, LoadedExpert] = {}
    expert_order: list[str] = []
    for item in expert_payloads:
        descriptor = ViewDescriptor.from_dict(item["descriptor"])
        artifact = item["artifact"]
        predictor = _load_predictor(artifact)
        expert = LoadedExpert(
            descriptor=descriptor,
            predictor=predictor,
            artifact_kind=str(artifact["kind"]),
        )
        experts[descriptor.expert_id] = expert
        expert_order.append(descriptor.expert_id)

    return LoadedPortfolio(
        expert_order=tuple(expert_order),
        experts=experts,
        full_expert_id=effective_full_id,
        manifest_path=manifest_path,
    ).validate()


def _load_predictor(artifact: dict[str, object]) -> ExpertPredictor:
    kind = str(artifact["kind"])
    if kind == "constant":
        return ConstantPredictor(value=float(artifact["value"]))
    if kind == "linear":
        coefficients = np.asarray(artifact["coefficients"], dtype=np.float32)
        bias = float(artifact.get("bias", 0.0))
        return LinearPredictor(coefficients=coefficients, bias=bias)
    raise ValueError(f"Unsupported artifact kind={kind!r}")
