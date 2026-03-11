from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .config import PortfolioLoadConfig, TaskType
from .view_descriptor import ViewDescriptor, normalize_descriptor_set


class ExpertPredictor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


class ClassificationPredictor(ExpertPredictor, Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
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


@dataclass(frozen=True)
class ConstantClassifierPredictor:
    class_probabilities: np.ndarray
    class_labels: tuple[int, ...]

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        label_indices = probabilities.argmax(axis=1)
        return np.asarray(self.class_labels, dtype=np.int64)[label_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.asarray(self.class_probabilities, dtype=np.float32).reshape(1, -1)
        return np.repeat(probs, len(X), axis=0).astype(np.float32)


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
    task_type: TaskType = "regression"
    class_labels: tuple[int, ...] | None = None
    input_adapter: Callable[[np.ndarray], np.ndarray] | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        adapter = self.input_adapter or _default_input_adapter(self.descriptor)
        view_matrix = adapter(X)
        pred = self.predictor.predict(view_matrix)
        if self.task_type == "classification":
            pred = np.asarray(pred, dtype=np.int64).reshape(-1)
        else:
            pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        if pred.shape[0] != len(X):
            raise ValueError(
                f"Expert {self.descriptor.expert_id!r} returned {pred.shape[0]} predictions for {len(X)} rows"
            )
        return pred

    def predict_values(self, X: np.ndarray) -> np.ndarray:
        adapter = self.input_adapter or _default_input_adapter(self.descriptor)
        view_matrix = adapter(X)
        if self.task_type == "classification":
            predictor = self.predictor
            if not hasattr(predictor, "predict_proba"):
                raise ValueError(
                    f"Expert {self.descriptor.expert_id!r} does not support predict_proba()"
                )
            proba = np.asarray(predictor.predict_proba(view_matrix), dtype=np.float32)
            if proba.ndim != 2:
                raise ValueError(
                    f"Expert {self.descriptor.expert_id!r} returned probability shape {proba.shape}"
                )
            if proba.shape[0] != len(X):
                raise ValueError(
                    f"Expert {self.descriptor.expert_id!r} returned {proba.shape[0]} probability rows for {len(X)} rows"
                )
            return proba
        return self.predict(X)


@dataclass(frozen=True)
class LoadedPortfolio:
    expert_order: tuple[str, ...]
    experts: dict[str, LoadedExpert]
    full_expert_id: str
    task_type: TaskType = "regression"
    class_labels: tuple[int, ...] | None = None
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
        if self.task_type == "classification":
            if not self.class_labels:
                raise ValueError("Classification portfolios require non-empty class_labels")
            for expert_id in self.expert_order:
                expert = self.experts[expert_id]
                if expert.task_type != "classification":
                    raise ValueError(
                        f"Classification portfolio expert {expert_id!r} has task_type={expert.task_type!r}"
                    )
                if tuple(expert.class_labels or ()) != tuple(self.class_labels):
                    raise ValueError(
                        f"Expert {expert_id!r} class_labels do not match portfolio class_labels"
                    )
        else:
            for expert_id in self.expert_order:
                expert = self.experts[expert_id]
                if expert.task_type != "regression":
                    raise ValueError(
                        f"Regression portfolio expert {expert_id!r} has task_type={expert.task_type!r}"
                    )
        return self


def load_portfolio(config: PortfolioLoadConfig, *, full_expert_id: str | None = None) -> LoadedPortfolio:
    manifest_path = config.resolved_manifest_path()
    payload = json.loads(manifest_path.read_text())
    manifest_full_id = str(payload.get("full_expert_id", full_expert_id or "FULL"))
    effective_full_id = full_expert_id or manifest_full_id
    task_type = str(payload.get("task_type", "regression"))
    class_labels = payload.get("class_labels")
    effective_class_labels = (
        tuple(int(v) for v in class_labels) if class_labels is not None else None
    )
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
            task_type=task_type,
            class_labels=effective_class_labels,
        )
        experts[descriptor.expert_id] = expert
        expert_order.append(descriptor.expert_id)

    return LoadedPortfolio(
        expert_order=tuple(expert_order),
        experts=experts,
        full_expert_id=effective_full_id,
        task_type=task_type,
        class_labels=effective_class_labels,
        manifest_path=manifest_path,
    ).validate()


def _load_predictor(artifact: dict[str, object]) -> ExpertPredictor:
    kind = str(artifact["kind"])
    if kind == "constant":
        return ConstantPredictor(value=float(artifact["value"]))
    if kind == "constant_classifier":
        return ConstantClassifierPredictor(
            class_probabilities=np.asarray(artifact["class_probabilities"], dtype=np.float32),
            class_labels=tuple(int(v) for v in artifact["class_labels"]),
        )
    if kind == "linear":
        coefficients = np.asarray(artifact["coefficients"], dtype=np.float32)
        bias = float(artifact.get("bias", 0.0))
        return LinearPredictor(coefficients=coefficients, bias=bias)
    raise ValueError(f"Unsupported artifact kind={kind!r}")
