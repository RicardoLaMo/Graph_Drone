from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .portfolio_loader import ConstantPredictor, LinearPredictor, LoadedExpert, LoadedPortfolio
from .view_descriptor import ViewDescriptor


@dataclass(frozen=True)
class ExpertPredictionBatch:
    expert_ids: tuple[str, ...]
    descriptors: tuple[ViewDescriptor, ...]
    predictions: np.ndarray
    full_expert_id: str
    full_index: int


class InputAdapterProtocol(Protocol):
    def fit(self, X: np.ndarray) -> "InputAdapterProtocol":
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass
class IdentitySelectorAdapter:
    indices: tuple[int, ...]

    def fit(self, X: np.ndarray) -> "IdentitySelectorAdapter":
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got {matrix.shape}")
        if len(self.indices) == 0:
            raise ValueError("IdentitySelectorAdapter requires at least one feature index")
        if max(self.indices) >= matrix.shape[1]:
            raise ValueError(
                f"IdentitySelectorAdapter expects at least {max(self.indices) + 1} features, got {matrix.shape[1]}"
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float32)
        return matrix[:, self.indices]


@dataclass
class PcaProjectionAdapter:
    n_components: int
    random_state: int = 42
    _pca: PCA | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray) -> "PcaProjectionAdapter":
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got {matrix.shape}")
        n_components = min(self.n_components, matrix.shape[0], matrix.shape[1])
        if n_components < 1:
            raise ValueError("PcaProjectionAdapter requires at least one component")
        self._pca = PCA(n_components=n_components, random_state=self.random_state)
        self._pca.fit(matrix)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PcaProjectionAdapter.fit() must be called before transform()")
        matrix = np.asarray(X, dtype=np.float32)
        return self._pca.transform(matrix).astype(np.float32)


@dataclass
class GeometryFeatureAdapter:
    indices: tuple[int, ...]
    feature_keys: tuple[str, ...]
    include_base_features: bool = True
    k_neighbors: int = 24
    _knn: NearestNeighbors | None = field(default=None, init=False, repr=False)
    _train_view: np.ndarray | None = field(default=None, init=False, repr=False)
    _train_density: np.ndarray | None = field(default=None, init=False, repr=False)
    _train_ref: np.ndarray | None = field(default=None, init=False, repr=False)
    _feature_mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _feature_std: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray) -> "GeometryFeatureAdapter":
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got {matrix.shape}")
        if len(self.indices) == 0:
            raise ValueError("GeometryFeatureAdapter requires at least one feature index")
        if max(self.indices) >= matrix.shape[1]:
            raise ValueError(
                f"GeometryFeatureAdapter expects at least {max(self.indices) + 1} features, got {matrix.shape[1]}"
            )
        view = matrix[:, self.indices].astype(np.float32)
        n_neighbors = min(max(self.k_neighbors + 1, 2), len(view))
        self._knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        self._knn.fit(view)
        self._train_view = view
        self._train_ref = matrix
        mean_distance = self._mean_knn_distance(view, drop_self=True)
        self._train_density = 1.0 / np.maximum(mean_distance, 1e-6)
        train_geometry = self._compute_geometry(view, drop_self=True)
        self._feature_mean = train_geometry.mean(axis=0).astype(np.float32)
        self._feature_std = np.maximum(train_geometry.std(axis=0), 1e-6).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._knn is None or self._train_view is None or self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("GeometryFeatureAdapter.fit() must be called before transform()")
        matrix = np.asarray(X, dtype=np.float32)
        view = matrix[:, self.indices].astype(np.float32)
        drop_self = self._train_ref is not None and X is self._train_ref
        geometry = self._compute_geometry(view, drop_self=drop_self)
        normalized_geometry = ((geometry - self._feature_mean) / self._feature_std).astype(np.float32)
        if not self.include_base_features:
            return normalized_geometry
        return np.concatenate([view, normalized_geometry], axis=1).astype(np.float32)

    def output_feature_names(self, base_feature_names: tuple[str, ...]) -> tuple[str, ...]:
        suffix_names = tuple(f"geometry_{name}" for name in self.feature_keys)
        if self.include_base_features:
            return tuple(base_feature_names) + suffix_names
        return suffix_names

    def output_dim(self, base_dim: int) -> int:
        return (base_dim if self.include_base_features else 0) + len(self.feature_keys)

    def _compute_geometry(self, view: np.ndarray, *, drop_self: bool) -> np.ndarray:
        if self._knn is None or self._train_view is None:
            raise RuntimeError("GeometryFeatureAdapter.fit() must be called before geometry computation")
        distances, indices = self._query_neighbors(view, drop_self=drop_self)
        distances = np.maximum(distances.astype(np.float32), 1e-6)

        features: list[np.ndarray] = []
        mean_distance = distances.mean(axis=1).astype(np.float32)
        if "lid" in self.feature_keys:
            r_max = distances[:, -1:]
            log_ratios = np.log(distances / r_max)
            denominator = np.sum(log_ratios, axis=1)
            safe_denominator = np.where(np.abs(denominator) < 1e-6, -1e-6, denominator)
            lid = np.clip(-distances.shape[1] / safe_denominator, 0.0, 100.0)
            features.append(lid.astype(np.float32))
        if "lof" in self.feature_keys:
            if self._train_density is None:
                raise RuntimeError("Training density cache is unavailable for LOF computation")
            query_density = 1.0 / np.maximum(mean_distance, 1e-6)
            neighbor_density = self._train_density[indices].mean(axis=1)
            lof = (neighbor_density / np.maximum(query_density, 1e-6)).astype(np.float32)
            features.append(lof)
        if "mean_knn_distance" in self.feature_keys:
            features.append(mean_distance)
        if not features:
            raise ValueError("GeometryFeatureAdapter requires at least one feature key")
        return np.stack(features, axis=1).astype(np.float32)

    def _mean_knn_distance(self, view: np.ndarray, *, drop_self: bool) -> np.ndarray:
        distances, _ = self._query_neighbors(view, drop_self=drop_self)
        distances = np.maximum(distances.astype(np.float32), 1e-6)
        return distances.mean(axis=1).astype(np.float32)

    def _query_neighbors(self, view: np.ndarray, *, drop_self: bool) -> tuple[np.ndarray, np.ndarray]:
        if self._knn is None or self._train_view is None:
            raise RuntimeError("GeometryFeatureAdapter.fit() must be called before neighbor queries")
        n_neighbors = min(max(self.k_neighbors + (1 if drop_self else 0), 1), len(self._train_view))
        distances, indices = self._knn.kneighbors(view, n_neighbors=n_neighbors, return_distance=True)
        if drop_self and distances.shape[1] > 1:
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        return distances, indices


@dataclass(frozen=True)
class ExpertBuildSpec:
    descriptor: ViewDescriptor
    model_kind: str
    input_adapter: InputAdapterProtocol
    model_params: dict[str, object] = field(default_factory=dict)


class PortfolioExpertFactory:
    def __init__(self, portfolio: LoadedPortfolio) -> None:
        self.portfolio = portfolio.validate()
        self.expert_ids = self.portfolio.expert_order
        self.descriptors = self.portfolio.descriptors
        self.full_expert_id = self.portfolio.full_expert_id
        self.full_index = self.expert_ids.index(self.full_expert_id)

    def predict_all(self, X: np.ndarray) -> ExpertPredictionBatch:
        preds = [
            self.portfolio.experts[expert_id].predict(X)
            for expert_id in self.expert_ids
        ]
        stacked = np.column_stack(preds).astype(np.float32)
        return ExpertPredictionBatch(
            expert_ids=self.expert_ids,
            descriptors=self.descriptors,
            predictions=stacked,
            full_expert_id=self.full_expert_id,
            full_index=self.full_index,
        )


def fit_portfolio_from_specs(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    specs: tuple[ExpertBuildSpec, ...],
    full_expert_id: str,
) -> LoadedPortfolio:
    matrix = np.asarray(X_train, dtype=np.float32)
    target = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError(f"Expected X_train to be 2D, got {matrix.shape}")
    if target.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"Expected y_train with {matrix.shape[0]} rows, got {target.shape[0]}"
        )

    experts: dict[str, LoadedExpert] = {}
    expert_order: list[str] = []
    for spec in specs:
        descriptor = spec.descriptor.validate()
        fitted_adapter = spec.input_adapter.fit(matrix)
        X_view = fitted_adapter.transform(matrix)
        predictor = _fit_predictor(
            model_kind=spec.model_kind,
            X_view=X_view,
            y_train=target,
            model_params=spec.model_params,
        )
        experts[descriptor.expert_id] = LoadedExpert(
            descriptor=descriptor,
            predictor=predictor,
            artifact_kind=spec.model_kind,
            input_adapter=fitted_adapter.transform,
        )
        expert_order.append(descriptor.expert_id)

    return LoadedPortfolio(
        expert_order=tuple(expert_order),
        experts=experts,
        full_expert_id=full_expert_id,
    ).validate()


def _fit_predictor(
    *,
    model_kind: str,
    X_view: np.ndarray,
    y_train: np.ndarray,
    model_params: dict[str, object],
):
    if model_kind == "constant":
        value = float(model_params.get("value", float(np.mean(y_train))))
        return ConstantPredictor(value=value)

    if model_kind == "linear":
        coef, *_ = np.linalg.lstsq(X_view, y_train, rcond=None)
        return LinearPredictor(coefficients=np.asarray(coef, dtype=np.float32), bias=0.0)

    if model_kind == "tabpfn_regressor":
        from tabpfn import TabPFNRegressor

        model = TabPFNRegressor(
            n_estimators=int(model_params.get("n_estimators", 1)),
            random_state=int(model_params.get("random_state", 42)),
            device=model_params.get("device", "auto"),
            ignore_pretraining_limits=bool(model_params.get("ignore_pretraining_limits", len(X_view) > 1000)),
            n_preprocessing_jobs=int(model_params.get("n_preprocessing_jobs", 1)),
        )
        model.fit(X_view, y_train)
        return model

    raise ValueError(f"Unsupported model_kind={model_kind!r}")
