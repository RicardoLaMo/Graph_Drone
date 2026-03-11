from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GraphDroneConfig
from .defer_integrator import IntegrationOutputs, integrate_predictions
from .expert_factory import ExpertBuildSpec, ExpertPredictionBatch, PortfolioExpertFactory, fit_portfolio_from_specs
from .portfolio_loader import LoadedPortfolio, load_portfolio
from .set_router import build_set_router
from .support_encoder import MomentSupportEncoder, SupportEncoding
from .token_builder import PerViewTokenBuilder, QualityEncoding, TokenBatch


@dataclass(frozen=True)
class GraphDronePredictResult:
    predictions: np.ndarray
    probabilities: np.ndarray | None
    diagnostics: dict[str, object]
    expert_ids: tuple[str, ...]
    token_shape: tuple[int, int, int]
    class_labels: tuple[int, ...] | None = None


class GraphDrone:
    def __init__(self, config: GraphDroneConfig) -> None:
        self.config = config.validate()
        self._portfolio: LoadedPortfolio | None = None
        self._expert_factory: PortfolioExpertFactory | None = None
        self._token_builder: PerViewTokenBuilder | None = None
        self._support_encoder: MomentSupportEncoder | None = None
        self._router = None
        self._router_fit_summary: dict[str, object] = {}
        self.n_features_in_: int | None = None
        self.task_type = self.config.task_type
        self.class_labels_: tuple[int, ...] | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        portfolio: LoadedPortfolio | None = None,
        expert_specs: tuple[ExpertBuildSpec, ...] | None = None,
    ) -> "GraphDrone":
        matrix = _coerce_matrix(X)
        self.n_features_in_ = matrix.shape[1]
        if portfolio is not None and expert_specs is not None:
            raise ValueError("Provide either portfolio or expert_specs, not both")
        if portfolio is not None:
            self._portfolio = portfolio
        elif expert_specs is not None:
            if y is None:
                raise ValueError("y is required when fitting GraphDrone from expert_specs")
            self._portfolio = fit_portfolio_from_specs(
                X_train=matrix,
                y_train=np.asarray(y),
                specs=expert_specs,
                full_expert_id=self.config.full_expert_id,
                task_type=self.config.task_type,
            )
        else:
            if self.config.portfolio is None:
                raise ValueError(
                    "GraphDroneConfig.portfolio is required when portfolio and expert_specs are not provided"
                )
            self._portfolio = load_portfolio(self.config.portfolio, full_expert_id=self.config.full_expert_id)
        self._expert_factory = PortfolioExpertFactory(self._portfolio)
        self.task_type = self._portfolio.task_type
        self.class_labels_ = self._portfolio.class_labels
        self._token_builder = PerViewTokenBuilder()
        self._support_encoder = MomentSupportEncoder()
        self._router = build_set_router(self.config.router)
        self._router_fit_summary = {}
        return self

    def fit_router(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        quality_features: np.ndarray | QualityEncoding | None = None,
        support_tensor: np.ndarray | SupportEncoding | None = None,
    ) -> dict[str, object]:
        if self._expert_factory is None or self._token_builder is None or self._support_encoder is None or self._router is None:
            raise RuntimeError("GraphDrone.fit() must be called before fit_router()")
        matrix = _coerce_matrix(X)
        if self.n_features_in_ is not None and matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features after fit(), got {matrix.shape[1]}"
            )
        if self.task_type == "classification":
            target = np.asarray(y, dtype=np.int64).reshape(-1)
        else:
            target = np.asarray(y, dtype=np.float32).reshape(-1)
        if target.shape[0] != matrix.shape[0]:
            raise ValueError(f"Expected y with {matrix.shape[0]} rows, got {target.shape[0]}")
        batch = self._expert_factory.predict_all(matrix)
        support_encoding = self._support_encoder.encode(
            n_rows=matrix.shape[0],
            descriptors=batch.descriptors,
            support_tensor=support_tensor,
        )
        tokens = self._token_builder.build(
            predictions=batch.predictions,
            descriptors=batch.descriptors,
            full_expert_id=batch.full_expert_id,
            quality_features=quality_features,
            support_encoding=support_encoding,
        )
        fit_fn = getattr(self._router, "fit_router", None)
        if fit_fn is None:
            raise RuntimeError("Configured router does not support fit_router()")
        summary = fit_fn(
            tokens.tokens,
            np.asarray(batch.predictions, dtype=np.float32),
            target,
            full_index=batch.full_index,
        )
        self._router_fit_summary = dict(summary)
        return dict(self._router_fit_summary)

    def predict(
        self,
        X: np.ndarray,
        *,
        quality_features: np.ndarray | QualityEncoding | None = None,
        support_tensor: np.ndarray | SupportEncoding | None = None,
        return_diagnostics: bool = False,
    ) -> np.ndarray | GraphDronePredictResult:
        result = self.predict_with_diagnostics(
            X,
            quality_features=quality_features,
            support_tensor=support_tensor,
        )
        if return_diagnostics:
            return result
        return result.predictions

    def predict_proba(
        self,
        X: np.ndarray,
        *,
        quality_features: np.ndarray | QualityEncoding | None = None,
        support_tensor: np.ndarray | SupportEncoding | None = None,
    ) -> np.ndarray:
        if self.task_type != "classification":
            raise RuntimeError("predict_proba() is available only for classification GraphDrone models")
        result = self.predict_with_diagnostics(
            X,
            quality_features=quality_features,
            support_tensor=support_tensor,
        )
        if result.probabilities is None:
            raise RuntimeError("Classification GraphDrone predictions did not include probabilities")
        return result.probabilities

    def predict_experts(self, X: np.ndarray) -> ExpertPredictionBatch:
        if self._expert_factory is None:
            raise RuntimeError("GraphDrone.fit() must be called before predict_experts()")
        matrix = _coerce_matrix(X)
        if self.n_features_in_ is not None and matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features after fit(), got {matrix.shape[1]}"
            )
        return self._expert_factory.predict_all(matrix)

    def predict_with_diagnostics(
        self,
        X: np.ndarray,
        *,
        quality_features: np.ndarray | QualityEncoding | None = None,
        support_tensor: np.ndarray | SupportEncoding | None = None,
    ) -> GraphDronePredictResult:
        if self._expert_factory is None or self._token_builder is None or self._support_encoder is None or self._router is None:
            raise RuntimeError("GraphDrone.fit() must be called before predict()")

        matrix = _coerce_matrix(X)
        if self.n_features_in_ is not None and matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features after fit(), got {matrix.shape[1]}"
            )

        batch = self._expert_factory.predict_all(matrix)
        support_encoding = self._support_encoder.encode(
            n_rows=matrix.shape[0],
            descriptors=batch.descriptors,
            support_tensor=support_tensor,
        )
        tokens = self._token_builder.build(
            predictions=batch.predictions,
            descriptors=batch.descriptors,
            full_expert_id=batch.full_expert_id,
            quality_features=quality_features,
            support_encoding=support_encoding,
        )
        router_outputs = self._router(tokens.tokens, full_index=batch.full_index)
        integration = integrate_predictions(
            expert_predictions=batch.predictions,
            router_outputs=router_outputs,
        )
        return GraphDronePredictResult(
            predictions=integration.predictions,
            probabilities=integration.probabilities,
            diagnostics=_build_diagnostics(
                batch=batch,
                tokens=tokens,
                support_encoding=support_encoding,
                integration=integration,
                router_fit_summary=self._router_fit_summary,
            ),
            expert_ids=batch.expert_ids,
            token_shape=tuple(int(v) for v in tokens.tokens.shape),
            class_labels=batch.class_labels,
        )


def _build_diagnostics(
    *,
    batch,
    tokens: TokenBatch,
    support_encoding: SupportEncoding,
    integration: IntegrationOutputs,
    router_fit_summary: dict[str, object],
) -> dict[str, object]:
    router_summary = dict(router_fit_summary)
    if "active_specialist_ids" not in router_summary:
        active_indices = router_summary.get("active_specialist_indices")
        if isinstance(active_indices, list):
            router_summary["active_specialist_ids"] = [
                batch.expert_ids[index]
                for index in active_indices
                if isinstance(index, int) and 0 <= index < len(batch.expert_ids)
            ]
    return {
        "full_expert_id": batch.full_expert_id,
        "task_type": batch.task_type,
        "class_labels": list(batch.class_labels) if batch.class_labels is not None else None,
        "expert_ids": list(batch.expert_ids),
        "token_field_slices": {key: list(value) for key, value in tokens.field_slices.items()},
        "token_field_names": {key: list(value) for key, value in tokens.field_names.items()},
        "quality_feature_names": list(tokens.field_names.get("quality", ())),
        "support_feature_names": list(support_encoding.feature_names),
        "router_fit_summary": router_summary,
        **integration.diagnostics,
    }


def _coerce_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {matrix.shape}")
    return matrix
