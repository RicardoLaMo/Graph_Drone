from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GraphDroneConfig
from .defer_integrator import IntegrationOutputs, integrate_predictions
from .expert_factory import ExpertBuildSpec, ExpertPredictionBatch, PortfolioExpertFactory, fit_portfolio_from_specs
from .portfolio_loader import LoadedPortfolio, load_portfolio
from .set_router import build_set_router
from .support_encoder import SupportEncoding, ZeroSupportEncoder
from .token_builder import PerViewTokenBuilder, TokenBatch


@dataclass(frozen=True)
class GraphDronePredictResult:
    predictions: np.ndarray
    diagnostics: dict[str, object]
    expert_ids: tuple[str, ...]
    token_shape: tuple[int, int, int]


class GraphDrone:
    def __init__(self, config: GraphDroneConfig) -> None:
        self.config = config.validate()
        self._portfolio: LoadedPortfolio | None = None
        self._expert_factory: PortfolioExpertFactory | None = None
        self._token_builder: PerViewTokenBuilder | None = None
        self._support_encoder: ZeroSupportEncoder | None = None
        self._router = None
        self.n_features_in_: int | None = None

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
                y_train=np.asarray(y, dtype=np.float32),
                specs=expert_specs,
                full_expert_id=self.config.full_expert_id,
            )
        else:
            if self.config.portfolio is None:
                raise ValueError(
                    "GraphDroneConfig.portfolio is required when portfolio and expert_specs are not provided"
                )
            self._portfolio = load_portfolio(self.config.portfolio, full_expert_id=self.config.full_expert_id)
        self._expert_factory = PortfolioExpertFactory(self._portfolio)
        self._token_builder = PerViewTokenBuilder()
        self._support_encoder = ZeroSupportEncoder()
        self._router = build_set_router(self.config.router)
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        quality_features: np.ndarray | None = None,
        support_tensor: np.ndarray | None = None,
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
        quality_features: np.ndarray | None = None,
        support_tensor: np.ndarray | None = None,
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
            diagnostics=_build_diagnostics(
                batch=batch,
                tokens=tokens,
                support_encoding=support_encoding,
                integration=integration,
            ),
            expert_ids=batch.expert_ids,
            token_shape=tuple(int(v) for v in tokens.tokens.shape),
        )


def _build_diagnostics(
    *,
    batch,
    tokens: TokenBatch,
    support_encoding: SupportEncoding,
    integration: IntegrationOutputs,
) -> dict[str, object]:
    return {
        "full_expert_id": batch.full_expert_id,
        "expert_ids": list(batch.expert_ids),
        "token_field_slices": {key: list(value) for key, value in tokens.field_slices.items()},
        "support_feature_names": list(support_encoding.feature_names),
        **integration.diagnostics,
    }


def _coerce_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {matrix.shape}")
    return matrix
