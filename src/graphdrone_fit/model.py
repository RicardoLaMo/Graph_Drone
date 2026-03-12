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
    diagnostics: dict[str, object]
    expert_ids: tuple[str, ...]
    token_shape: tuple[int, int, int]


class GraphDrone:
    """
    GraphDrone Meta-Model for Tabular Data.
    
    A Mixture-of-Experts architecture that integrates multiple foundation models
    (specialists) across different feature subspaces using a contextual set-router.
    
    Parameters
    ----------
    config : GraphDroneConfig
        Configuration for the model, including the full expert ID and router settings.
    """
    def __init__(self, config: GraphDroneConfig) -> None:
        self.config = config.validate()
        self._portfolio: LoadedPortfolio | None = None
        self._expert_factory: PortfolioExpertFactory | None = None
        self._token_builder: PerViewTokenBuilder | None = None
        self._support_encoder: MomentSupportEncoder | None = None
        self._router = None
        self.n_features_in_: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        expert_specs: tuple[ExpertBuildSpec, ...] | None = None,
    ) -> "GraphDrone":
        """
        Fit the GraphDrone specialists and the meta-router.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        expert_specs : tuple of ExpertBuildSpec, optional
            Definitions for the expert views. If None, a default full-view 
            expert will be used.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        matrix = _coerce_matrix(X)
        self.n_features_in_ = matrix.shape[1]
        
        # Default spec if none provided
        if expert_specs is None:
            full_idx = tuple(range(self.n_features_in_))
            expert_specs = (
                ExpertBuildSpec(
                    descriptor=ViewDescriptor(
                        expert_id=self.config.full_expert_id,
                        family="FULL",
                        view_name="Full dataset (Default)",
                        is_anchor=True,
                        input_dim=self.n_features_in_,
                        input_indices=full_idx
                    ),
                    model_kind="foundation_regressor", # Auto-detect in full version
                    input_adapter=IdentitySelectorAdapter(indices=full_idx)
                ),
            )

        self._portfolio = fit_portfolio_from_specs(
            X_train=matrix,
            y_train=np.asarray(y, dtype=np.float32),
            specs=expert_specs,
            full_expert_id=self.config.full_expert_id,
        )
        
        self._expert_factory = PortfolioExpertFactory(self._portfolio)
        self._token_builder = PerViewTokenBuilder()
        self._support_encoder = MomentSupportEncoder()
        self._router = build_set_router(self.config.router)
        return self

    def predict(
        self,
        X: np.ndarray,
        return_diagnostics: bool = False,
    ) -> np.ndarray | GraphDronePredictResult:
        """
        Predict target for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        return_diagnostics : bool, default=False
            Whether to return routing weights and SNR diagnostics.
            
        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted values.
        """
        result = self.predict_with_diagnostics(X)
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
    ) -> GraphDronePredictResult:
        if self._expert_factory is None or self._token_builder is None or self._support_encoder is None or self._router is None:
            raise RuntimeError("GraphDrone.fit() must be called before predict()")

        matrix = _coerce_matrix(X)
        if self.n_features_in_ is not None and matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features after fit(), got {matrix.shape[1]}"
            )

        batch = self._expert_factory.predict_all(matrix)
        
        # Automatic Support and Quality calculation (Internalized)
        # In a real sklearn package, these are handled inside predict()
        support_encoding = self._support_encoder.encode(
            n_rows=matrix.shape[0],
            descriptors=batch.descriptors,
            support_tensor=None, # Inferred in full implementation
        )
        tokens = self._token_builder.build(
            predictions=batch.predictions,
            descriptors=batch.descriptors,
            full_expert_id=batch.full_expert_id,
            quality_features=None,
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
        "token_field_names": {key: list(value) for key, value in tokens.field_names.items()},
        "quality_feature_names": list(tokens.field_names.get("quality", ())),
        "support_feature_names": list(support_encoding.feature_names),
        **integration.diagnostics,
    }


def _coerce_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {matrix.shape}")
    return matrix
