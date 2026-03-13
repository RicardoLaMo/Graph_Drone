from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from typing import Optional, Union

from .config import GraphDroneConfig
from .defer_integrator import IntegrationOutputs, integrate_predictions
from .expert_factory import ExpertBuildSpec, ExpertPredictionBatch, PortfolioExpertFactory, fit_portfolio_from_specs, IdentitySelectorAdapter
from .portfolio_loader import LoadedPortfolio, load_portfolio
from .set_router import build_set_router
from .support_encoder import MomentSupportEncoder, SupportEncoding
from .token_builder import UniversalTokenBuilder, TokenBatch
from .view_descriptor import ViewDescriptor

@dataclass(frozen=True)
class GraphDronePredictResult:
    predictions: np.ndarray
    diagnostics: dict[str, object]
    expert_ids: tuple[str, ...]

def _coerce_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {matrix.shape}")
    return matrix

class GraphDrone:
    """
    Consolidated GraphDrone Meta-Model.
    """
    def __init__(self, config: GraphDroneConfig) -> None:
        self.config = config.validate()
        self._expert_factory: Optional[PortfolioExpertFactory] = None
        self._token_builder = UniversalTokenBuilder()
        self._support_encoder = MomentSupportEncoder()
        self._router: Optional[torch.nn.Module] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X: np.ndarray, y: np.ndarray, expert_specs: Optional[tuple[ExpertBuildSpec, ...]] = None) -> "GraphDrone":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        matrix = _coerce_matrix(X)
        
        # Default spec if none provided
        if expert_specs is None:
            full_idx = tuple(range(matrix.shape[1]))
            expert_specs = (
                ExpertBuildSpec(
                    descriptor=ViewDescriptor(
                        expert_id=self.config.full_expert_id,
                        family="FULL",
                        view_name="Full dataset (Default)",
                        is_anchor=True,
                        input_dim=matrix.shape[1],
                        input_indices=full_idx
                    ),
                    model_kind="foundation_regressor", 
                    input_adapter=IdentitySelectorAdapter(indices=full_idx)
                ),
            )

        # Fit specialists
        portfolio = fit_portfolio_from_specs(
            X_train=matrix, y_train=y, specs=expert_specs, full_expert_id=self.config.full_expert_id
        )
        self._expert_factory = PortfolioExpertFactory(portfolio)
        
        # Infer token dim and init router
        dummy_batch = self._expert_factory.predict_all(X[:1])
        dummy_enc = self._support_encoder.encode(n_rows=1, descriptors=dummy_batch.descriptors)
        dummy_tokens = self._token_builder.build(
            predictions=dummy_batch.predictions,
            descriptors=dummy_batch.descriptors,
            full_expert_id=dummy_batch.full_expert_id,
            support_encoding=dummy_enc
        )
        self._router = build_set_router(self.config.router, token_dim=dummy_tokens.tokens.shape[-1]).to(self.device)
        return self

    def predict(self, X: np.ndarray, return_diagnostics: bool = False) -> Union[np.ndarray, GraphDronePredictResult]:
        X = np.asarray(X, dtype=np.float32)
        matrix = _coerce_matrix(X)
        batch = self._expert_factory.predict_all(matrix)
        
        # Internalized Support/Token workflow
        support_enc = self._support_encoder.encode(n_rows=matrix.shape[0], descriptors=batch.descriptors)
        tokens = self._token_builder.build(
            predictions=batch.predictions,
            descriptors=batch.descriptors,
            full_expert_id=batch.full_expert_id,
            support_encoding=support_enc
        )
        
        self._router.eval()
        with torch.no_grad():
            # Ensure tokens on same device as router
            token_tensor = tokens.tokens.to(self.device)
            router_out = self._router(token_tensor, full_index=batch.full_index)
            integration = integrate_predictions(expert_predictions=batch.predictions, router_outputs=router_out)
            
        if return_diagnostics:
            return GraphDronePredictResult(
                predictions=integration.predictions,
                expert_ids=batch.expert_ids,
                diagnostics=integration.diagnostics
            )
        return integration.predictions
